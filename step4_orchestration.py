# =============================================================================
# 4-A: 8개 업무 Agent의 오케스트레이션
# =============================================================================
#
# 목적
# - 서류 검증 → 분석 4개 병렬 실행 → 종합의견 → 신청서 → 사전 점검
# - 서류 보완과 직원 검토 시 중단하고 같은 thread_id로 재개
# - 실패한 분석만 1회 재시도
# - 신청서 금액 오류 발생 시 초안 단계만 1회 재실행
# - SQLite 체크포인트로 프로세스 종료 후에도 상태 유지
#
# 범위
# - 가상 데이터와 Python 처리 로직을 사용하는 모의 업무 Agent입니다.
# - 이번 단계는 Bedrock, RAG, 실제 사용자 인증을 호출하지 않습니다.
# - 서류 검증은 제출 여부만 확인하며 서류 본문의 진위를 검증하지 않습니다.
# - 직원 확인은 초안 검토 완료를 의미하며 대출 승인이 아닙니다.
# - RAG·Gateway·Bedrock 연동은 4-B에서 추가합니다.
#
# 실행
#   uv run python step4_orchestration.py --scenario normal
#   uv run python step4_orchestration.py --scenario missing
#   uv run python step4_orchestration.py --scenario retry
#   uv run python step4_orchestration.py --scenario draft_error
#   uv run python step4_orchestration.py --thread 실행ID --resume supply
#   uv run python step4_orchestration.py --thread 실행ID --resume confirm
#   uv run python step4_orchestration.py --thread 실행ID --show
# =============================================================================

import argparse
import json
import operator
from pathlib import Path
from typing import Annotated, TypedDict
from uuid import uuid4

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt


ROOT = Path(__file__).resolve().parent
ANALYSTS = ("finance", "industry", "relations", "collateral")


def merge_dicts(current, update):
    # 병렬 노드는 자신의 키만 갱신합니다.
    # 재시도 시에는 해당 Agent의 이전 결과를 대체합니다.
    return {**current, **update}


class ReviewState(TypedDict):
    scenario: str
    documents: list[str]
    case: dict
    results: Annotated[dict, merge_dicts]
    attempts: Annotated[dict, merge_dicts]
    events: Annotated[list[str], operator.add]
    retry_count: int
    draft_attempts: int
    opinion: str
    draft: dict
    check_errors: list[str]
    status: str


def initial_state(scenario):
    documents = ["financial_statement", "cashflow"]

    if scenario == "missing":
        documents = ["financial_statement"]

    return {
        "scenario": scenario,
        "documents": documents,
        "case": {
            "case_id": "CASE_001",
            "unit": "백만원",
            "requested_amount": 1000,
            "revenue_previous": 10000,
            "revenue_current": 8500,
            "industry_note": "가상 산업자료: 원재료 가격 변동 확인 필요",
            "related_party_share_pct": 35,
            "collateral_value": 1200,
        },
        "results": {},
        "attempts": {},
        "events": [],
        "retry_count": 0,
        "draft_attempts": 0,
        "opinion": "",
        "draft": {},
        "check_errors": [],
        "status": "STARTED",
    }


def document_agent(state):
    required = {"financial_statement", "cashflow"}
    documents = set(state["documents"])
    missing = sorted(required - documents)

    if missing:
        # 중단 전에 외부 호출이나 DB 변경을 하지 않습니다.
        response = interrupt({
            "kind": "DOCUMENTS_REQUIRED",
            "missing": missing,
        })

        # CLI에서 가상 서류 제출을 모사합니다.
        if not isinstance(response, dict):
            raise ValueError("INVALID_DOCUMENT_RESPONSE")

        supplied = response.get("provided", [])
        if not isinstance(supplied, list) or not all(
            isinstance(item, str) for item in supplied
        ):
            raise ValueError("INVALID_DOCUMENT_LIST")

        documents.update(set(supplied) & required)

        if required - documents:
            raise ValueError("REQUIRED_DOCUMENTS_STILL_MISSING")

    return {
        "documents": sorted(documents),
        "status": "ANALYZING",
        "events": ["documents:PASS"],
    }


def analyze(state, name):
    attempt = state["attempts"].get(name, 0) + 1

    # 일시 장애를 재현: 산업 Agent의 첫 실행만 실패합니다.
    if (
        state["scenario"] == "retry"
        and name == "industry"
        and attempt == 1
    ):
        return {
            "results": {
                name: {"status": "FAILED", "reason": "SIMULATED_TIMEOUT"}
            },
            "attempts": {name: attempt},
            "events": [f"{name}:FAILED"],
        }

    case = state["case"]

    if name == "finance":
        growth = round(
            (case["revenue_current"] / case["revenue_previous"] - 1) * 100,
            2,
        )
        summary = f"매출 증감률은 {growth}%이다."

    elif name == "industry":
        summary = case["industry_note"]

    elif name == "relations":
        summary = (
            f"가상 관계사 거래 비중은 {case['related_party_share_pct']}%이다. "
            "거래 조건과 회수 현황 추가 확인이 필요하다."
        )

    elif name == "collateral":
        summary = (
            f"가상 담보 평가액은 {case['collateral_value']}백만원이다. "
            "선순위 권리와 처분 가능성은 별도 확인이 필요하다."
        )

    else:
        raise ValueError("UNKNOWN_AGENT")

    return {
        "results": {
            name: {
                "status": "PASS",
                "summary": summary,
                "source": "synthetic_case_fixture",
            }
        },
        "attempts": {name: attempt},
        "events": [f"{name}:PASS"],
    }


def finance_agent(state):
    return analyze(state, "finance")


def industry_agent(state):
    return analyze(state, "industry")


def relations_agent(state):
    return analyze(state, "relations")


def collateral_agent(state):
    return analyze(state, "collateral")


def failed_agents(state):
    return [
        name
        for name in ANALYSTS
        if state["results"].get(name, {}).get("status") != "PASS"
    ]


def analysis_gate(state):
    failed = failed_agents(state)

    if not failed:
        status = "ANALYSIS_COMPLETE"
    elif state["retry_count"] < 1:
        status = "RETRY_REQUIRED"
    else:
        status = "ANALYSIS_FAILED"

    return {
        "status": status,
        "events": [f"analysis_gate:{status}"],
    }


def route_analysis(state):
    if state["status"] == "ANALYSIS_COMPLETE":
        return "opinion"
    if state["status"] == "RETRY_REQUIRED":
        return "retry_failed"
    return END


def retry_failed(state):
    # 성공한 Agent는 호출하지 않고 기존 결과를 보존합니다.
    update = {
        "results": {},
        "attempts": {},
        "events": [],
        "retry_count": state["retry_count"] + 1,
    }

    for name in failed_agents(state):
        result = analyze(state, name)
        update["results"].update(result["results"])
        update["attempts"].update(result["attempts"])
        update["events"].extend(result["events"])

    return update


def opinion_agent(state):
    if failed_agents(state):
        raise ValueError("ANALYSIS_NOT_COMPLETE")

    opinion = "\n".join(
        state["results"][name]["summary"] for name in ANALYSTS
    )

    return {
        "opinion": opinion,
        "events": ["opinion:PASS"],
    }


def draft_agent(state):
    attempt = state["draft_attempts"] + 1
    amount = state["case"]["requested_amount"]

    # 초안 오류 수정 경로를 검증하기 위한 의도적 오류입니다.
    if state["scenario"] == "draft_error" and attempt == 1:
        amount += 1

    return {
        "draft": {
            "case_id": state["case"]["case_id"],
            "requested_amount": amount,
            "unit": state["case"]["unit"],
            "opinion": state["opinion"],
            "document_status": "DRAFT",
        },
        "draft_attempts": attempt,
        "events": [f"draft:CREATED:{attempt}"],
    }


def precheck_agent(state):
    errors = []

    if state["draft"]["requested_amount"] != state["case"]["requested_amount"]:
        errors.append("AMOUNT_MISMATCH")

    if failed_agents(state):
        errors.append("INCOMPLETE_ANALYSIS")

    if not errors:
        status = "PRECHECK_PASS"
    elif state["draft_attempts"] < 2:
        status = "DRAFT_REPAIR_REQUIRED"
    else:
        status = "PRECHECK_FAILED"

    return {
        "check_errors": errors,
        "status": status,
        "events": [f"precheck:{status}"],
    }


def route_precheck(state):
    if state["status"] == "PRECHECK_PASS":
        return "human_review"
    if state["status"] == "DRAFT_REPAIR_REQUIRED":
        return "draft"
    return END


def human_review(state):
    response = interrupt({
        "kind": "HUMAN_REVIEW",
        "case_id": state["case"]["case_id"],
        "message": "가상 신청서 초안을 검토하고 확인 또는 반려하세요.",
    })

    if not isinstance(response, dict):
        raise ValueError("INVALID_REVIEW_RESPONSE")

    action = response.get("action")
    if action not in ("confirm", "return"):
        raise ValueError("INVALID_REVIEW_ACTION")

    original_opinion = state["draft"]["opinion"]
    edited_opinion = response.get("edited_opinion", original_opinion)
    note = response.get("note", "")

    if (
        not isinstance(edited_opinion, str)
        or not edited_opinion.strip()
        or len(edited_opinion) > 30000
    ):
        raise ValueError("INVALID_EDITED_OPINION")

    if not isinstance(note, str) or len(note) > 3000:
        raise ValueError("INVALID_REVIEW_NOTE")

    # 검토 완료는 대출 승인 결정과 구분합니다.
    status = "REVIEWED" if action == "confirm" else "RETURNED"

    return {
        "status": status,
        "draft": {
            **state["draft"],
            "opinion": edited_opinion,
            "document_status": status,
            "review": {
                "action": action,
                "note": note,
                "original_opinion": original_opinion,
                "edited": edited_opinion != original_opinion,
                "reviewer": "local_demo_reviewer",
            },
        },
        "events": [f"human_review:{status}"],
    }


def build_graph(checkpointer):
    graph = StateGraph(ReviewState)

    graph.add_node("documents", document_agent)
    graph.add_node("finance", finance_agent)
    graph.add_node("industry", industry_agent)
    graph.add_node("relations", relations_agent)
    graph.add_node("collateral", collateral_agent)
    graph.add_node("analysis_gate", analysis_gate)
    graph.add_node("retry_failed", retry_failed)
    graph.add_node("opinion", opinion_agent)
    graph.add_node("draft", draft_agent)
    graph.add_node("precheck", precheck_agent)
    graph.add_node("human_review", human_review)

    graph.add_edge(START, "documents")

    for name in ANALYSTS:
        graph.add_edge("documents", name)

    # 리스트를 시작점으로 지정하면 네 노드가 모두 끝날 때까지 기다립니다.
    graph.add_edge(list(ANALYSTS), "analysis_gate")

    graph.add_conditional_edges(
        "analysis_gate",
        route_analysis,
        {
            "opinion": "opinion",
            "retry_failed": "retry_failed",
            END: END,
        },
    )
    graph.add_edge("retry_failed", "analysis_gate")
    graph.add_edge("opinion", "draft")
    graph.add_edge("draft", "precheck")

    graph.add_conditional_edges(
        "precheck",
        route_precheck,
        {
            "draft": "draft",
            "human_review": "human_review",
            END: END,
        },
    )

    graph.add_edge("human_review", END)
    return graph.compile(checkpointer=checkpointer)


def pending_interrupts(snapshot):
    return [
        item.value
        for task in snapshot.tasks
        for item in task.interrupts
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenario",
        choices=["normal", "missing", "retry", "draft_error"],
        default="normal",
    )
    parser.add_argument("--thread")
    parser.add_argument(
        "--resume",
        choices=["supply", "confirm", "return"],
    )
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    if (args.resume or args.show) and not args.thread:
        parser.error("--resume 또는 --show에는 --thread가 필요합니다.")

    thread_id = args.thread or uuid4().hex

    config = {
        "configurable": {"thread_id": thread_id},
        "max_concurrency": 4,
        "recursion_limit": 30,
    }

    runtime_dir = ROOT / ".runtime"
    runtime_dir.mkdir(exist_ok=True)

    with SqliteSaver.from_conn_string(
        str(runtime_dir / "step4.sqlite")
    ) as saver:
        graph = build_graph(saver)
        snapshot = graph.get_state(config)

        if args.show:
            if not snapshot.values:
                raise ValueError("THREAD_NOT_FOUND")

        elif args.resume:
            pending = pending_interrupts(snapshot)
            if len(pending) != 1:
                raise ValueError("SINGLE_PENDING_INTERRUPT_REQUIRED")

            kind = pending[0]["kind"]

            if args.resume == "supply":
                if kind != "DOCUMENTS_REQUIRED":
                    raise ValueError("NOT_WAITING_FOR_DOCUMENTS")
                response = {"provided": ["cashflow"]}
            else:
                if kind != "HUMAN_REVIEW":
                    raise ValueError("NOT_WAITING_FOR_REVIEW")
                response = {"action": args.resume}

            graph.invoke(Command(resume=response), config)

        else:
            if snapshot.values:
                raise ValueError("THREAD_ALREADY_EXISTS_USE_RESUME_OR_SHOW")
            graph.invoke(initial_state(args.scenario), config)

        snapshot = graph.get_state(config)

        report = {
            "thread_id": thread_id,
            "mode": "mock_orchestration",
            "status": snapshot.values.get("status"),
            "next_nodes": list(snapshot.next),
            "pending": pending_interrupts(snapshot),
            "attempts": snapshot.values.get("attempts", {}),
            "draft_attempts": snapshot.values.get("draft_attempts", 0),
            "events": snapshot.values.get("events", []),
            "draft": snapshot.values.get("draft", {}),
        }

        output_dir = ROOT / "outputs"
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / f"step4_{thread_id}.json"
        output_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        print(json.dumps(report, ensure_ascii=False, indent=2))
        print(f"\n저장 위치: {output_path}")


if __name__ == "__main__":
    main()