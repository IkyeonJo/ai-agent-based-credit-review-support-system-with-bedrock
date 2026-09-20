# =============================================================================
# 4-B: Multi-Agent + pgvector + Gateway + Bedrock 통합
# =============================================================================
#
# 목적
# - 재무·산업·관계사·담보 Agent를 실제 RAG 분석으로 교체합니다.
# - 4-A의 서류 검증, 종합의견 취합, 신청서, 사전 점검, 직원 검토를 재사용합니다.
# - Agent별 검색 근거, 검증된 응답, 사용량, 시간, 오류 분류를 기록합니다.
#
# 데이터
# - 모든 문서는 가상 자료·모의 규정입니다.
# - 기존 credit_chunks와 분리된 credit_agent_chunks 테이블에 저장합니다.
# - 각 역할에 기업자료 1개와 모의 규정 1개를 준비합니다.
#
# 실행 통제
# - 분석 노드는 병렬로 실행됩니다.
# - 초기 실습에서는 Bedrock 동시 호출을 1개로 제한합니다.
# - 일시적 통신·서비스 오류만 실패한 Agent에 한해 1회 재시도합니다.
# - 권한·전송 정책·출처 검증 오류는 자동 재시도하지 않습니다.
#
# 한계
# - 사용자 인증과 서버 권한은 고정된 실습 설정으로 모사합니다.
# - 개인정보 검사는 앞 단계에서 구현한 제한된 탐지 범위를 따릅니다.
# - 종합의견은 검증된 분석문을 취합하며 별도의 LLM 호출은 하지 않습니다.
# - 출처 ID·인용문 일치가 분석의 의미적 정확성까지 보장하지는 않습니다.
# - 결과와 체크포인트에는 검증된 가상 분석문이 저장됩니다.
#   실제 고객 데이터로 운영할 때는 별도 접근·암호화·보관 정책이 필요합니다.
#
# 실행
#   uv run python step4b_integrated_agents.py --seed
#   uv run python step4b_integrated_agents.py
#   uv run python step4b_integrated_agents.py --live
#   uv run python step4b_integrated_agents.py --thread 실행ID --show
#   uv run python step4b_integrated_agents.py --thread 실행ID --confirm
# =============================================================================

import argparse
import json
import os
import threading
import time
from datetime import date
from pathlib import Path
from uuid import uuid4

from botocore.exceptions import (
    ClientError,
    ConnectTimeoutError,
    EndpointConnectionError,
    ReadTimeoutError,
)
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command
from pgvector.psycopg import register_vector

from step2_privacy_gateway import (
    PolicyViolation,
    PrivacyContext,
    guarded_call,
    inspect_text,
)
from step3_secure_rag import (
    Document,
    SERVER_GRANTS,
    bedrock_transport,
    build_payload,
    mock_transport,
    validate_answer,
)
from step3b_pgvector_rag import (
    MODEL_NAME,
    connect_db,
    embed,
    get_model,
)
from step4_orchestration import (
    ANALYSTS,
    ReviewState,
    document_agent,
    draft_agent,
    failed_agents,
    human_review,
    initial_state,
    opinion_agent,
    pending_interrupts,
    precheck_agent,
    route_analysis,
    route_precheck,
)


ROOT = Path(__file__).resolve().parent

# 初期実習ではローカルモデルの同時実行とクラウド呼び出しを制限します。
EMBED_LOCK = threading.Lock()
BEDROCK_LIMIT = threading.Semaphore(1)

QUESTIONS = {
    "finance": (
        "財務資料と現金流動を確認し、返済能力を判断するための"
        "確認事項を根拠付きで説明してください。"
    ),
    "industry": (
        "제조업 산업자료를 바탕으로 원재료 가격과 수요 관련 "
        "확인 사항을 근거와 함께 설명하세요."
    ),
    "relations": (
        "관계사 거래자료를 바탕으로 거래 집중과 회수 조건 관련 "
        "확인 사항을 근거와 함께 설명하세요."
    ),
    "collateral": (
        "담보자료를 바탕으로 평가액, 선순위 권리와 처분 가능성 관련 "
        "확인 사항을 근거와 함께 설명하세요."
    ),
}

# 모든 질문은 한국어로 통일합니다.
QUESTIONS["finance"] = (
    "재무자료와 현금흐름 자료를 확인하고, 채무상환 능력을 "
    "판단하기 위한 확인 사항을 근거와 함께 설명하세요."
)

# 역할: (기업자료, 모의 규정)
FIXTURES = {
    "finance": (
        "가상 기업의 전기 매출은 10000백만원, 당기 매출은 "
        "8500백만원이며 Python 계산 매출 증감률은 -15%이다. "
        "당기 현금흐름표는 접수되었지만 차입금 상환 일정은 미제출이다.",
        "모의 재무검토 규정: 매출 변화의 원인과 현금흐름을 확인한다. "
        "채무상환 능력 판단에는 차입금 상환 일정이 필요하며, "
        "자료가 부족하면 판단을 유보한다.",
    ),
    "industry": (
        "가상 제조업 산업자료: 최근 원재료 가격 변동이 관찰되었다. "
        "해당 기업의 가격 전가 가능성과 수주잔고는 확인되지 않았다.",
        "모의 산업검토 규정: 산업 수요, 원재료 가격 변화, "
        "가격 전가 가능성과 수주잔고를 확인한다. "
        "자료가 없는 항목은 추가 확인 사항으로 표시한다.",
    ),
    "relations": (
        "가상 기업의 관계사 거래 비중은 35%이다. "
        "관계사 거래의 결제 조건과 연체 내역은 제출되지 않았다.",
        "모의 관계사검토 규정: 관계사 거래 비중과 결제 조건, "
        "채권 회수 현황 및 보증 관계를 확인한다. "
        "거래 비중만으로 위험 수준을 단정하지 않는다.",
    ),
    "collateral": (
        "가상 담보 평가액은 1200백만원이고 대출 신청금액은 "
        "1000백만원이다. 선순위 권리와 처분 가능성은 미확인이다.",
        "모의 담보검토 규정: 평가 기준일, 선순위 권리, "
        "처분 가능성을 확인한다. 평가액만으로 "
        "대출 가능금액이나 담보 충분 여부를 확정하지 않는다.",
    ),
}


def seed_documents():
    records = []

    for role, texts in FIXTURES.items():
        for kind, text in zip(("CASE", "POLICY"), texts):
            records.append((
                f"DEMO_{role.upper()}_{kind}_V1",
                role,
                "SYNTHETIC_001" if kind == "CASE" else None,
                text,
            ))

    vectors = embed([row[3] for row in records], "passage")

    with connect_db() as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(conn)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS credit_agent_chunks (
                doc_id TEXT PRIMARY KEY,
                role TEXT NOT NULL,
                company_id TEXT,
                valid_from DATE NOT NULL,
                valid_to DATE,
                cloud_allowed BOOLEAN NOT NULL,
                content TEXT NOT NULL,
                embedding_model TEXT NOT NULL,
                embedding VECTOR(384) NOT NULL,
                CHECK (valid_to IS NULL OR valid_to > valid_from)
            )
        """)

        for (doc_id, role, company_id, text), vector in zip(records, vectors):
            conn.execute(
                """
                INSERT INTO credit_agent_chunks (
                    doc_id, role, company_id, valid_from, valid_to,
                    cloud_allowed, content, embedding_model, embedding
                )
                VALUES (%s, %s, %s, %s, NULL, TRUE, %s, %s, %s)
                ON CONFLICT (doc_id) DO UPDATE SET
                    role = EXCLUDED.role,
                    company_id = EXCLUDED.company_id,
                    valid_from = EXCLUDED.valid_from,
                    valid_to = EXCLUDED.valid_to,
                    cloud_allowed = EXCLUDED.cloud_allowed,
                    content = EXCLUDED.content,
                    embedding_model = EXCLUDED.embedding_model,
                    embedding = EXCLUDED.embedding
                """,
                (
                    doc_id, role, company_id, date(2026, 1, 1),
                    text, MODEL_NAME, vector,
                ),
            )

    print(f"Agent용 가상 문서 {len(records)}개 적재 완료")


def retrieve_for_agent(role, user_id, company_id, as_of):
    if role not in QUESTIONS:
        raise PolicyViolation("UNKNOWN_AGENT_ROLE")

    if company_id not in SERVER_GRANTS.get(user_id, frozenset()):
        raise PolicyViolation("CASE_ACCESS_DENIED")

    with EMBED_LOCK:
        vector = embed([QUESTIONS[role]], "query")[0]

    with connect_db() as conn:
        register_vector(conn)

        rows = conn.execute(
            """
            SELECT doc_id, company_id, valid_from, valid_to,
                   cloud_allowed, content
            FROM credit_agent_chunks
            WHERE role = %s
              AND (company_id IS NULL OR company_id = %s)
              AND valid_from <= %s
              AND (valid_to IS NULL OR %s < valid_to)
              AND embedding_model = %s
            ORDER BY embedding <=> %s, doc_id
            LIMIT 2
            """,
            (role, company_id, as_of, as_of, MODEL_NAME, vector),
        ).fetchall()

    docs = [
        Document(
            doc_id=row[0],
            company_id=row[1],
            valid_from=row[2],
            valid_to=row[3],
            cloud_allowed=row[4],
            text=row[5],
        )
        for row in rows
    ]

    for doc in docs:
        if (
            doc.company_id not in (None, company_id)
            or doc.valid_from > as_of
            or (doc.valid_to is not None and as_of >= doc.valid_to)
        ):
            raise PolicyViolation("RETRIEVAL_SCOPE_VIOLATION")

    # 이번 실습에서는 기업자료와 공통 규정이 모두 있어야 분석합니다.
    if not any(doc.company_id == company_id for doc in docs):
        raise PolicyViolation("CASE_EVIDENCE_MISSING")

    if not any(doc.company_id is None for doc in docs):
        raise PolicyViolation("POLICY_EVIDENCE_MISSING")

    return docs


def classify_error(exc):
    if isinstance(exc, PolicyViolation):
        return str(exc), False

    if isinstance(exc, ClientError):
        code = exc.response.get("Error", {}).get("Code", "AWS_ERROR")
        return code, code in {
            "ThrottlingException",
            "ServiceUnavailableException",
            "InternalServerException",
            "ModelNotReadyException",
            "ModelTimeoutException",
        }

    if isinstance(
        exc,
        (ConnectTimeoutError, ReadTimeoutError, EndpointConnectionError),
    ):
        return type(exc).__name__, True

    # 검증·설정·DB 오류 등은 원인을 확인한 후 재실행합니다.
    return type(exc).__name__, False


def analyze_agent(state, role, retriever=None, transport=None):
    started = time.perf_counter()
    attempt = state["attempts"].get(role, 0) + 1
    case = state["case"]

    result = {
        "status": "FAILED",
        "retryable": False,
        "source_ids": [],
        "usage": {},
    }

    try:
        docs = (retriever or retrieve_for_agent)(
            role,
            case["user_id"],
            case["company_id"],
            date.fromisoformat(case["as_of"]),
        )

        result["source_ids"] = [doc.doc_id for doc in docs]

        context = PrivacyContext(
            case_id=case["case_id"],
            token_map={},
            originals=(
                "가상대표",
                "synthetic@example.invalid",
                "010-0000-0000",
            ),
        )

        payload = build_payload(QUESTIONS[role], docs)
        selected_transport = transport or (
            bedrock_transport
            if case["llm_mode"] == "live"
            else mock_transport
        )

        # 모델 호출부는 공통 Gateway를 통과합니다.
        # 초기 실습에서는 실제 모델 동시 호출을 1개로 제한합니다.
        if case["llm_mode"] == "live":
            with BEDROCK_LIMIT:
                response = guarded_call(
                    payload, context, selected_transport
                )
        else:
            response = guarded_call(
                payload, context, selected_transport
            )

        # 응답 검증 실패 시에도 반환받은 사용량은 보존합니다.
        result["usage"] = response["usage"]
        result["stop_reason"] = response["stop_reason"]

        if response["stop_reason"] != "end_turn":
            raise PolicyViolation("INCOMPLETE_RESPONSE")

        inspect_text(response["text"], context)
        answer = validate_answer(response["text"], docs)

        result.update({
            "status": "PASS",
            "summary": "\n".join(
                claim.statement for claim in answer.claims
            ),
            "claims": answer.model_dump()["claims"],
        })

    except Exception as exc:
        code, retryable = classify_error(exc)
        result.update({
            "error_code": code,
            "retryable": retryable,
        })

    result["elapsed_seconds"] = round(time.perf_counter() - started, 3)

    # 최종 결과는 덮어쓰되 시도 이력은 유지합니다.
    old_history = state["results"].get(role, {}).get("history", [])
    result["history"] = old_history + [{
        "attempt": attempt,
        "status": result["status"],
        "error_code": result.get("error_code"),
        "usage": result["usage"],
        "elapsed_seconds": result["elapsed_seconds"],
    }]

    return {
        "results": {role: result},
        "attempts": {role: attempt},
        "events": [f"{role}:{result['status']}"],
    }


def integration_gate(state):
    failed = failed_agents(state)

    if not failed:
        status = "ANALYSIS_COMPLETE"
    elif (
        state["retry_count"] < 1
        and all(state["results"][name].get("retryable") for name in failed)
    ):
        status = "RETRY_REQUIRED"
    else:
        status = "ANALYSIS_FAILED"

    return {
        "status": status,
        "events": [f"analysis_gate:{status}"],
    }


def retry_failed_live(state):
    update = {
        "results": {},
        "attempts": {},
        "events": [],
        "retry_count": state["retry_count"] + 1,
    }

    for role in failed_agents(state):
        part = analyze_agent(state, role)
        update["results"].update(part["results"])
        update["attempts"].update(part["attempts"])
        update["events"].extend(part["events"])

    return update


def make_analysis_node(role):
    def node(state):
        return analyze_agent(state, role)
    return node


def build_integrated_graph(checkpointer):
    graph = StateGraph(ReviewState)

    graph.add_node("documents", document_agent)
    for role in ANALYSTS:
        graph.add_node(role, make_analysis_node(role))

    graph.add_node("analysis_gate", integration_gate)
    graph.add_node("retry_failed", retry_failed_live)
    graph.add_node("opinion", opinion_agent)
    graph.add_node("draft", draft_agent)
    graph.add_node("precheck", precheck_agent)
    graph.add_node("human_review", human_review)

    graph.add_edge(START, "documents")
    for role in ANALYSTS:
        graph.add_edge("documents", role)

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", action="store_true")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--thread")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--confirm", action="store_true")
    args = parser.parse_args()

    if args.seed:
        seed_documents()
        return

    if (args.show or args.confirm) and not args.thread:
        parser.error("--show 또는 --confirm에는 --thread가 필요합니다.")

    thread_id = args.thread or uuid4().hex
    config = {
        "configurable": {"thread_id": thread_id},
        "max_concurrency": 4,
        "recursion_limit": 30,
    }

    runtime_dir = ROOT / ".runtime"
    runtime_dir.mkdir(exist_ok=True)

    # 4-A 체크포인트와 분리합니다.
    with SqliteSaver.from_conn_string(
        str(runtime_dir / "step4b.sqlite")
    ) as saver:
        graph = build_integrated_graph(saver)
        snapshot = graph.get_state(config)

        if args.show:
            if not snapshot.values:
                raise ValueError("THREAD_NOT_FOUND")

        elif args.confirm:
            pending = pending_interrupts(snapshot)
            if (
                len(pending) != 1
                or pending[0]["kind"] != "HUMAN_REVIEW"
            ):
                raise ValueError("NOT_WAITING_FOR_REVIEW")

            graph.invoke(
                Command(resume={"action": "confirm"}),
                config,
            )

        else:
            if snapshot.values:
                raise ValueError("THREAD_ALREADY_EXISTS")

            # 병렬 노드가 시작되기 전에 모델을 한 번 로드합니다.
            get_model()

            state = initial_state("normal")
            state["case"].update({
                "user_id": "reviewer_a",
                "company_id": "SYNTHETIC_001",
                "as_of": "2026-09-13",
                "llm_mode": "live" if args.live else "mock",
                "model_id": os.getenv("BEDROCK_MODEL_ID", ""),
                "request_region": os.getenv("AWS_REGION", ""),
            })
            graph.invoke(state, config)

        snapshot = graph.get_state(config)
        state = snapshot.values

        report = {
            "thread_id": thread_id,
            "mode": state["case"]["llm_mode"],
            "status": state["status"],
            "pending": pending_interrupts(snapshot),
            "attempts": state["attempts"],
            "model_id": state["case"]["model_id"],
            "request_region": state["case"]["request_region"],
            "results": state["results"],
            "draft": state["draft"],
            "events": state["events"],
        }

        output_dir = ROOT / "outputs"
        output_dir.mkdir(exist_ok=True)
        path = output_dir / f"step4b_{thread_id}.json"
        path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        print(json.dumps(report, ensure_ascii=False, indent=2))
        print(f"\n저장 위치: {path}")


if __name__ == "__main__":
    main()