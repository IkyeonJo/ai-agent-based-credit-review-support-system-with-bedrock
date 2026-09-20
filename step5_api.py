# =============================================================================
# 5단계: Multi-Agent 실행·조회·검토 FastAPI
# =============================================================================
#
# 기능
# - 실행 요청에 run_id를 즉시 반환
# - 작업 스레드에서 4-B 그래프 실행
# - 체크포인트로 분석 상태·근거·초안 조회
# - 직원 수정 의견과 검토 결과를 전달해 그래프 재개
# - 중복 검토 요청은 상태 검사로 차단
#
# 범위
# - 가상 기업 한 곳과 고정된 실습 사용자만 사용합니다.
# - 실제 로그인·SSO 인증은 구현하지 않습니다.
# - 서버는 단일 프로세스, 단일 작업 실행기로 구동합니다.
# - 로컬 실습용이므로 127.0.0.1에서만 실행합니다.
# - 강제 종료된 작업은 자동 재호출하지 않고 RECOVERY_REQUIRED로 표시합니다.
# - 실습 결과와 체크포인트에는 가상 분석문이 저장됩니다.
#
# 실행
#   uv run uvicorn step5_api:app --host 127.0.0.1 --port 8010
# =============================================================================

import json
import os
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager, closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
from uuid import UUID, uuid4

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command
from pydantic import BaseModel, ConfigDict, Field

from step3b_pgvector_rag import get_model
from step4_orchestration import initial_state, pending_interrupts
from step4b_integrated_agents import build_integrated_graph


ROOT = Path(__file__).resolve().parent
RUNTIME = ROOT / ".runtime"
JOBS_DB = RUNTIME / "step5_jobs.sqlite"
GRAPH_DB = RUNTIME / "step5_graph.sqlite"


def now():
    return datetime.now(timezone.utc).isoformat()


def db_execute(sql, parameters=(), fetch=False):
    with closing(sqlite3.connect(JOBS_DB, timeout=10)) as conn:
        conn.row_factory = sqlite3.Row
        with conn:
            cursor = conn.execute(sql, parameters)
            if fetch:
                return [dict(row) for row in cursor.fetchall()]
            return cursor.rowcount


def load_job(run_id):
    rows = db_execute(
        "SELECT * FROM jobs WHERE run_id = ?",
        (run_id,),
        fetch=True,
    )
    if not rows:
        raise HTTPException(404, "실행 ID를 찾을 수 없습니다.")
    return rows[0]


def graph_config(run_id):
    return {
        "configurable": {"thread_id": run_id},
        "max_concurrency": 4,
        "recursion_limit": 30,
    }


def set_job_status(run_id, status, error_type=None):
    db_execute(
        """
        UPDATE jobs
        SET status = ?, updated_at = ?, error_type = ?
        WHERE run_id = ?
        """,
        (status, now(), error_type, run_id),
    )


def execute_job(run_id, review=None):
    try:
        job = load_job(run_id)
        set_job_status(run_id, "RUNNING")
        config = graph_config(run_id)

        with SqliteSaver.from_conn_string(str(GRAPH_DB)) as saver:
            graph = build_integrated_graph(saver)

            if review is None:
                get_model()

                state = initial_state("normal")
                state["case"].update({
                    "user_id": "reviewer_a",
                    "company_id": "SYNTHETIC_001",
                    "as_of": "2026-09-13",
                    "llm_mode": job["mode"],
                    "model_id": os.getenv("BEDROCK_MODEL_ID", ""),
                    "request_region": os.getenv("AWS_REGION", ""),
                })
                graph.invoke(state, config)

            else:
                snapshot = graph.get_state(config)
                pending = pending_interrupts(snapshot)

                if (
                    len(pending) != 1
                    or pending[0]["kind"] != "HUMAN_REVIEW"
                ):
                    raise ValueError("NOT_WAITING_FOR_REVIEW")

                graph.invoke(Command(resume=review), config)

            snapshot = graph.get_state(config)
            pending = pending_interrupts(snapshot)

            if pending:
                status = "REVIEW_PENDING"
            else:
                status = snapshot.values.get("status", "FAILED")

            set_job_status(run_id, status)

    except Exception as exc:
        # 예외 전문에 원문이나 연결 정보가 포함될 수 있어 유형만 기록합니다.
        set_job_status(run_id, "FAILED", type(exc).__name__)


@asynccontextmanager
async def lifespan(app):
    RUNTIME.mkdir(exist_ok=True)

    db_execute("""
        CREATE TABLE IF NOT EXISTS jobs (
            run_id TEXT PRIMARY KEY,
            mode TEXT NOT NULL,
            status TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            error_type TEXT,
            review_request TEXT
        )
    """)

    # 강제 종료된 실행을 자동으로 다시 호출하지 않습니다.
    db_execute(
        """
        UPDATE jobs
        SET status = 'RECOVERY_REQUIRED', updated_at = ?
        WHERE status IN ('QUEUED', 'RUNNING')
        """,
        (now(),),
    )

    # 첫 조회 전에 체크포인트 테이블을 준비합니다.
    with SqliteSaver.from_conn_string(str(GRAPH_DB)) as saver:
        saver.setup()

    app.state.executor = ThreadPoolExecutor(max_workers=1)

    try:
        yield
    finally:
        # 정상 종료에서는 접수된 작업이 마무리될 때까지 기다립니다.
        app.state.executor.shutdown(wait=True)


app = FastAPI(
    title="기업여신 심사지원 PoC",
    lifespan=lifespan,
)


class StartRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mode: Literal["mock", "live"] = "mock"


class ReviewRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    action: Literal["confirm", "return"]
    edited_opinion: str = Field(min_length=1, max_length=30000)
    note: str = Field(default="", max_length=3000)


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(ROOT / "web" / "index.html")


@app.get("/api/health")
def health():
    # 웹 API의 생존 확인입니다. DB·Bedrock 연결 전체 검증은 아닙니다.
    return {"status": "ok"}


@app.post("/api/runs", status_code=202)
def start_run(body: StartRequest):
    run_id = uuid4().hex
    created = now()

    db_execute(
        """
        INSERT INTO jobs (
            run_id, mode, status, created_at, updated_at
        ) VALUES (?, ?, 'QUEUED', ?, ?)
        """,
        (run_id, body.mode, created, created),
    )

    app.state.executor.submit(execute_job, run_id)
    return {"run_id": run_id, "status": "QUEUED"}


@app.get("/api/runs")
def list_runs():
    return db_execute(
        """
        SELECT run_id, mode, status, created_at, updated_at, error_type
        FROM jobs
        ORDER BY created_at DESC
        LIMIT 50
        """,
        fetch=True,
    )


@app.get("/api/runs/{run_id}")
def get_run(run_id: UUID):
    key = run_id.hex
    job = load_job(key)

    with SqliteSaver.from_conn_string(str(GRAPH_DB)) as saver:
        graph = build_integrated_graph(saver)
        snapshot = graph.get_state(graph_config(key))

    state = snapshot.values or {}

    return {
        "run_id": key,
        "mode": job["mode"],
        "status": job["status"],
        "error_type": job["error_type"],
        "workflow_status": state.get("status"),
        "pending": pending_interrupts(snapshot),
        "results": state.get("results", {}),
        "attempts": state.get("attempts", {}),
        "events": state.get("events", []),
        "draft": state.get("draft", {}),
    }


@app.post("/api/runs/{run_id}/review", status_code=202)
def review_run(run_id: UUID, body: ReviewRequest):
    key = run_id.hex
    load_job(key)

    if not body.edited_opinion.strip():
        raise HTTPException(422, "종합의견을 입력하세요.")

    review = body.model_dump()

    # 상태 확인과 변경을 하나의 SQL 문으로 처리합니다.
    # 동일 실행을 두 번 검토 요청하면 두 번째 요청은 거부됩니다.
    changed = db_execute(
        """
        UPDATE jobs
        SET status = 'QUEUED',
            updated_at = ?,
            review_request = ?
        WHERE run_id = ? AND status = 'REVIEW_PENDING'
        """,
        (
            now(),
            json.dumps(review, ensure_ascii=False),
            key,
        ),
    )

    if changed != 1:
        raise HTTPException(409, "현재 검토 가능한 상태가 아닙니다.")

    app.state.executor.submit(execute_job, key, review)
    return {"run_id": key, "status": "QUEUED"}