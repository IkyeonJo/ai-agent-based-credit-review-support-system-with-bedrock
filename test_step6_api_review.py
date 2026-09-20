# =============================================================================
# 6단계: 웹 API와 직원 검토 기능 검증
# =============================================================================
#
# 검증 항목
# 1. 분석 기본 모드가 mock인지 확인
# 2. 클라이언트가 임의의 사용자·기업 정보를 추가하면 거부
# 3. 동일 실행에 대한 중복 검토 요청 차단
# 4. 공백만 있는 검토 의견 거부
# 5. 존재하지 않는 실행 조회 시 404
# 6. 직원 수정본·원본·메모 저장 및 분석 재호출 방지
#
# 범위
# - API 테스트의 작업 제출은 Mock으로 대체합니다.
# - 실제 AWS·PostgreSQL을 호출하지 않습니다.
# - 각 테스트는 별도의 임시 SQLite 파일을 사용합니다.
# - 필드 거부 테스트는 실제 사용자 인증·권한 검증을 대체하지 않습니다.
#
# 실행
#   uv run python -m unittest -v test_step6_api_review.py
# =============================================================================

import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

from fastapi.testclient import TestClient
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

import step5_api
from step4_orchestration import build_graph, initial_state


class ApiReviewTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)

        temp_dir = self.stack.enter_context(
            tempfile.TemporaryDirectory()
        )
        self.root = Path(temp_dir)

        self.stack.enter_context(
            patch.multiple(
                step5_api,
                RUNTIME=self.root,
                JOBS_DB=self.root / "jobs.sqlite",
                GRAPH_DB=self.root / "graph.sqlite",
            )
        )

        # context manager로 서버 lifespan도 실행합니다.
        self.client = self.stack.enter_context(
            TestClient(step5_api.app)
        )

        # 작업 접수만 검증하며 실제 분석 작업은 실행하지 않습니다.
        self.submit = self.stack.enter_context(
            patch.object(step5_api.app.state.executor, "submit")
        )

    def create_run(self):
        response = self.client.post("/api/runs", json={})
        self.assertEqual(response.status_code, 202)
        return response.json()["run_id"]

    def test_default_mode_is_mock(self):
        run_id = self.create_run()
        job = step5_api.load_job(run_id)

        self.assertEqual(job["mode"], "mock")
        self.assertEqual(job["status"], "QUEUED")
        self.submit.assert_called_once()

    def test_unexpected_identity_fields_rejected(self):
        response = self.client.post(
            "/api/runs",
            json={
                "mode": "mock",
                "user_id": "another_user",
                "company_id": "SYNTHETIC_002",
            },
        )

        self.assertEqual(response.status_code, 422)
        self.submit.assert_not_called()

    def test_duplicate_review_blocked(self):
        run_id = self.create_run()
        step5_api.set_job_status(run_id, "REVIEW_PENDING")
        self.submit.reset_mock()

        body = {
            "action": "confirm",
            "edited_opinion": "가상 검토 의견입니다.",
            "note": "담당자 확인",
        }

        first = self.client.post(
            f"/api/runs/{run_id}/review",
            json=body,
        )
        second = self.client.post(
            f"/api/runs/{run_id}/review",
            json=body,
        )

        self.assertEqual(first.status_code, 202)
        self.assertEqual(second.status_code, 409)
        self.submit.assert_called_once()

    def test_blank_review_rejected(self):
        run_id = self.create_run()
        step5_api.set_job_status(run_id, "REVIEW_PENDING")
        self.submit.reset_mock()

        response = self.client.post(
            f"/api/runs/{run_id}/review",
            json={
                "action": "confirm",
                "edited_opinion": "   ",
                "note": "",
            },
        )

        self.assertEqual(response.status_code, 422)
        self.assertEqual(
            step5_api.load_job(run_id)["status"],
            "REVIEW_PENDING",
        )
        self.submit.assert_not_called()

    def test_unknown_run_returns_404(self):
        response = self.client.get(f"/api/runs/{uuid4().hex}")
        self.assertEqual(response.status_code, 404)

    def test_edited_draft_preserved_without_reanalysis(self):
        config = {
            "configurable": {"thread_id": "review-test"},
            "max_concurrency": 4,
        }

        with SqliteSaver.from_conn_string(
            str(self.root / "review.sqlite")
        ) as saver:
            graph = build_graph(saver)
            graph.invoke(initial_state("normal"), config)

            before = graph.get_state(config).values
            original = before["draft"]["opinion"]
            attempts_before = dict(before["attempts"])
            edited = original + "\n담당자 확인: 추가 자료 검토 예정."

            graph.invoke(
                Command(resume={
                    "action": "confirm",
                    "edited_opinion": edited,
                    "note": "현금흐름 확인 필요",
                }),
                config,
            )

            after = graph.get_state(config).values

            self.assertEqual(after["status"], "REVIEWED")
            self.assertEqual(after["draft"]["opinion"], edited)
            self.assertEqual(
                after["draft"]["review"]["original_opinion"],
                original,
            )
            self.assertEqual(
                after["draft"]["review"]["note"],
                "현금흐름 확인 필요",
            )
            self.assertTrue(after["draft"]["review"]["edited"])
            self.assertEqual(after["attempts"], attempts_before)


if __name__ == "__main__":
    unittest.main()