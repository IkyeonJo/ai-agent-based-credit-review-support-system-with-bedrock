# =============================================================================
# 4-A 테스트: 병렬 결과 취합, 재시도, 초안 수정, 영속 상태 재개
#
# 실행:
#   uv run python -m unittest -v test_step4_orchestration.py
#
# AWS·PostgreSQL을 호출하지 않습니다.
# 각 테스트는 임시 SQLite 파일을 사용합니다.
# =============================================================================

import tempfile
import unittest
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from step4_orchestration import (
    ANALYSTS,
    build_graph,
    initial_state,
    pending_interrupts,
)


class OrchestrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.db = str(Path(self.temp.name) / "state.sqlite")
        self.config = {
            "configurable": {"thread_id": "test-run"},
            "max_concurrency": 4,
            "recursion_limit": 30,
        }

    def tearDown(self):
        self.temp.cleanup()

    def test_normal_join_and_review(self):
        with SqliteSaver.from_conn_string(self.db) as saver:
            graph = build_graph(saver)
            graph.invoke(initial_state("normal"), self.config)
            snapshot = graph.get_state(self.config)

            self.assertEqual(
                pending_interrupts(snapshot)[0]["kind"],
                "HUMAN_REVIEW",
            )

            events = snapshot.values["events"]
            for name in ANALYSTS:
                self.assertLess(
                    events.index(f"{name}:PASS"),
                    events.index("opinion:PASS"),
                )

            graph.invoke(
                Command(resume={"action": "confirm"}),
                self.config,
            )
            self.assertEqual(
                graph.get_state(self.config).values["status"],
                "REVIEWED",
            )

    def test_missing_documents_stop_analysis(self):
        with SqliteSaver.from_conn_string(self.db) as saver:
            graph = build_graph(saver)
            graph.invoke(initial_state("missing"), self.config)
            snapshot = graph.get_state(self.config)

            self.assertEqual(snapshot.values["attempts"], {})
            self.assertEqual(
                pending_interrupts(snapshot)[0]["kind"],
                "DOCUMENTS_REQUIRED",
            )

    def test_retry_only_failed_agent(self):
        with SqliteSaver.from_conn_string(self.db) as saver:
            graph = build_graph(saver)
            graph.invoke(initial_state("retry"), self.config)
            state = graph.get_state(self.config).values

            self.assertEqual(
                state["attempts"],
                {
                    "finance": 1,
                    "industry": 2,
                    "relations": 1,
                    "collateral": 1,
                },
            )
            self.assertEqual(state["retry_count"], 1)

    def test_draft_repair_preserves_analysis(self):
        with SqliteSaver.from_conn_string(self.db) as saver:
            graph = build_graph(saver)
            graph.invoke(initial_state("draft_error"), self.config)
            state = graph.get_state(self.config).values

            self.assertEqual(state["draft_attempts"], 2)
            self.assertEqual(state["draft"]["requested_amount"], 1000)
            self.assertTrue(
                all(state["attempts"][name] == 1 for name in ANALYSTS)
            )
            self.assertEqual(state["check_errors"], [])

    def test_resume_after_reopening_database(self):
        with SqliteSaver.from_conn_string(self.db) as saver:
            graph = build_graph(saver)
            graph.invoke(initial_state("missing"), self.config)

        # 关闭连接后重新建立图，验证持久化状态恢复。
        with SqliteSaver.from_conn_string(self.db) as saver:
            graph = build_graph(saver)
            graph.invoke(
                Command(resume={"provided": ["cashflow"]}),
                self.config,
            )
            graph.invoke(
                Command(resume={"action": "confirm"}),
                self.config,
            )

            snapshot = graph.get_state(self.config)
            self.assertEqual(snapshot.values["status"], "REVIEWED")
            self.assertEqual(pending_interrupts(snapshot), [])


if __name__ == "__main__":
    unittest.main()