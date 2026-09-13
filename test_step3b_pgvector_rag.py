# =============================================================================
# 3-B 통합 테스트: pgvector 저장·검색·통제 연결
# =============================================================================
#
# 사전 조건
# - PostgreSQL 컨테이너 실행
# - step3b_pgvector_rag.py --init 완료
#
# 검증
# - 문서 6개와 384차원 벡터 저장
# - 다른 기업 접근 차단
# - SQL 검색 결과의 기업·적용일 범위 제한
# - 적용 가능한 문서가 없으면 모델 호출 차단
# - 전송 금지 문서가 포함되면 모델 호출 차단
#
# 실행
#   uv run python -m unittest -v test_step3b_pgvector_rag.py
# =============================================================================

import unittest
from datetime import date
from unittest.mock import Mock

from step2_privacy_gateway import PolicyViolation, PrivacyContext
from step3_secure_rag import DOCUMENTS
from step3b_pgvector_rag import (
    answer_from_hits,
    connect_db,
    search,
)


class PgvectorRagTests(unittest.TestCase):
    def setUp(self):
        self.context = PrivacyContext(
            case_id="CASE_001",
            token_map={},
            originals=("가상대표",),
        )

    def test_storage(self):
        with connect_db() as conn:
            rows = conn.execute(
                """
                SELECT doc_id, vector_dims(embedding)
                FROM credit_chunks
                """
            ).fetchall()

        stored = dict(rows)
        for doc in DOCUMENTS:
            self.assertEqual(stored.get(doc.doc_id), 384)

    def test_unauthorized_company(self):
        with self.assertRaisesRegex(
            PolicyViolation, "CASE_ACCESS_DENIED"
        ):
            search(
                "현금흐름",
                "reviewer_a",
                "SYNTHETIC_002",
                date(2026, 9, 13),
            )

    def test_sql_scope(self):
        # 전체 후보 수보다 큰 top_k로 범위 필터 자체를 확인합니다.
        hits = search(
            "현금흐름",
            "reviewer_a",
            "SYNTHETIC_001",
            date(2026, 9, 13),
            top_k=20,
        )

        ids = {doc.doc_id for doc, _ in hits}

        self.assertEqual(
            ids,
            {
                "POLICY_CASHFLOW_V2_P1",
                "COMPANY_A_FINANCE_V1_P1",
                "COMPANY_A_RESTRICTED_V1_P1",
            },
        )

    def test_no_valid_documents_blocks_call(self):
        hits = search(
            "현금흐름",
            "reviewer_a",
            "SYNTHETIC_001",
            date(2020, 1, 1),
        )
        transport = Mock()

        with self.assertRaisesRegex(PolicyViolation, "NO_EVIDENCE"):
            answer_from_hits(
                "현금흐름", hits, self.context, transport
            )

        transport.assert_not_called()

    def test_restricted_result_blocks_call(self):
        hits = search(
            "기밀약정",
            "reviewer_a",
            "SYNTHETIC_001",
            date(2026, 9, 13),
            top_k=20,
        )
        self.assertTrue(
            any(not doc.cloud_allowed for doc, _ in hits)
        )
        transport = Mock()

        with self.assertRaisesRegex(
            PolicyViolation, "CLOUD_TRANSFER_DENIED"
        ):
            answer_from_hits(
                "기밀약정", hits, self.context, transport
            )

        transport.assert_not_called()


if __name__ == "__main__":
    unittest.main()