# =============================================================================
# 3-A단계 테스트: 내부 RAG의 검색 범위·적용일·전송·출처 검증
# =============================================================================
#
# 실제 AWS 호출 없이 검증합니다.
# - 권한 없는 기업 접근 차단
# - 다른 기업 문서 제외
# - 심사 기준일에 맞는 규정 버전 선택
# - 외부 전송 금지 문서 발견 시 모델 호출 차단
# - 검색 근거가 없을 때 모델 호출 차단
# - 존재하지 않는 출처와 잘못된 인용문 거부
# - 정상 RAG 흐름 통과
#
# 실행:
#   uv run python -m unittest -v test_step3_secure_rag.py
# =============================================================================

import json
import unittest
from datetime import date
from unittest.mock import Mock

from step2_privacy_gateway import PolicyViolation, PrivacyContext
from step3_secure_rag import (
    DOCUMENTS,
    eligible_documents,
    mock_transport,
    run_rag,
    validate_answer,
)


class SecureRagTests(unittest.TestCase):
    def setUp(self):
        self.context = PrivacyContext(
            case_id="CASE_001",
            token_map={},
            originals=("가상대표",),
        )
        self.as_of = date(2026, 9, 13)

    def run_query(self, question, transport, company="SYNTHETIC_001"):
        return run_rag(
            question,
            "reviewer_a",
            company,
            self.as_of,
            self.context,
            transport,
        )

    def test_other_company_access_denied(self):
        transport = Mock()
        with self.assertRaises(PolicyViolation):
            self.run_query("현금흐름", transport, "SYNTHETIC_002")
        transport.assert_not_called()

    def test_other_company_documents_excluded(self):
        docs = eligible_documents(
            "reviewer_a", "SYNTHETIC_001", self.as_of
        )
        self.assertNotIn(
            "COMPANY_B_FINANCE_V1_P1",
            {doc.doc_id for doc in docs},
        )

    def test_current_policy_selected(self):
        docs = eligible_documents(
            "reviewer_a", "SYNTHETIC_001", self.as_of
        )
        ids = {doc.doc_id for doc in docs}
        self.assertIn("POLICY_CASHFLOW_V2_P1", ids)
        self.assertNotIn("POLICY_CASHFLOW_V1_P1", ids)
        self.assertNotIn("POLICY_CASHFLOW_V3_P1", ids)

    def test_policy_boundary(self):
        docs = eligible_documents(
            "reviewer_a", "SYNTHETIC_001", date(2026, 7, 1)
        )
        ids = {doc.doc_id for doc in docs}
        self.assertIn("POLICY_CASHFLOW_V2_P1", ids)
        self.assertNotIn("POLICY_CASHFLOW_V1_P1", ids)

    def test_restricted_document_blocks_call(self):
        transport = Mock()
        with self.assertRaisesRegex(
            PolicyViolation, "CLOUD_TRANSFER_DENIED"
        ):
            self.run_query("기밀약정 특별약정 비공개약정", transport)
        transport.assert_not_called()

    def test_no_evidence_blocks_call(self):
        transport = Mock()
        with self.assertRaisesRegex(PolicyViolation, "NO_EVIDENCE"):
            self.run_query("ZXQW987654", transport)
        transport.assert_not_called()

    def test_unknown_source_rejected(self):
        raw = json.dumps({
            "claims": [{
                "statement": "설명",
                "source_id": "FAKE_SOURCE",
                "quote": "없는 근거",
            }]
        })
        with self.assertRaisesRegex(PolicyViolation, "UNKNOWN_SOURCE"):
            validate_answer(raw, DOCUMENTS)

    def test_wrong_quote_rejected(self):
        raw = json.dumps({
            "claims": [{
                "statement": "설명",
                "source_id": "POLICY_CASHFLOW_V2_P1",
                "quote": "원문에 존재하지 않는 문장",
            }]
        })
        with self.assertRaisesRegex(PolicyViolation, "QUOTE_MISMATCH"):
            validate_answer(raw, DOCUMENTS)

    def test_normal_flow(self):
        transport = Mock(side_effect=mock_transport)
        docs, answer, _ = self.run_query(
            "현금흐름과 차입금 상환 일정 확인", transport
        )
        self.assertTrue(docs)
        self.assertTrue(answer.claims)
        transport.assert_called_once()


if __name__ == "__main__":
    unittest.main()