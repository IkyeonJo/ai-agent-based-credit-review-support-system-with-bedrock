# =============================================================================
# 4-B 단위 테스트
# - 정상 분석의 근거 저장
# - 전송 금지 시 모델 호출 차단
# - 일시적 서비스 오류의 재시도 분류
# - 권한 오류의 재시도 금지
# =============================================================================

import unittest
from datetime import date
from unittest.mock import Mock

from botocore.exceptions import ClientError

from step3_secure_rag import Document, mock_transport
from step4_orchestration import initial_state
from step4b_integrated_agents import analyze_agent, classify_error


class IntegratedAgentTests(unittest.TestCase):
    def setUp(self):
        self.state = initial_state("normal")
        self.state["case"].update({
            "user_id": "reviewer_a",
            "company_id": "SYNTHETIC_001",
            "as_of": "2026-09-13",
            "llm_mode": "mock",
        })

    def document(self, allowed=True):
        return Document(
            doc_id="TEST_FINANCE_V1",
            company_id="SYNTHETIC_001",
            valid_from=date(2026, 1, 1),
            valid_to=None,
            cloud_allowed=allowed,
            text="가상 기업은 차입금 상환 일정을 추가 제출해야 한다.",
        )

    def test_success_records_evidence(self):
        result = analyze_agent(
            self.state,
            "finance",
            retriever=Mock(return_value=[self.document()]),
            transport=mock_transport,
        )["results"]["finance"]

        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["source_ids"], ["TEST_FINANCE_V1"])
        self.assertTrue(result["claims"])
        self.assertEqual(len(result["history"]), 1)

    def test_policy_failure_does_not_call_model(self):
        transport = Mock()

        result = analyze_agent(
            self.state,
            "finance",
            retriever=Mock(return_value=[self.document(False)]),
            transport=transport,
        )["results"]["finance"]

        self.assertEqual(result["status"], "FAILED")
        self.assertEqual(result["error_code"], "CLOUD_TRANSFER_DENIED")
        self.assertFalse(result["retryable"])
        transport.assert_not_called()

    def test_transient_error_is_retryable(self):
        error = ClientError(
            {"Error": {"Code": "ServiceUnavailableException"}},
            "Converse",
        )
        code, retryable = classify_error(error)

        self.assertEqual(code, "ServiceUnavailableException")
        self.assertTrue(retryable)

    def test_access_error_is_not_retryable(self):
        error = ClientError(
            {"Error": {"Code": "AccessDeniedException"}},
            "Converse",
        )
        _, retryable = classify_error(error)
        self.assertFalse(retryable)


if __name__ == "__main__":
    unittest.main()