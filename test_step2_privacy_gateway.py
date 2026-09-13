# =============================================================================
# 2단계 테스트: 개인정보 전송 차단 및 가명 토큰 복원 검증
# =============================================================================
#
# 목적
# - 실제 AWS 호출 없이 개인정보 통제와 복원 로직을 검증합니다.
# - 차단 상황에서 예외가 발생하는 것뿐 아니라,
#   모델 호출 함수가 실제로 실행되지 않았는지 확인합니다.
# - unittest와 Mock을 사용하므로 별도 테스트 패키지가 필요하지 않습니다.
#
# 테스트 항목: 총 7개
# 1. 정상 처리 및 복원
#    - 모델 호출 인자에 원본 이름, 이메일, 전화번호, 심사 건 ID가 없는지 확인
#    - 모델 호출이 한 번 수행되고 대표자 토큰이 정상 복원되는지 확인
#
# 2. 원본 이름 유출 차단
#    - 전송 후보에 원본 이름을 추가하면 모델 호출 전에 차단되는지 확인
#
# 3. 새로운 이메일 유출 차단
#    - 입력 원본과 다른 이메일도 패턴 검사로 차단되는지 확인
#
# 4. 검사기 장애 시 전송 차단
#    - 개인정보 검사기가 예외를 발생시키면 모델을 호출하지 않는지 확인
#
# 5. 다른 심사 건의 복원 차단
#    - 요청한 심사 건과 매핑의 심사 건이 다르면 복원을 거부하는지 확인
#
# 6. 알 수 없는 토큰 차단
#    - 모델이 매핑에 없는 대표자 토큰을 반환하면 거부하는지 확인
#
# 7. 미정의 입력 필드 차단
#    - 입력 스키마에 없는 추가 자료가 검증 단계에서 거부되는지 확인
#
# 실행 방법: 프로젝트 루트에서 실행
#   uv run python -m unittest -v test_step2_privacy_gateway.py
#
# 기대 결과
#   Ran 7 tests ...
#   OK
#
# 검증 범위
# - 정의된 사례에서 함수 수준의 통제가 작동하는지 확인하는 테스트입니다.
# - 실제 Bedrock 연동, 네트워크 격리, IAM 권한, 사용자 인증,
#   모든 개인정보 패턴의 탐지와 분석 내용의 정확성은 검증하지 않습니다.
# - 실제 연동은 step2_privacy_gateway.py --live로 별도 확인합니다.
# =============================================================================

import json
import unittest
from unittest.mock import Mock

from pydantic import ValidationError

from step2_privacy_gateway import (
    CreditInput,
    PolicyViolation,
    guarded_call,
    mock_transport,
    prepare_request,
    restore_response,
)


class PrivacyGatewayTests(unittest.TestCase):
    def setUp(self):
        self.data = CreditInput(
            case_id="CASE_001",
            company_id="SYNTHETIC_001",
            representative_name="가상대표",
            email="synthetic@example.invalid",
            phone="010-0000-0000",
            revenue_growth_pct=-15.0,
            operating_margin_pct=4.0,
            debt_to_equity_pct=200.0,
        )
        self.text, self.context = prepare_request(self.data)

    def test_allowed_request_and_restore(self):
        transport = Mock(side_effect=mock_transport)

        response = guarded_call(
            self.text, self.context, transport
        )
        restored = restore_response(
            response["text"], self.context, "CASE_001"
        )

        sent_text = transport.call_args.args[0]

        self.assertNotIn(self.data.representative_name, sent_text)
        self.assertNotIn(self.data.email, sent_text)
        self.assertNotIn(self.data.phone, sent_text)
        self.assertNotIn(self.data.case_id, sent_text)
        self.assertEqual(
            restored["representative"],
            self.data.representative_name,
        )
        transport.assert_called_once()

    def test_original_name_blocks_transport(self):
        transport = Mock()
        unsafe = self.text + self.data.representative_name

        with self.assertRaises(PolicyViolation):
            guarded_call(unsafe, self.context, transport)

        transport.assert_not_called()

    def test_new_email_blocks_transport(self):
        transport = Mock()
        unsafe = self.text + " other@example.invalid"

        with self.assertRaises(PolicyViolation):
            guarded_call(unsafe, self.context, transport)

        transport.assert_not_called()

    def test_inspector_failure_blocks_transport(self):
        transport = Mock()
        broken_inspector = Mock(
            side_effect=RuntimeError("검사 서비스 장애")
        )

        with self.assertRaises(RuntimeError):
            guarded_call(
                self.text,
                self.context,
                transport,
                inspector=broken_inspector,
            )

        transport.assert_not_called()

    def test_cross_case_restore_blocked(self):
        response = mock_transport(self.text)

        with self.assertRaises(PolicyViolation):
            restore_response(
                response["text"], self.context, "CASE_002"
            )

    def test_unknown_token_blocked(self):
        response = json.loads(mock_transport(self.text)["text"])
        response["representative"] = "[PERSON_UNKNOWN]"

        with self.assertRaises(PolicyViolation):
            restore_response(
                json.dumps(response),
                self.context,
                "CASE_001",
            )

    def test_extra_input_field_rejected(self):
        fields = self.data.model_dump()
        fields["unreviewed_document"] = "검사되지 않은 추가 자료"

        with self.assertRaises(ValidationError):
            CreditInput.model_validate(fields)


if __name__ == "__main__":
    unittest.main()