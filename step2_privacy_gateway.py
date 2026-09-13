# =============================================================================
# 2단계: 개인정보 처리 및 Bedrock 전송 통제 Gateway
# =============================================================================
#
# 목적
# - 온프레미스에서 개인정보를 처리한 뒤, 검사를 통과한 요청만
#   클라우드의 Bedrock Claude로 보내는 흐름을 학습합니다.
# - 가상 기업 데이터만 사용하는 포트폴리오용 구현입니다.
#
# 처리 순서
# 1. 입력 검증: 정의되지 않은 필드는 거부합니다.
# 2. 데이터 최소화: 외부 전송을 허용한 필드만 새로 구성합니다.
# 3. 가명 처리: 대표자 이름을 요청별 [PERSON_...] 토큰으로 치환합니다.
# 4. 전송 제외: 이메일, 전화번호, 기업 ID, 심사 건 ID는 보내지 않습니다.
# 5. 전송 직전 검사: 원본 개인정보와 미처리 개인정보 패턴을 확인합니다.
# 6. 모델 호출: 검사 통과 시에만 모의 응답 또는 Bedrock을 호출합니다.
# 7. 응답 검증: 정상 종료 여부, JSON 구조, 가명 토큰을 확인합니다.
# 8. 내부 복원: 심사 건 일치 여부를 확인하고 해당 요청의 토큰만 복원합니다.
# 9. 감사 기록: 처리 상태, 정책 버전, 사용량, 오류 유형 등을 저장합니다.
#
# 핵심 통제
# - 개인정보가 발견되거나 검사기가 고장 나면 모델 호출을 중단합니다.
#   즉, 검사에 실패했을 때 전송을 허용하지 않는 fail-closed 방식입니다.
# - 다른 심사 건의 매핑이나 모델이 생성한 알 수 없는 토큰은 거부합니다.
# - 원문, 응답 본문, API 키, 복원 매핑은 감사 파일에 저장하지 않습니다.
# - 처리 후 전송 후보와 내부 복원 결과의 콘솔 출력은 가상 데이터 시연용입니다.
#
# 구현 범위와 한계
# - 구조화된 입력과 일부 정규식 패턴을 대상으로 하는 학습용 통제입니다.
# - 자유문서의 모든 이름, 주소, 개인정보를 탐지하는 기능은 아닙니다.
# - 복원 가능한 토큰 치환은 가명 처리이며 완전한 익명화가 아닙니다.
# - 복원 매핑은 요청별 메모리에만 존재하며 파일에 저장하지 않습니다.
# - 심사 건 비교는 실제 사용자 인증 및 복원 권한 검사를 대체하지 않습니다.
# - Gateway는 Python 함수 수준의 논리적 통제입니다. 우회 호출 방지를 위한
#   별도 서비스, 네트워크 제한, IAM 권한 분리는 이후 단계에서 구현합니다.
# - 실제 금융기관의 보안 요건 충족을 입증하는 운영용 구현은 아닙니다.
#
# 의존성
# - step1_bedrock.py의 create_client, normalize_json_response를 재사용합니다.
# - 프로젝트 루트의 .env에 설정한 Bedrock 인증, 리전, 모델을 사용합니다.
#
# 실행 방법: 프로젝트 루트에서 실행
#   uv run python step2_privacy_gateway.py
#       → AWS 호출 없이 모의 실행
#
#   uv run python step2_privacy_gateway.py --live
#       → 가상 데이터로 실제 Bedrock 호출
#
# 결과 파일
# - outputs/step2_<실행ID>.json
# - status: PASS는 구현된 처리·검증 흐름의 통과를 의미하며,
#   분석 내용의 정확성이나 모든 개인정보 보호를 보장하지 않습니다.
# =============================================================================


import argparse
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from step1_bedrock import (
    create_client,
    normalize_json_response,
)


ROOT = Path(__file__).resolve().parent


class PolicyViolation(ValueError):
    """전송·복원 정책 위반. 오류 메시지에 원문을 넣지 않습니다."""


class CreditInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    case_id: str = Field(pattern=r"^CASE_[0-9]{3}$")
    company_id: str = Field(pattern=r"^SYNTHETIC_[0-9]{3}$")
    representative_name: str = Field(min_length=2, max_length=30)
    email: str = Field(min_length=1)
    phone: str = Field(min_length=1)
    revenue_growth_pct: float
    operating_margin_pct: float
    debt_to_equity_pct: float


class AnalysisOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    representative: str
    summary: str = Field(min_length=1)
    review_points: list[str] = Field(min_length=1)


@dataclass
class PrivacyContext:
    # 매핑은 요청별 메모리에만 두고 파일에 저장하지 않습니다.
    case_id: str
    token_map: dict[str, str] = field(repr=False)
    originals: tuple[str, ...] = field(repr=False)


TOKEN_PATTERN = re.compile(r"\[PERSON_[A-Za-z0-9_]+\]")

# 이 패턴들이 모든 개인정보를 탐지하는 것은 아닙니다.
PII_PATTERNS = (
    re.compile(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}"),
    re.compile(r"(?<!\d)01[016789][ -]?\d{3,4}[ -]?\d{4}(?!\d)"),
    re.compile(r"(?<!\d)\d{6}[ -]?[1-8]\d{6}(?!\d)"),
)


def prepare_request(data: CreditInput):
    token = f"[PERSON_{uuid4().hex}]"

    context = PrivacyContext(
        case_id=data.case_id,
        token_map={token: data.representative_name},
        originals=(
            data.representative_name,
            data.email,
            data.phone,
        ),
    )

    # 원본 전체를 복사하지 않고 외부 전송 허용 필드만 구성합니다.
    outbound = {
        "representative": token,
        "metrics": {
            "revenue_growth_pct": data.revenue_growth_pct,
            "operating_margin_pct": data.operating_margin_pct,
            "debt_to_equity_pct": data.debt_to_equity_pct,
        },
    }

    # case_id, company_id, email, phone은 전송하지 않습니다.
    return json.dumps(outbound, ensure_ascii=False), context


def inspect_text(text: str, context: PrivacyContext):
    for original in context.originals:
        if original in text:
            raise PolicyViolation("ORIGINAL_PII_DETECTED")

    for pattern in PII_PATTERNS:
        if pattern.search(text):
            raise PolicyViolation("PII_PATTERN_DETECTED")


def guarded_call(
    request_text: str,
    context: PrivacyContext,
    transport: Callable[[str], dict],
    inspector=inspect_text,
):
    # 검사 실패 또는 검사기 장애 시 transport는 실행되지 않습니다.
    inspector(request_text, context)
    return transport(request_text)


def restore_response(
    raw_text: str,
    context: PrivacyContext,
    expected_case_id: str,
):
    if context.case_id != expected_case_id:
        raise PolicyViolation("CASE_MISMATCH")

    # 복원하기 전에 응답 자체의 미처리 개인정보를 검사합니다.
    inspect_text(raw_text, context)

    result = AnalysisOutput.model_validate_json(
        normalize_json_response(raw_text)
    )

    if result.representative not in context.token_map:
        raise PolicyViolation("UNKNOWNUNKNOWN_REPRESENTATIVE_TOKEN")

    serialized = result.model_dump_json()

    for token in TOKEN_PATTERN.findall(serialized):
        if token not in context.token_map:
            raise PolicyViolation("UNKNOWN_PERSON_TOKEN")

    def restore(value: str) -> str:
        for token, original in context.token_map.items():
            value = value.replace(token, original)
        return value

    return {
        "representative": restore(result.representative),
        "summary": restore(result.summary),
        "review_points": [
            restore(item) for item in result.review_points
        ],
    }


def mock_transport(request_text: str):
    request = json.loads(request_text)

    response = {
        "representative": request["representative"],
        "summary": "매출은 전년 대비 15% 감소했습니다.",
        "review_points": [
            "채무상환 능력 검토를 위해 현금흐름과 상환 일정이 필요합니다."
        ],
    }

    return {
        "text": json.dumps(response, ensure_ascii=False),
        "usage": {},
        "stop_reason": "end_turn",
    }


def bedrock_transport(request_text: str):
    import os

    client = create_client()

    system_prompt = """
기업여신 심사지원 보조자로서 제공된 지표만 설명하세요.
representative의 가명 토큰을 입력 그대로 유지하세요.
토큰의 실제 인물을 추측하거나 새로운 토큰을 만들지 마세요.
신용등급이나 승인 여부를 결정하지 마세요.
비교 기준 없이 비율이 높거나 낮다고 단정하지 마세요.
전년도 영업이익률이 없으므로 유지·개선·악화를 판단하지 마세요.
현금흐름과 상환 일정 없이 채무상환 능력을 단정하지 마세요.
한국어로 다음 JSON 객체만 출력하세요.
{
  "representative": "입력받은 토큰",
  "summary": "관찰된 사실 요약",
  "review_points": ["추가로 확인할 사항"]
}
"""

    response = client.converse(
        modelId=os.environ["BEDROCK_MODEL_ID"],
        system=[{"text": system_prompt}],
        messages=[
            {
                "role": "user",
                "content": [{"text": request_text}],
            }
        ],
        inferenceConfig={"maxTokens": 1000},
    )

    return {
        "text": "\n".join(
            block["text"]
            for block in response["output"]["message"]["content"]
            if "text" in block
        ),
        "usage": response.get("usage", {}),
        "stop_reason": response.get("stopReason"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--live",
        action="store_true",
        help="가상 데이터를 실제 Bedrock으로 전송합니다.",
    )
    args = parser.parse_args()

    # 이름·연락처를 포함해 모두 실습용으로 구성한 가상 데이터입니다.
    data = CreditInput(
        case_id="CASE_001",
        company_id="SYNTHETIC_001",
        representative_name="가상대표",
        email="synthetic@example.invalid",
        phone="010-0000-0000",
        revenue_growth_pct=-15.0,
        operating_margin_pct=4.0,
        debt_to_equity_pct=200.0,
    )

    request_text, context = prepare_request(data)

    # 실습 화면: 실제 동적 전송 내용을 확인합니다.
    print("[처리 후 전송 후보 데이터]")
    print(request_text)

    audit = {
        "run_id": uuid4().hex,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode": "live" if args.live else "mock",
        "policy_version": "structured-pii-v1",
        "case_id": data.case_id,
        "tokenized_fields": ["representative_name"],
        "dropped_fields": ["email", "phone", "case_id", "company_id"],
        "status": "FAILED",
    }

    succeeded = False

    try:
        transport = bedrock_transport if args.live else mock_transport

        response = guarded_call(
            request_text,
            context,
            transport,
        )

        audit["usage"] = response["usage"]
        audit["stop_reason"] = response["stop_reason"]

        if response["stop_reason"] != "end_turn":
            raise PolicyViolation("INCOMPLETE_RESPONSE")

        restored = restore_response(
            response["text"],
            context,
            expected_case_id=data.case_id,
        )

        # 복원은 내부 처리 단계입니다. 이번에는 가상 데이터만 출력합니다.
        print("\n[내부 복원 결과: 가상 데이터]")
        print(json.dumps(restored, ensure_ascii=False, indent=2))

        audit["status"] = "PASS"
        succeeded = True

    except Exception as exc:
        # 예외 전문에는 요청 내용이 포함될 수 있어 기록하지 않습니다.
        audit["error_type"] = type(exc).__name__

        if isinstance(exc, PolicyViolation):
            audit["reason_code"] = str(exc)

        print(f"\n처리 실패: {audit['error_type']}")

    finally:
        output_dir = ROOT / "outputs"
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / f"step2_{audit['run_id']}.json"

        # 원문, 응답 본문, API 키, 복원 매핑은 감사 파일에 저장하지 않습니다.
        output_path.write_text(
            json.dumps(audit, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        print("\n[감사 기록]")
        print(json.dumps(audit, ensure_ascii=False, indent=2))
        print(f"\n저장 위치: {output_path}")

    if not succeeded:
        raise SystemExit(1)


if __name__ == "__main__":
    main()