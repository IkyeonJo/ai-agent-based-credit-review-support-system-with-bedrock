# =============================================================================
# 1단계: Bedrock Claude 호출 및 기업 재무 분석 기초
# =============================================================================
#
# 목적
# - 로컬 Python에서 AWS Bedrock Claude를 호출하는 방법을 학습합니다.
# - 재무 수치 계산과 LLM의 설명·해석 역할을 분리합니다.
# - 후속 Agent에서 활용할 구조화된 응답과 실행 기록의 기반을 만듭니다.
#
# 처리 순서
# 1. 프로젝트 루트의 .env에서 인증·리전·모델 설정을 읽습니다.
# 2. 가상 기업의 재무 데이터를 준비합니다.
# 3. Python으로 매출 증감률, 영업이익률, 부채비율을 계산합니다.
# 4. 재무 데이터와 계산 결과를 Bedrock Converse API로 전달합니다.
# 5. 응답 전체를 감싸는 JSON 코드 블록이 있으면 제거합니다.
# 6. Pydantic으로 JSON 구조, 필수 항목, 자료형을 검증합니다.
# 7. 분석 결과와 실행 메타데이터를 JSON 파일로 저장합니다.
#
# 인증 방식: BEDROCK_AUTH_MODE로 선택
# - api_key: AWS_BEARER_TOKEN_BEDROCK 환경변수 사용
# - profile: AWS_PROFILE에 지정한 AWS 인증 프로파일 사용
# - 공통 설정: AWS_REGION, BEDROCK_MODEL_ID
# - 단기 API 키가 만료되면 .env 값을 갱신한 뒤 다시 실행합니다.
# - API 키와 AWS 자격증명은 코드·Git·실행 결과에 기록하지 않습니다.
#
# 입력 데이터
# - 실제 고객정보가 아닌 고정된 가상 제조업체 데이터입니다.
# - 금액 단위는 백만원입니다.
# - total_equity는 자본총계이며 자본금과 구분합니다.
#
# Python 계산 기대값
# - 매출 증감률: -15.0%
# - 영업이익률: 4.0%
# - 부채비율: 200.0%
#
# LLM 분석 원칙
# - 제공된 수치와 사실만 사용합니다.
# - 업종 평균, 은행 기준, 신용등급, 대출 승인 여부를 만들어내지 않습니다.
# - 전년도 영업이익률 없이 유지·개선·악화를 판단하지 않습니다.
# - 비교 기준 없이 재무비율의 높고 낮음을 단정하지 않습니다.
# - 현금흐름·상환 일정이 부족하면 채무상환 능력 판단을 유보합니다.
# - 관찰된 사실과 추가 확인 사항을 구분합니다.
#
# 출력 스키마
# - summary: 비어 있지 않은 분석 요약 문자열
# - risk_factors: 하나 이상의 문자열을 포함한 배열
# - additional_documents: 하나 이상의 문자열을 포함한 배열
# - 위 항목 외의 추가 필드는 허용하지 않습니다.
#
# 검증과 오류 처리
# - stop_reason이 end_turn인지 확인합니다.
# - JSON 코드 블록 제거 후에도 기존 스키마 검증을 그대로 수행합니다.
# - 검증 실패 시 validation_error와 원본 응답을 저장합니다.
# - 원본 응답 저장은 가상 데이터 실습에 한정한 디버그 기능입니다.
# - API 호출 자체에서 발생한 예외는 현재 코드에서 별도로 저장하지 않습니다.
# - SDK 자동 재시도는 사용하지 않습니다.
#
# 결과 파일
# - outputs/<실행ID>.json
# - 실행 ID, UTC 생성 시각, 인증 방식, 요청 리전, 모델 ID
# - 호출 소요시간, 토큰 사용량, 종료 사유, AWS 요청 ID
# - 계산 지표, 코드 블록 제거 여부, 검증 상태, 분석 결과
#
# 해석상 주의점
# - validation_status: PASS는 응답 형식 검증 통과를 의미합니다.
#   분석 내용의 사실성·업무 타당성까지 보장하지 않습니다.
# - JSON은 프롬프트로 요청하고 Pydantic으로 검증합니다.
#   모델 API의 강제 구조화 출력 기능을 적용한 것은 아닙니다.
# - request_region은 요청 리전이며 실제 추론 처리 위치와 다를 수 있습니다.
# - 개인정보 처리·전송 차단·RAG·Multi-Agent 제어는 이후 단계에서 추가합니다.
#
# 실행: 프로젝트 루트에서
#   uv run python step1_bedrock.py
#
# 완료 기준
# - 실제 Bedrock 호출 성공
# - 계산 지표 확인
# - validation_status: PASS
# - 실행 결과 JSON 저장 및 분석 내용 수동 확인
# =============================================================================

import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import boto3
from botocore.config import Config
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field


ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env", override=True)


class CreditAnalysis(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    summary: str = Field(min_length=1)
    risk_factors: list[str] = Field(min_length=1)
    additional_documents: list[str] = Field(min_length=1)


def normalize_json_response(text: str) -> str:
    """응답 전체를 감싸는 JSON 코드 블록만 제거합니다."""
    text = text.strip()

    match = re.fullmatch(
        r"```(?:json)?[ \t]*\r?\n(.*?)\r?\n```",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )

    return match.group(1).strip() if match else text


def create_client():
    mode = os.getenv("BEDROCK_AUTH_MODE", "api_key")
    region = os.environ["AWS_REGION"]

    config = Config(
        connect_timeout=10,
        read_timeout=120,
        # 1단계에서는 오류를 직접 확인하도록 자동 재시도를 하지 않습니다.
        retries={"mode": "standard", "total_max_attempts": 1},
    )

    if mode == "api_key":
        if not os.getenv("AWS_BEARER_TOKEN_BEDROCK"):
            raise ValueError("Bedrock API 키가 설정되지 않았습니다.")

        return boto3.client(
            "bedrock-runtime",
            region_name=region,
            config=config,
        )

    if mode == "profile":
        # Profile 모드에서는 기존 API 키 환경변수를 제거합니다.
        os.environ.pop("AWS_BEARER_TOKEN_BEDROCK", None)

        session = boto3.Session(
            profile_name=os.environ["AWS_PROFILE"],
            region_name=region,
        )

        return session.client(
            "bedrock-runtime",
            config=config,
        )

    raise ValueError(
        "BEDROCK_AUTH_MODE는 api_key 또는 profile이어야 합니다."
    )


def main():
    model_id = os.environ["BEDROCK_MODEL_ID"]

    # 실제 고객정보가 아닌 고정된 가상 데이터입니다.
    company = {
        "company_id": "SYNTHETIC_COMPANY_001",
        "industry": "제조업",
        "unit": "백만원",
        "revenue_previous": 10000,
        "revenue_current": 8500,
        "operating_profit_current": 340,
        "total_liabilities": 6000,
        "total_equity": 3000,
    }

    # 재무비율은 Python으로 계산하고, LLM에는 해석을 요청합니다.
    metrics = {
        "revenue_growth_pct": round(
            (
                company["revenue_current"]
                / company["revenue_previous"]
                - 1
            ) * 100,
            2,
        ),
        "operating_margin_pct": round(
            company["operating_profit_current"]
            / company["revenue_current"] * 100,
            2,
        ),
        "debt_to_equity_pct": round(
            company["total_liabilities"]
            / company["total_equity"] * 100,
            2,
        ),
    }

    system_prompt = """
당신은 기업여신 심사지원 분석 보조자입니다.
제공된 가상 기업자료와 이미 계산된 지표만 사용하세요.

[분석 원칙]
- 외부 사실, 은행 기준, 신용등급, 대출 승인 여부를 만들어내지 마세요.
- 업종 평균이 없으므로 업종 평균과 비교하지 마세요.
- 수치는 재계산하지 말고 제공된 값을 사용하세요.
- total_equity는 자본총계이며 자본금으로 표현하지 마세요.
- 전년도 영업이익률이 없으므로 현재 영업이익률을
  '유지', '개선', '악화'했다고 표현하지 마세요.
- 비교 기준이 없으면 부채비율이나 영업이익률을
  '높다', '과다하다', '낮다'고 단정하지 마세요.
- 채무상환 능력은 현금흐름과 상환 일정이 없으면 판단을 유보하세요.
- 전년 대비 매출 감소를 장기간 지속된 추세로 단정하지 마세요.
- 관찰된 사실과 추가 확인이 필요한 사항을 구분하세요.
- 자료가 부족하면 추가로 필요한 서류를 명시하세요.
- 한국어로 답변하세요.

[출력 형식]
마크다운, 코드 블록, 앞뒤 설명 없이 JSON 객체 하나만 출력하세요.
아래 세 가지 키만 사용하세요.
summary는 비어 있지 않은 문자열이어야 합니다.
risk_factors와 additional_documents는 각각 하나 이상의 문자열을
포함하는 배열이어야 합니다.

{
  "summary": "분석 요약",
  "risk_factors": ["위험요인 또는 확인이 필요한 사항"],
  "additional_documents": ["추가로 필요한 서류"]
}
"""

    payload = {
        "company": company,
        "calculated_metrics": metrics,
    }

    client = create_client()
    run_id = uuid4().hex
    started = time.perf_counter()

    response = client.converse(
        modelId=model_id,
        system=[{"text": system_prompt}],
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "text": json.dumps(
                            payload,
                            ensure_ascii=False,
                        )
                    }
                ],
            }
        ],
        inferenceConfig={"maxTokens": 1200},
    )

    elapsed = round(time.perf_counter() - started, 3)

    raw_text = "\n".join(
        block["text"]
        for block in response["output"]["message"]["content"]
        if "text" in block
    )

    normalized_text = normalize_json_response(raw_text)

    record = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "auth_mode": os.getenv("BEDROCK_AUTH_MODE", "api_key"),
        "request_region": os.environ["AWS_REGION"],
        "model_id": model_id,
        "elapsed_seconds": elapsed,
        "usage": response.get("usage", {}),
        "stop_reason": response.get("stopReason"),
        "request_id": response["ResponseMetadata"].get("RequestId"),
        "calculated_metrics": metrics,
        "code_fence_removed": normalized_text != raw_text.strip(),
        "validation_status": "FAILED",
    }

    try:
        if response.get("stopReason") != "end_turn":
            raise ValueError(
                "응답이 정상적으로 완료되지 않았습니다. "
                f"stop_reason={response.get('stopReason')}"
            )

        # 코드 블록을 제거한 뒤에도 JSON 구조 검증은 그대로 수행합니다.
        analysis = CreditAnalysis.model_validate_json(normalized_text)

        record["analysis"] = analysis.model_dump()
        record["validation_status"] = "PASS"

    except ValueError as exc:
        # 가상 데이터 실습에 한정한 디버그 저장입니다.
        record["validation_error"] = str(exc)
        record["raw_response_for_synthetic_debug"] = raw_text

        print(f"응답 검증 실패:\n{exc}")
        print(f"stop_reason: {response.get('stopReason')}")

    output_dir = ROOT / "outputs"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / f"{run_id}.json"

    output_path.write_text(
        json.dumps(record, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(record, ensure_ascii=False, indent=2))
    print(f"\n저장 위치: {output_path}")

    if record["validation_status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()