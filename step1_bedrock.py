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