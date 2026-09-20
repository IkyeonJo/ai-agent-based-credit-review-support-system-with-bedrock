# =============================================================================
# 6단계: 전체 자동 평가 및 보고서 생성
# =============================================================================
#
# 실행 대상
# - 개인정보 Gateway
# - 권한·적용일·출처 검증 RAG
# - pgvector 통합
# - Multi-Agent 오케스트레이션
# - Agent 통합 오류 분류
# - 웹 API·직원 검토
#
# 사전 조건
# - PostgreSQL 실행
# - 3-B 문서 적재 및 로컬 모델 준비 완료
#
# Bedrock 호출
# - 아래 지정된 테스트들은 실제 Bedrock을 호출하지 않습니다.
#
# 산출물
# - outputs/evaluation/<평가ID>.json
# - outputs/evaluation/<평가ID>.md
#
# 실행
#   uv run python step6_evaluate.py
# =============================================================================

import hashlib
import io
import json
import platform
import time
import unittest
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from uuid import uuid4


ROOT = Path(__file__).resolve().parent

TEST_MODULES = [
    ("개인정보 Gateway", "test_step2_privacy_gateway"),
    ("RAG 권한·시점·출처", "test_step3_secure_rag"),
    ("pgvector 통합", "test_step3b_pgvector_rag"),
    ("오케스트레이션", "test_step4_orchestration"),
    ("Agent 통합", "test_step4b_integrated_agents"),
    ("웹 API·직원 검토", "test_step6_api_review"),
]

PACKAGES = [
    "boto3",
    "pydantic",
    "scikit-learn",
    "sentence-transformers",
    "psycopg",
    "pgvector",
    "langgraph",
    "langgraph-checkpoint-sqlite",
    "fastapi",
    "httpx",
]


def package_versions():
    result = {}

    for name in PACKAGES:
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = "NOT_INSTALLED"

    return result


def source_hashes():
    # 평가 당시의 코드를 식별합니다. .env와 실행 DB는 읽지 않습니다.
    paths = set(ROOT.glob("step*.py"))
    paths.update(ROOT.glob("test_step*.py"))

    for name in ("pyproject.toml", "uv.lock"):
        path = ROOT / name
        if path.exists():
            paths.add(path)

    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(paths)
    }


def main():
    evaluation_id = (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + "_"
        + uuid4().hex[:8]
    )

    groups = []
    all_started = time.perf_counter()

    for label, module_name in TEST_MODULES:
        print(f"\n[{label}] 실행 중")

        started = time.perf_counter()
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromName(module_name)

        buffer = io.StringIO()
        result = unittest.TextTestRunner(
            stream=buffer,
            verbosity=2,
        ).run(suite)

        print(buffer.getvalue())

        failed = len(result.failures)
        errors = len(result.errors)
        skipped = len(result.skipped)
        expected_failures = len(result.expectedFailures)
        unexpected_successes = len(result.unexpectedSuccesses)

        passed = (
            result.testsRun
            - failed
            - errors
            - skipped
            - expected_failures
            - unexpected_successes
        )

        if not result.wasSuccessful():
            status = "FAILED"
        elif skipped or expected_failures or result.testsRun == 0:
            status = "INCOMPLETE"
        else:
            status = "PASS"

        groups.append({
            "group": label,
            "module": module_name,
            "status": status,
            "total": result.testsRun,
            "passed": passed,
            "failed": failed,
            "errors": errors,
            "skipped": skipped,
            "expected_failures": expected_failures,
            "unexpected_successes": unexpected_successes,
            "elapsed_seconds": round(
                time.perf_counter() - started, 3
            ),
            # 상세 traceback은 콘솔에서 확인합니다.
            "failed_test_ids": [
                test.id()
                for test, _ in result.failures + result.errors
            ],
        })

    total = sum(group["total"] for group in groups)
    passed = sum(group["passed"] for group in groups)

    if any(group["status"] == "FAILED" for group in groups):
        overall = "FAILED"
    elif any(group["status"] == "INCOMPLETE" for group in groups):
        overall = "INCOMPLETE"
    else:
        overall = "PASS"

    report = {
        "evaluation_id": evaluation_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "자동 기능·통제 테스트",
        "overall_status": overall,
        "total": total,
        "passed": passed,
        "elapsed_seconds": round(
            time.perf_counter() - all_started, 3
        ),
        "python": platform.python_version(),
        "packages": package_versions(),
        "source_sha256": source_hashes(),
        "groups": groups,
        "not_evaluated": [
            "실제 Claude 분석의 의미적 정확성",
            "자유문서 전체의 개인정보 탐지 성능",
            "프롬프트 인젝션에 대한 종합적인 방어 성능",
            "실제 로그인·SSO·운영 권한 관리",
            "금융기관의 실제 보안 요건 충족 여부",
            "다수 영업점 동시 사용 성능",
        ],
    }

    lines = [
        "# 기업여신 심사지원 자동 평가",
        "",
        f"- 평가 ID: `{evaluation_id}`",
        f"- 결과: **{overall}**",
        f"- 통과: **{passed}/{total}**",
        f"- Python: `{report['python']}`",
        "",
        "| 평가 영역 | 전체 | 통과 | 실패 | 오류 | 상태 |",
        "|---|---:|---:|---:|---:|---|",
    ]

    for group in groups:
        lines.append(
            f"| {group['group']} | {group['total']} | "
            f"{group['passed']} | {group['failed']} | "
            f"{group['errors']} | {group['status']} |"
        )

    lines.extend([
        "",
        "## 해석",
        "",
        "이 결과는 정의된 자동 테스트의 통과 여부입니다.",
        "운영 보안 적합성이나 LLM 답변의 정확성을 보장하지 않습니다.",
        "평가 실행시간은 업무 한 건의 처리시간이 아닙니다.",
        "",
        "## 별도 평가가 필요한 항목",
        "",
    ])

    lines.extend(
        f"- {item}" for item in report["not_evaluated"]
    )

    output_dir = ROOT / "outputs" / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / f"{evaluation_id}.json"
    md_path = output_dir / f"{evaluation_id}.md"

    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    md_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"\n자동 평가 결과: {overall} / {passed}/{total}")
    print(f"JSON: {json_path}")
    print(f"Markdown: {md_path}")

    if overall != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()