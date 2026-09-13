# =============================================================================
# 3-B단계: 로컬 E5 임베딩 + PostgreSQL/pgvector RAG
# =============================================================================
#
# 목적
# - 3-A의 TF-IDF 검색을 로컬 의미 기반 임베딩 검색으로 교체합니다.
# - 문서, 적용 기간, 기업 범위, 전송 정책, 벡터를 PostgreSQL에 저장합니다.
# - 기존 Gateway와 출처 검증을 재사용합니다.
#
# 처리 흐름
# 1. 최초 모델 다운로드 및 로컬 저장
# 2. 문서에 passage: 접두어를 붙여 임베딩 생성
# 3. PostgreSQL에 문서와 384차원 벡터 적재
# 4. 서버 측 권한 확인
# 5. 질문에 query: 접두어를 붙여 로컬 임베딩 생성
# 6. SQL에서 기업 범위·적용일을 제한한 뒤 코사인 거리순 검색
# 7. 전송 정책·개인정보 검사 후 모델 호출
# 8. 응답 출처 ID와 인용문 검증
#
# 범위와 한계
# - 가상 문서 6개를 사용하는 학습용 구현입니다.
# - 로그인·SSO·DB Row-Level Security는 아직 적용하지 않습니다.
# - 현재 권한 통제는 애플리케이션과 SQL 조건에 구현되어 있습니다.
# - 근사 검색 인덱스 없이 정확 검색을 사용합니다.
# - 의미 검색 점수는 정답 확률이 아닙니다.
# - 무관한 질문의 답변 거부 기준은 별도 평가로 보완해야 합니다.
#
# 실행
#   uv run python step3b_pgvector_rag.py --init
#   uv run python step3b_pgvector_rag.py --search-only
#   uv run python step3b_pgvector_rag.py
#   uv run python step3b_pgvector_rag.py --live
# =============================================================================

import argparse
import json
from datetime import date, datetime, timezone
from functools import lru_cache
from pathlib import Path
from uuid import uuid4

import psycopg
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer

# 이 모듈의 import 과정에서 기존 .env 설정도 로드됩니다.
from step3_secure_rag import (
    DOCUMENTS,
    SERVER_GRANTS,
    Document,
    bedrock_transport,
    build_payload,
    mock_transport,
    validate_answer,
)
from step2_privacy_gateway import (
    PolicyViolation,
    PrivacyContext,
    guarded_call,
    inspect_text,
)


ROOT = Path(__file__).resolve().parent
MODEL_NAME = "intfloat/multilingual-e5-small"
MODEL_DIR = ROOT / ".models" / "multilingual-e5-small"
DIMENSION = 384


@lru_cache(maxsize=1)
def get_model():
    if (MODEL_DIR / "modules.json").exists():
        model = SentenceTransformer(
            str(MODEL_DIR),
            device="cpu",
            local_files_only=True,
        )
    else:
        print("최초 임베딩 모델 다운로드 중...")
        model = SentenceTransformer(MODEL_NAME, device="cpu")
        MODEL_DIR.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(MODEL_DIR))

    if model.get_sentence_embedding_dimension() != DIMENSION:
        raise ValueError("EMBEDDING_DIMENSION_MISMATCH")

    return model


def embed(texts, prefix):
    return get_model().encode(
        [f"{prefix}: {text}" for text in texts],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )


def connect_db():
    # psycopg가 .env에서 로드된 PGHOST, PGPORT 등의 환경변수를 사용합니다.
    return psycopg.connect(connect_timeout=5)


def initialize():
    # 먼저 모델을 준비해 다운로드 중 DB 트랜잭션이 열려 있지 않게 합니다.
    vectors = embed([doc.text for doc in DOCUMENTS], "passage")

    with connect_db() as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(conn)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS credit_chunks (
                doc_id TEXT PRIMARY KEY,
                company_id TEXT,
                valid_from DATE NOT NULL,
                valid_to DATE,
                cloud_allowed BOOLEAN NOT NULL,
                content TEXT NOT NULL,
                embedding_model TEXT NOT NULL,
                embedding VECTOR(384) NOT NULL,
                CHECK (valid_to IS NULL OR valid_to > valid_from)
            )
        """)

        for doc, vector in zip(DOCUMENTS, vectors):
            conn.execute(
                """
                INSERT INTO credit_chunks (
                    doc_id, company_id, valid_from, valid_to,
                    cloud_allowed, content, embedding_model, embedding
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (doc_id) DO UPDATE SET
                    company_id = EXCLUDED.company_id,
                    valid_from = EXCLUDED.valid_from,
                    valid_to = EXCLUDED.valid_to,
                    cloud_allowed = EXCLUDED.cloud_allowed,
                    content = EXCLUDED.content,
                    embedding_model = EXCLUDED.embedding_model,
                    embedding = EXCLUDED.embedding
                """,
                (
                    doc.doc_id,
                    doc.company_id,
                    doc.valid_from,
                    doc.valid_to,
                    doc.cloud_allowed,
                    doc.text,
                    MODEL_NAME,
                    vector,
                ),
            )

        count = conn.execute(
            "SELECT COUNT(*) FROM credit_chunks"
        ).fetchone()[0]

    print(f"적재 완료: 이번 처리 {len(DOCUMENTS)}개 / DB 전체 {count}개")


def search(question, user_id, company_id, as_of, top_k=2):
    # 권한은 임베딩·DB 검색 전에 검사합니다.
    if company_id not in SERVER_GRANTS.get(user_id, frozenset()):
        raise PolicyViolation("CASE_ACCESS_DENIED")

    if not question.strip():
        raise PolicyViolation("EMPTY_QUESTION")

    if not 1 <= top_k <= 20:
        raise ValueError("INVALID_TOP_K")

    query_vector = embed([question], "query")[0]

    with connect_db() as conn:
        register_vector(conn)

        rows = conn.execute(
            """
            SELECT
                doc_id, company_id, valid_from, valid_to,
                cloud_allowed, content,
                1 - (embedding <=> %s) AS similarity
            FROM credit_chunks
            WHERE
                (company_id IS NULL OR company_id = %s)
                AND valid_from <= %s
                AND (valid_to IS NULL OR %s < valid_to)
                AND embedding_model = %s
            ORDER BY embedding <=> %s, doc_id
            LIMIT %s
            """,
            (
                query_vector,
                company_id,
                as_of,
                as_of,
                MODEL_NAME,
                query_vector,
                top_k,
            ),
        ).fetchall()

    results = []

    for row in rows:
        doc = Document(
            doc_id=row[0],
            company_id=row[1],
            valid_from=row[2],
            valid_to=row[3],
            cloud_allowed=row[4],
            text=row[5],
        )

        # SQL 결과를 애플리케이션 경계에서도 재검사합니다.
        if (
            doc.company_id not in (None, company_id)
            or doc.valid_from > as_of
            or (doc.valid_to is not None and as_of >= doc.valid_to)
        ):
            raise PolicyViolation("RETRIEVAL_SCOPE_VIOLATION")

        results.append((doc, float(row[6])))

    return results


def answer_from_hits(question, hits, context, transport):
    docs = [doc for doc, _ in hits]

    # 문서 없음과 전송 금지 여부는 기존 3-A 함수가 검사합니다.
    payload = build_payload(question, docs)

    # 질문과 검색 근거 전체에 기존 2단계 검사를 적용합니다.
    response = guarded_call(payload, context, transport)

    if response["stop_reason"] != "end_turn":
        raise PolicyViolation("INCOMPLETE_RESPONSE")

    inspect_text(response["text"], context)
    answer = validate_answer(response["text"], docs)

    return answer, response["usage"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--init", action="store_true")
    parser.add_argument("--search-only", action="store_true")
    parser.add_argument("--live", action="store_true")
    parser.add_argument(
        "--question",
        default="현금흐름과 채무상환 능력을 검토하려면 어떤 서류가 필요한가?",
    )
    args = parser.parse_args()

    if args.init:
        initialize()
        return

    audit = {
        "run_id": uuid4().hex,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "retriever": "pgvector-exact-cosine",
        "embedding_model": MODEL_NAME,
        "mode": (
            "search-only" if args.search_only
            else "live" if args.live
            else "mock"
        ),
        "status": "FAILED",
    }

    succeeded = False

    try:
        hits = search(
            args.question,
            user_id="reviewer_a",
            company_id="SYNTHETIC_001",
            as_of=date(2026, 9, 13),
        )

        audit["retrieved"] = [
            {
                "source_id": doc.doc_id,
                "similarity": round(score, 4),
            }
            for doc, score in hits
        ]

        print("[검색 결과: 가상 문서]")
        for doc, score in hits:
            print(
                f"\n{doc.doc_id} / 유사도={score:.4f}"
                f" / 클라우드전송={doc.cloud_allowed}"
            )
            print(doc.text)

        if not args.search_only:
            context = PrivacyContext(
                case_id="CASE_001",
                token_map={},
                originals=(
                    "가상대표",
                    "synthetic@example.invalid",
                    "010-0000-0000",
                ),
            )

            answer, usage = answer_from_hits(
                args.question,
                hits,
                context,
                bedrock_transport if args.live else mock_transport,
            )

            audit["usage"] = usage
            audit["cited_source_ids"] = sorted(
                {claim.source_id for claim in answer.claims}
            )

            print("\n[응답]")
            print(answer.model_dump_json(indent=2))

        audit["status"] = "PASS"
        succeeded = True

    except Exception as exc:
        audit["error_type"] = type(exc).__name__
        if isinstance(exc, PolicyViolation):
            audit["reason_code"] = str(exc)

        print(f"처리 실패: {audit['error_type']}")

    finally:
        output_dir = ROOT / "outputs"
        output_dir.mkdir(exist_ok=True)
        path = output_dir / f"step3b_{audit['run_id']}.json"

        path.write_text(
            json.dumps(audit, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        print("\n[감사 기록]")
        print(json.dumps(audit, ensure_ascii=False, indent=2))
        print(f"\n저장 위치: {path}")

    if not succeeded:
        raise SystemExit(1)


if __name__ == "__main__":
    main()