import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchText
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

load_dotenv()

QDRANT_HOST     = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT     = int(os.getenv("QDRANT_PORT", 6333))
COLLECTION_NAME = "business_reviews"

embedder = SentenceTransformer("all-MiniLM-L6-v2")
client   = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

# common city names to detect in queries
KNOWN_CITIES = [
    "nashville", "philadelphia", "tucson", "tampa", "new orleans",
    "indianapolis", "saint louis", "st. louis", "reno", "boise",
    "edmonton", "clearwater", "wilmington", "cherry hill"
]

def detect_city(query: str) -> str | None:
    query_lower = query.lower()
    for city in KNOWN_CITIES:
        if city in query_lower:
            return city.title()
    return None

def semantic_search(query: str, top_k: int = 20, city: str = None) -> list:
    vector = embedder.encode(query).tolist()

    # if city detected, filter by it
    query_filter = None
    if city:
        query_filter = Filter(
            must=[FieldCondition(
                key="city",
                match=MatchText(text=city)
            )]
        )

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=vector,
        limit=top_k,
        with_payload=True,
        query_filter=query_filter
    ).points

    # if city filter returns too few results, fall back to no filter
    if city and len(results) < 3:
        results = client.query_points(
            collection_name=COLLECTION_NAME,
            query=vector,
            limit=top_k,
            with_payload=True,
        ).points

    return results

def hybrid_search(query: str, top_k: int = 8) -> list[dict]:
    city = detect_city(query)
    candidates = semantic_search(query, top_k=30, city=city)
    if not candidates:
        return []

    texts     = [c.payload.get("text", "") for c in candidates]
    tokenized = [t.lower().split() for t in texts]
    bm25      = BM25Okapi(tokenized)
    scores    = bm25.get_scores(query.lower().split())

    combined = []
    for i, candidate in enumerate(candidates):
        semantic_score = candidate.score
        bm25_score     = scores[i]
        max_bm25       = max(scores) if max(scores) > 0 else 1
        norm_bm25      = bm25_score / max_bm25
        final_score    = (0.7 * semantic_score) + (0.3 * norm_bm25)
        combined.append((candidate, final_score))

    combined.sort(key=lambda x: x[1], reverse=True)
    top = combined[:top_k]

    chunks = []
    for candidate, score in top:
        chunks.append({
            "business_name": candidate.payload.get("business_name"),
            "city":          candidate.payload.get("city"),
            "state":         candidate.payload.get("state"),
            "biz_stars":     candidate.payload.get("biz_stars"),
            "categories":    candidate.payload.get("categories"),
            "review_stars":  candidate.payload.get("review_stars"),
            "review_date":   candidate.payload.get("review_date"),
            "text":          candidate.payload.get("text"),
            "score":         round(score, 3),
        })

    return chunks


def format_chunks_for_llm(chunks: list[dict]) -> str:
    if not chunks:
        return "No relevant reviews found."
    formatted = []
    for i, c in enumerate(chunks, 1):
        formatted.append(
            f"[Review {i}]\n"
            f"Business : {c['business_name']} ({c['city']}, {c['state']})\n"
            f"Rating   : {c['review_stars']}/5 | Overall: {c['biz_stars']}/5\n"
            f"Date     : {c['review_date']}\n"
            f"Text     : {c['text']}\n"
        )
    return "\n".join(formatted)


if __name__ == "__main__":
    results = hybrid_search("burger places in Nashville")
    print(format_chunks_for_llm(results))
