# RAG Evaluation Script
# Measures retrieval quality and generation faithfulness
# Run: python3 app/evaluate.py

from search import hybrid_search, format_chunks_for_llm
from llm import chat

# Test queries with known expected businesses
TEST_CASES = [
    {
        "query":    "pros and cons of Papa Murphy's in Tucson",
        "expected": "Papa Murphy's"
    },
    {
        "query":    "restaurants in Nashville",
        "expected": "Nashville"
    },
    {
        "query":    "best pizza place",
        "expected": "pizza"
    },
    {
        "query":    "Passport Health Saint Louis",
        "expected": "Passport Health"
    },
    {
        "query":    "burger places in Nashville",
        "expected": "Nashville"
    },
]

def precision_at_k(retrieved_chunks, expected_keyword, k=5):
    relevant = sum(
        1 for chunk in retrieved_chunks[:k]
        if expected_keyword.lower() in chunk['business_name'].lower()
        or expected_keyword.lower() in chunk['city'].lower()
    )
    return round(relevant / k, 2)

def mean_reciprocal_rank(retrieved_chunks, expected_keyword):
    for i, chunk in enumerate(retrieved_chunks, 1):
        if (expected_keyword.lower() in chunk['business_name'].lower()
                or expected_keyword.lower() in chunk['city'].lower()):
            return round(1 / i, 2)
    return 0.0

def faithfulness_score(response, context_chunks):
    # simple proxy: what % of sentences in response
    # contain words from the context
    context_words = set()
    for chunk in context_chunks:
        context_words.update(chunk['text'].lower().split())

    sentences  = [s.strip() for s in response.split('.') if len(s.strip()) > 10]
    if not sentences:
        return 0.0

    faithful = 0
    for sentence in sentences:
        words = set(sentence.lower().split())
        overlap = words & context_words
        if len(overlap) / max(len(words), 1) > 0.3:
            faithful += 1

    return round(faithful / len(sentences), 2)

def run_evaluation():
    print("=" * 60)
    print("RAG EVALUATION REPORT")
    print("=" * 60)

    total_p5  = []
    total_mrr = []
    total_fth = []

    for i, test in enumerate(TEST_CASES, 1):
        print(f"\nTest {i}: {test['query']}")
        print("-" * 40)

        # Retrieval
        chunks  = hybrid_search(test['query'])
        p5      = precision_at_k(chunks, test['expected'])
        mrr     = mean_reciprocal_rank(chunks, test['expected'])

        # Generation
        context  = format_chunks_for_llm(chunks)
        response = chat(test['query'], context, history=[])
        faith    = faithfulness_score(response, chunks)

        print(f"Precision@5      : {p5}")
        print(f"MRR              : {mrr}")
        print(f"Faithfulness     : {faith}")

        total_p5.append(p5)
        total_mrr.append(mrr)
        total_fth.append(faith)

    print("\n" + "=" * 60)
    print("OVERALL SCORES")
    print("=" * 60)
    print(f"Avg Precision@5  : {round(sum(total_p5)  / len(total_p5),  2)}")
    print(f"Avg MRR          : {round(sum(total_mrr) / len(total_mrr), 2)}")
    print(f"Avg Faithfulness : {round(sum(total_fth) / len(total_fth), 2)}")
    print("=" * 60)

if __name__ == "__main__":
    run_evaluation()
