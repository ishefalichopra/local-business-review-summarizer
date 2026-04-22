import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL  = "llama-3.3-70b-versatile"

# Prompting Framework: RISEN + RAG
#
# The system prompt is structured using the RISEN framework:
#   Role        - defines what the assistant is and what it does
#   Instructions - rules the model must follow (no hallucination, stay grounded)
#   Steps       - structured pros/cons output format for business summaries
#   End goal    - help users evaluate local businesses through review analysis
#   Narrowing   - restrict the model to only use the provided review context
#
# On top of RISEN, two additional techniques are layered:
#   RAG (Retrieval Augmented Generation) - real review chunks from Qdrant
#   are injected into every prompt so the model responds based on actual
#   customer data rather than its own training knowledge.
#
#   Multi-turn memory - last 6 conversation turns are passed with every
#   request so follow-up questions like "what about their desserts?" work.

# Role + Instructions + End goal + Narrowing

SYSTEM_PROMPT = """You are a local business review analyst.
    Your job is to fetch relevant reviews for a business or location and deliver
    a concise, structured summary of what customers actually experienced.

    Rules:
    - Only use information explicitly stated in the review excerpts provided
    - Never infer or assume pros/cons that are not directly mentioned by reviewers

    - Never calculate your own ratings — always use the biz_stars field as the official rating
    - If reviews don't have enough info to answer, say so honestly
    - Always mention the business name and city
    - Every first-time business summary MUST include a pros and cons breakdown

    Steps for first-time business summary:
    📍 [Business Name] — [City, State]

    ⭐ Rating: [biz_stars from data]/5  |  Category: [type of business]

    PROS

    - [only what reviewers explicitly praised]

    CONS

    - [only what reviewers explicitly complained about]

    VERDICT

    [2-3 sentence summary based strictly on the reviews]


    For follow-up questions answer specifically what was asked.
    Use plain prose for short factual questions."""


def build_rag_prompt(user_message: str, review_context: str) -> str:
    # RAG context injection - retrieved review chunks are passed here
    # model is explicitly told to use only this data
    return f"""Here are real customer review excerpts for the business or location asked about.
Analyze these reviews and extract the key pros and cons.
Use ONLY this information — do not use outside knowledge.

{review_context}

User question: {user_message}"""


def chat(user_message: str, review_context: str, history: list[dict]) -> str:
    # system prompt first
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # multi-turn memory - capped at 6 turns to stay within token limits
    messages += history[-6:]

    # RAG prompt with user question
    messages.append({
        "role":    "user",
        "content": build_rag_prompt(user_message, review_context)
    })

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=1,
        max_tokens=1024,
    )

    return response.choices[0].message.content


if __name__ == "__main__":
    from search import search_reviews, format_chunks_for_llm

    query   = "Tell me about burger places in Nashville"
    chunks  = search_reviews(query)
    context = format_chunks_for_llm(chunks)
    reply   = chat(query, context, history=[])
    print(reply)
