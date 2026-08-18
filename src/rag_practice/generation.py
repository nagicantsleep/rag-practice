import ollama

from .store import SearchResult


SYSTEM_PROMPT = """You are a RAG learning assistant.
Answer only from the retrieved context below. If the context is insufficient, say so.
Cite supporting chunks using [1], [2], etc. Do not invent citations.
"""


def build_context(results: list[SearchResult]) -> str:
    return "\n\n".join(
        f"[{i}] source={result.source} chunk={result.chunk_index}\n{result.text}"
        for i, result in enumerate(results, start=1)
    )


def generate_answer(
    question: str,
    results: list[SearchResult],
    model: str = "llama3.2",
) -> str:
    context = build_context(results)
    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Retrieved context:\n\n{context}\n\nQuestion: {question}",
            },
        ],
    )
    return response.message.content
