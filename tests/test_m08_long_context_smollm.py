from __future__ import annotations

from rag_practice.long_context.smollm import (
    MAX_NEW_TOKENS,
    MODEL_ID,
    MODEL_REVISION,
    build_messages,
)


def test_smollm_checkpoint_is_exactly_pinned() -> None:
    assert MODEL_ID == "HuggingFaceTB/SmolLM2-135M-Instruct"
    assert MODEL_REVISION == "12fd25f77366fa6b3b4b768ec3050bf629380bac"
    assert MAX_NEW_TOKENS == 32


def test_smollm_prompt_contains_only_question_and_selected_context() -> None:
    messages = build_messages(
        "What is the audit phrase?",
        ("Lumen audit phrase: silver pine.", "Routine distractor text."),
    )
    rendered = "\n".join(message["content"] for message in messages)

    assert "What is the audit phrase?" in rendered
    assert "Lumen audit phrase: silver pine." in rendered
    assert "Routine distractor text." in rendered
    assert "ABSTAIN" in rendered
    assert "preferred_route" not in rendered
    assert "expected_answer" not in rendered
    assert "relevant" not in rendered
    assert "qrel" not in rendered.lower()


def test_smollm_prompt_requires_raw_final_answer_without_posthoc_hints() -> None:
    messages = build_messages("List every code.", ("Code A. Code B.",))
    system = messages[0]["content"]
    assert "Return only the final answer" in system
    assert "separate items with semicolons" in system
    assert "using only the supplied context" in system
