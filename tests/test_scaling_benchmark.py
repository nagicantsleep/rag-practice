from rag_practice.evaluation.scaling import expand_corpus, generate_distractors


def test_generate_distractors_is_deterministic_and_unique() -> None:
    first = generate_distractors(25)
    second = generate_distractors(25)
    assert first == second
    assert len(first) == 25
    assert len(set(first)) == 25
    assert first["scale-00000"] != first["scale-00001"]


def test_expand_corpus_preserves_base_documents() -> None:
    base = {"d1": "target evidence", "d2": "other evidence"}
    expanded = expand_corpus(base, 12)
    assert len(expanded) == 12
    assert expanded["d1"] == "target evidence"
    assert expanded["d2"] == "other evidence"


def test_expand_corpus_rejects_smaller_target() -> None:
    try:
        expand_corpus({"d1": "text"}, 0)
    except ValueError as exc:
        assert "smaller" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("smaller target must fail")
