from rag_practice.visual_document import (
    BASE_MODEL_NAME,
    BASE_MODEL_REVISION,
    MODEL_NAME,
    MODEL_REVISION,
)


def test_colsmol_control_is_pinned() -> None:
    assert MODEL_NAME == "vidore/colSmol-256M"
    assert MODEL_REVISION == "a59110fdf114638b8018e6c9a018907e12f14855"
    assert BASE_MODEL_NAME == "vidore/ColSmolVLM-Instruct-256M-base"
    assert BASE_MODEL_REVISION == "8a0cee6d479200dbce31dbfef88c66175d89cddc"
