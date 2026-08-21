from rag_practice.visual_document import MODEL_NAME, MODEL_REVISION


def test_colsmol_control_is_pinned() -> None:
    assert MODEL_NAME == "vidore/colSmol-256M"
    assert MODEL_REVISION == "a59110fdf114638b8018e6c9a018907e12f14855"
