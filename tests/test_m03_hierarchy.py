from rag_practice.core.models import Document
from rag_practice.indexing.hierarchy import HierarchicalBM25Index, ParentChildBM25Index


def test_parent_child_returns_wider_parent_context():
    document = Document(
        "d1",
        "Alpha evidence starts here. Beta evidence completes the answer.\n\nUnrelated maintenance note.",
        metadata={"title": "Evidence Guide"},
    )
    index = ParentChildBM25Index([document])
    parent_id, _ = index.search("alpha evidence", k=1)[0]
    parent = index.parent_by_id[parent_id]
    assert "Alpha evidence starts here." in parent.text
    assert "Beta evidence completes the answer." in parent.text


def test_hierarchical_route_uses_metadata_without_polluting_leaf_text():
    arctic = Document(
        "d1",
        "Run the retrieval health probe before accepting requests.",
        metadata={"title": "Arctic Deployment Handbook", "region": "arctic"},
    )
    tropical = Document(
        "d2",
        "Check network connectivity before accepting requests.",
        metadata={"title": "Tropical Deployment Handbook", "region": "tropical"},
    )
    index = HierarchicalBM25Index([arctic, tropical])
    assert index.route("arctic deployment", k=1)[0][0] == "d1"
    result_id, _ = index.search("arctic deployment before accepting requests", k=1)[0]
    assert index.leaf_by_id[result_id].document_id == "d1"
    assert "Arctic" not in index.leaf_by_id[result_id].text
