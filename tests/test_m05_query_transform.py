from rag_practice.query_transform import GenerativeQueryTransformer


class FakeGenerator:
    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.prompts = []

    def generate(self, prompt: str, *, max_new_tokens: int = 64) -> str:
        self.prompts.append((prompt, max_new_tokens))
        return next(self.outputs)


def test_rewrite_falls_back_to_original_on_empty_generation():
    transformer = GenerativeQueryTransformer(FakeGenerator(["   "]))
    assert transformer.rewrite("semantic matching") == "semantic matching"


def test_multi_query_keeps_original_deduplicates_and_parses_numbering():
    generator = FakeGenerator(["1. semantic vector search\n2) embedding retrieval\n- semantic vector search"])
    transformer = GenerativeQueryTransformer(generator)
    assert transformer.multi_query("meaning search", count=3) == [
        "meaning search",
        "semantic vector search",
        "embedding retrieval",
    ]


def test_query2doc_preserves_original_query():
    transformer = GenerativeQueryTransformer(FakeGenerator(["Embeddings match semantically similar passages."]))
    expanded = transformer.query2doc("conceptual likeness")
    assert expanded.startswith("conceptual likeness ")
    assert "Embeddings" in expanded


def test_hyde_falls_back_to_query_when_empty():
    transformer = GenerativeQueryTransformer(FakeGenerator([""]))
    assert transformer.hyde_document("what is RAG") == "what is RAG"


def test_decomposition_parses_subquestions_without_reference_data():
    transformer = GenerativeQueryTransformer(
        FakeGenerator(["1. What is BM25?\n2. What is reciprocal rank fusion?"])
    )
    assert transformer.decompose("compare both", max_parts=3) == [
        "What is BM25?",
        "What is reciprocal rank fusion?",
    ]
