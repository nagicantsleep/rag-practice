import unittest

from rag_practice.evaluation import (
    EvalExample,
    RelevantPassage,
    evaluate_results,
    result_matches_passage,
)
from rag_practice.store import SearchResult


class RetrievalEvaluationTests(unittest.TestCase):
    def test_result_matches_relative_source_and_normalized_text(self) -> None:
        result = SearchResult(
            text="Retrieval-Augmented   Generation combines retrieval with generation.",
            source="/tmp/project/data/raw/rag-notes.md",
            chunk_index=0,
            score=0.9,
        )
        passage = RelevantPassage(
            source="data/raw/rag-notes.md",
            contains="retrieval-augmented generation combines retrieval",
        )

        self.assertTrue(result_matches_passage(result, passage))

    def test_recall_and_mrr(self) -> None:
        example = EvalExample(
            id="q1",
            question="test",
            relevant_passages=(
                RelevantPassage(source="a.md", contains="first fact"),
                RelevantPassage(source="b.md", contains="second fact"),
            ),
        )
        results = [
            SearchResult("noise", "x.md", 0, 0.95),
            SearchResult("the first fact is here", "a.md", 1, 0.90),
            SearchResult("more noise", "y.md", 0, 0.80),
        ]

        report = evaluate_results(example, results)

        self.assertEqual(report.first_relevant_rank, 2)
        self.assertEqual(report.matched_passages, 1)
        self.assertAlmostEqual(report.recall_at_k, 0.5)
        self.assertAlmostEqual(report.reciprocal_rank, 0.5)

    def test_miss_has_zero_reciprocal_rank(self) -> None:
        example = EvalExample(
            id="q2",
            question="test",
            relevant_passages=(
                RelevantPassage(source="a.md", contains="expected evidence"),
            ),
        )
        results = [SearchResult("wrong chunk", "a.md", 0, 0.9)]

        report = evaluate_results(example, results)

        self.assertIsNone(report.first_relevant_rank)
        self.assertEqual(report.recall_at_k, 0.0)
        self.assertEqual(report.reciprocal_rank, 0.0)


if __name__ == "__main__":
    unittest.main()
