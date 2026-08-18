import argparse
from pathlib import Path

from .chunking import chunk_documents
from .documents import load_directory
from .embeddings import Embedder
from .evaluation import evaluate_retrieval, format_report, load_eval_dataset
from .generation import generate_answer
from .store import VectorStore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Learn RAG from first principles")
    subparsers = parser.add_subparsers(dest="command", required=True)

    index_parser = subparsers.add_parser("index", help="Index documents")
    index_parser.add_argument("directory", type=Path)
    index_parser.add_argument("--chunk-size", type=int, default=900)
    index_parser.add_argument("--overlap", type=int, default=150)

    ask_parser = subparsers.add_parser("ask", help="Retrieve context and ask the LLM")
    ask_parser.add_argument("question")
    ask_parser.add_argument("--top-k", type=int, default=5)
    ask_parser.add_argument("--model", default="llama3.2")

    eval_parser = subparsers.add_parser(
        "eval-retrieval",
        help="Measure retrieval quality against a JSONL evaluation dataset",
    )
    eval_parser.add_argument("dataset", type=Path)
    eval_parser.add_argument("--top-k", type=int, default=5)
    eval_parser.add_argument(
        "--show-misses",
        action="store_true",
        help="Print questions with no relevant passage in the top-k results",
    )

    return parser


def main() -> None:
    args = build_parser().parse_args()
    embedder = Embedder()
    store = VectorStore(embedder)

    if args.command == "index":
        documents = load_directory(args.directory)
        chunks = chunk_documents(
            documents,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
        )
        store.index(chunks)
        print(f"Indexed {len(chunks)} chunks from {len(documents)} documents.")
        return

    if args.command == "eval-retrieval":
        examples = load_eval_dataset(args.dataset)
        report = evaluate_retrieval(store, examples, top_k=args.top_k)
        print(format_report(report, show_misses=args.show_misses))
        return

    results = store.search(args.question, top_k=args.top_k)

    print("\n=== RETRIEVED CONTEXT ===")
    for i, result in enumerate(results, start=1):
        print(
            f"\n[{i}] score={result.score:.4f} "
            f"source={result.source} chunk={result.chunk_index}"
        )
        print(result.text)

    print("\n=== ANSWER ===")
    print(generate_answer(args.question, results, model=args.model))


if __name__ == "__main__":
    main()
