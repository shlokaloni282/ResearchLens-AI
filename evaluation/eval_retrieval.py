"""
Retrieval Evaluation Module for ResearchLens AI.

Calculates Hit Rate@1, Hit Rate@3, Hit Rate@5, and MRR (Mean Reciprocal Rank)
across different chunking strategies (fixed, recursive, semantic) using the
benchmark ground-truth dataset.
"""

import argparse
import json
import os
import re
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import fitz

from backend.app.services.chunking import chunk_pages
from backend.app.services.retrieval import InMemoryVectorIndex


STRATEGIES = ("fixed", "recursive", "semantic")


def load_ground_truth(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Ground truth file not found at: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_text_for_matching(text: str) -> str:
    """
    Normalizes whitespace and line-wrap hyphens for robust comparison
    against text extracted from multi-column PDFs without altering ground truth.
    """
    cleaned = re.sub(r"-\s+", "-", text)
    return " ".join(cleaned.split()).lower()


def is_chunk_relevant(chunk: Dict[str, Any], query_item: Dict[str, Any]) -> bool:
    """
    Determines whether a retrieved chunk constitutes a 'Hit' for a query.

    Criteria:
      1. Source paper matches expected paper (if specified)
      2. Page number matches expected_page
      3. Chunk text contains at least one target key_fact
    """
    # Negative / unanswerable questions have no relevant chunks in paper
    if query_item.get("question_type") == "unanswerable" or query_item.get("expected_page") is None:
        return False

    expected_page = query_item.get("expected_page")
    expected_paper = query_item.get("paper_filename")

    # Paper match check (if source_filename recorded)
    if expected_paper and chunk.get("source_filename"):
        if chunk["source_filename"] != expected_paper:
            return False

    # Page number check
    if chunk.get("page_number") != expected_page:
        return False

    # Key fact preservation check
    key_facts = query_item.get("key_facts", [])
    chunk_text_norm = normalize_text_for_matching(chunk.get("text", ""))

    if key_facts:
        has_fact = any(
            normalize_text_for_matching(fact) in chunk_text_norm
            for fact in key_facts
        )
        return has_fact

    # Fallback to checking expected answer tokens if no key_facts specified
    expected_ans_norm = normalize_text_for_matching(query_item.get("expected_answer", ""))
    return any(word in chunk_text_norm for word in expected_ans_norm.split() if len(word) > 4)


def evaluate_retrieval_on_index(
    index: InMemoryVectorIndex,
    ground_truth: List[Dict[str, Any]],
    paper_filename: str,
    top_k_max: int = 5,
) -> Dict[str, Any]:
    """
    Evaluates an index against the ground truth questions.
    Computes Hit Rate@1, @3, @5 and MRR.
    """
    # Evaluate only answerable questions for standard retrieval metrics
    eval_queries = [
        q for q in ground_truth
        if q.get("paper_filename") == paper_filename and q.get("question_type") != "unanswerable"
    ]

    if not eval_queries:
        return {
            "total_queries": 0,
            "hit_rate_at_1": 0.0,
            "hit_rate_at_3": 0.0,
            "hit_rate_at_5": 0.0,
            "mrr": 0.0,
            "query_results": [],
        }

    hits_at_1 = 0
    hits_at_3 = 0
    hits_at_5 = 0
    reciprocal_ranks = []
    query_details = []

    for q in eval_queries:
        retrieved_chunks = index.search(
            query=q["question"],
            top_k=top_k_max,
            source_filename=paper_filename,
        )

        first_hit_rank: Optional[int] = None
        for rank, chunk in enumerate(retrieved_chunks, start=1):
            if is_chunk_relevant(chunk, q):
                first_hit_rank = rank
                break

        # Calculate hits
        if first_hit_rank is not None:
            if first_hit_rank <= 1:
                hits_at_1 += 1
            if first_hit_rank <= 3:
                hits_at_3 += 1
            if first_hit_rank <= 5:
                hits_at_5 += 1
            reciprocal_ranks.append(1.0 / first_hit_rank)
        else:
            reciprocal_ranks.append(0.0)

        query_details.append({
            "id": q["id"],
            "question": q["question"],
            "expected_page": q["expected_page"],
            "first_hit_rank": first_hit_rank,
            "top_retrieved_pages": [c.get("page_number") for c in retrieved_chunks],
            "top_scores": [c.get("score") for c in retrieved_chunks],
        })

    num_queries = len(eval_queries)
    return {
        "total_queries": num_queries,
        "hit_rate_at_1": round((hits_at_1 / num_queries) * 100, 2),
        "hit_rate_at_3": round((hits_at_3 / num_queries) * 100, 2),
        "hit_rate_at_5": round((hits_at_5 / num_queries) * 100, 2),
        "mrr": round(sum(reciprocal_ranks) / num_queries, 4),
        "query_results": query_details,
    }


def run_retrieval_benchmark(
    pdf_path: str,
    ground_truth_path: str = "evaluation/data/ground_truth.json",
    chunk_size: int = 500,
    overlap: int = 50,
    strategies: tuple = STRATEGIES,
    save_report: bool = True,
) -> Dict[str, Any]:
    """
    Runs chunking for each strategy, indexes chunks in InMemoryVectorIndex,
    and computes retrieval evaluation metrics.
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found at: {pdf_path}")

    # Extract pages
    with fitz.open(str(path)) as doc:
        pages = [
            {"page_number": idx, "text": page.get_text("text")}
            for idx, page in enumerate(doc, start=1)
        ]

    gt_data = load_ground_truth(Path(ground_truth_path))
    results = {}

    for strat in strategies:
        chunks = chunk_pages(
            pages=pages,
            paper_id=f"eval-{path.stem}",
            source_filename=path.name,
            strategy=strat,
            chunk_size=chunk_size,
            overlap=overlap,
        )

        # Build fresh vector index for this strategy
        index = InMemoryVectorIndex()
        index.add_chunks(chunks)

        # Evaluate retrieval
        eval_result = evaluate_retrieval_on_index(
            index=index,
            ground_truth=gt_data,
            paper_filename=path.name,
            top_k_max=5,
        )
        eval_result["chunks_indexed"] = len(index)
        results[strat] = eval_result

    report = {
        "timestamp": datetime.now().isoformat(),
        "pdf": path.name,
        "chunk_size": chunk_size,
        "overlap": overlap,
        "results": results,
    }

    if save_report:
        report_dir = Path("evaluation/reports")
        report_dir.mkdir(parents=True, exist_ok=True)
        report_file = report_dir / f"retrieval_eval_{path.stem}.json"
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

    return report


def print_retrieval_scorecard(report: Dict[str, Any]) -> None:
    results = report["results"]
    strategies = list(results.keys())

    print("\n" + "=" * 65)
    print(f"  RETRIEVAL EVALUATION SCORECARD: {report['pdf']}")
    print(f"  Chunk Size: {report['chunk_size']} | Overlap: {report['overlap']}")
    print("=" * 65)

    header = f"{'Metric':<26}" + "".join(f"{s.upper():>12}" for s in strategies)
    print(header)
    print("-" * len(header))

    metrics = [
        ("chunks_indexed", "Indexed Chunks"),
        ("total_queries", "Evaluated Queries"),
        ("hit_rate_at_1", "Hit Rate @ 1"),
        ("hit_rate_at_3", "Hit Rate @ 3"),
        ("hit_rate_at_5", "Hit Rate @ 5"),
        ("mrr", "MRR (Mean Recip Rank)"),
    ]

    for key, label in metrics:
        row = f"{label:<26}"
        for s in strategies:
            val = results[s][key]
            if "hit_rate" in key:
                row += f"{val:>11.1f}%"
            elif key == "mrr":
                row += f"{val:>12.4f}"
            else:
                row += f"{val:>12}"
        print(row)

    print("\n--- Per-Query Retrieval Details (Ranks) ---")
    query_header = f"{'Query ID':<10}{'Expected Page':<16}" + "".join(f"{s.upper():>12}" for s in strategies)
    print(query_header)
    print("-" * len(query_header))

    sample_queries = results[strategies[0]]["query_results"]
    for q_idx, q in enumerate(sample_queries):
        row = f"{q['id']:<10}{'Page ' + str(q['expected_page']):<16}"
        for s in strategies:
            rank = results[s]["query_results"][q_idx]["first_hit_rank"]
            rank_str = f"Rank {rank}" if rank is not None else "MISS"
            row += f"{rank_str:>12}"
        print(row)
    print("=" * 65 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG retrieval across chunking strategies.")
    parser.add_argument(
        "--pdf",
        type=str,
        default="data/benchmark_papers/rng_paper.pdf",
        help="Path to test PDF file",
    )
    parser.add_argument(
        "--gt",
        type=str,
        default="evaluation/data/ground_truth.json",
        help="Path to ground truth JSON file",
    )
    parser.add_argument("--chunk-size", type=int, default=500, help="Chunk size")
    parser.add_argument("--overlap", type=int, default=50, help="Chunk overlap")
    args = parser.parse_args()

    report = run_retrieval_benchmark(
        pdf_path=args.pdf,
        ground_truth_path=args.gt,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
    )
    print_retrieval_scorecard(report)


if __name__ == "__main__":
    main()
