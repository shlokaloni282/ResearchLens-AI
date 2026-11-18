"""
Chunking Evaluation Module for ResearchLens AI.

Evaluates and benchmarks chunking strategies (fixed, recursive, semantic) on
academic PDFs using metrics:
- Fact retention integrity (from ground_truth.json)
- Chunk size distribution (min, median, max, std dev)
- Fragment count (<50 chars)
- Boundary coherence (mid-sentence splits, dangling abbreviations)
- Header/footer cleanup efficiency
"""

import argparse
import json
import os
import re
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import fitz

from backend.app.services.chunking import chunk_pages


STRATEGIES = ("fixed", "recursive", "semantic")
ABBREV_END_RE = re.compile(r"\b(?:Fig|Figs|Eq|Eqs|Ref|Refs|vs)\.$", re.I)
ROMAN_END_RE = re.compile(r"(?<!Table )(?<!Section )\b[IVX]{2,}\.$", re.I)
FOOTER_JUNK_PATTERNS = ["Authorized licensed use", "IEEE Xplore", "Restrictions apply"]


def normalize_whitespace(text: str) -> str:
    return " ".join(text.split())


def load_ground_truth(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate_chunks(
    chunks: List[Dict[str, Any]],
    ground_truth: List[Dict[str, Any]],
    paper_filename: str,
) -> Dict[str, Any]:
    """
    Computes statistical and factual evaluation metrics on a set of generated chunks.
    """
    if not chunks:
        return {
            "total_chunks": 0,
            "min_len": 0,
            "median_len": 0,
            "max_len": 0,
            "fragments_under_50": 0,
            "starts_lower": 0,
            "ends_abbrev": 0,
            "ends_roman": 0,
            "footer_junk": 0,
            "fact_retention_rate": 0.0,
            "facts_passed": 0,
            "facts_total": 0,
            "facts_detail": {},
        }

    texts = [normalize_whitespace(c["text"]) for c in chunks]
    lengths = [len(t) for t in texts]

    # Boundary & text hygiene checks
    starts_lower = sum(bool(t and t[0].islower()) for t in texts)
    ends_abbrev = sum(bool(ABBREV_END_RE.search(t)) for t in texts)
    ends_roman = sum(bool(ROMAN_END_RE.search(t)) for t in texts)
    fragments_under_50 = sum(length < 50 for length in lengths)
    footer_junk = sum(
        any(pat in t for pat in FOOTER_JUNK_PATTERNS)
        for t in texts
    )

    # Fact retention checks from ground truth
    relevant_gt = [
        gt for gt in ground_truth
        if gt.get("paper_filename") == paper_filename and gt.get("key_facts")
    ]

    facts_detail = {}
    facts_passed = 0
    facts_total = 0

    for item in relevant_gt:
        expected_page = item.get("expected_page")
        for fact in item.get("key_facts", []):
            facts_total += 1
            # Check if this fact survives intact in at least one chunk (optionally matching page)
            found_intact = any(
                fact in c["text"]
                for c in chunks
                if expected_page is None or c.get("page_number") == expected_page
            )
            # Fallback check across all pages if not found on expected page
            if not found_intact:
                found_intact = any(fact in c["text"] for c in chunks)

            facts_detail[fact] = found_intact
            if found_intact:
                facts_passed += 1

    retention_rate = (facts_passed / facts_total * 100) if facts_total > 0 else 100.0

    return {
        "total_chunks": len(chunks),
        "min_len": min(lengths),
        "median_len": statistics.median(lengths),
        "max_len": max(lengths),
        "fragments_under_50": fragments_under_50,
        "starts_lower": starts_lower,
        "ends_abbrev": ends_abbrev,
        "ends_roman": ends_roman,
        "footer_junk": footer_junk,
        "fact_retention_rate": round(retention_rate, 2),
        "facts_passed": facts_passed,
        "facts_total": facts_total,
        "facts_detail": facts_detail,
    }


def run_chunking_evaluation(
    pdf_path: str,
    ground_truth_path: str = "evaluation/data/ground_truth.json",
    chunk_size: int = 500,
    overlap: int = 50,
    strategies: tuple = STRATEGIES,
    save_report: bool = True,
) -> Dict[str, Any]:
    """
    Runs extraction and evaluates each strategy on the given PDF.
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found at: {pdf_path}")

    # Extract pages with PyMuPDF
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
        results[strat] = evaluate_chunks(chunks, gt_data, path.name)

    report = {
        "timestamp": datetime.now().isoformat(),
        "pdf": path.name,
        "total_pages": len(pages),
        "chunk_size": chunk_size,
        "overlap": overlap,
        "results": results,
    }

    if save_report:
        report_dir = Path("evaluation/reports")
        report_dir.mkdir(parents=True, exist_ok=True)
        report_file = report_dir / f"chunking_eval_{path.stem}.json"
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

    return report


def print_scorecard(report: Dict[str, Any]) -> None:
    results = report["results"]
    strategies = list(results.keys())

    print("\n" + "=" * 65)
    print(f"  CHUNKING EVALUATION SCORECARD: {report['pdf']}")
    print(f"  Pages: {report['total_pages']} | Target Chunk Size: {report['chunk_size']} | Overlap: {report['overlap']}")
    print("=" * 65)

    header = f"{'Metric':<26}" + "".join(f"{s.upper():>12}" for s in strategies)
    print(header)
    print("-" * len(header))

    metrics = [
        ("total_chunks", "Total Chunks"),
        ("min_len", "Min Length (chars)"),
        ("median_len", "Median Length (chars)"),
        ("max_len", "Max Length (chars)"),
        ("fragments_under_50", "Fragments (<50 chars)"),
        ("starts_lower", "Starts Lowercase"),
        ("ends_abbrev", "Dangling Abbreviation"),
        ("ends_roman", "Split Roman Numeral"),
        ("footer_junk", "Footer Junk Chunks"),
        ("fact_retention_rate", "Fact Retention %"),
    ]

    for key, label in metrics:
        row = f"{label:<26}"
        for s in strategies:
            val = results[s][key]
            if key == "fact_retention_rate":
                row += f"{val:>11.1f}%"
            else:
                row += f"{val:>12}"
        print(row)

    print("\n--- Key Facts Retention Breakdown ---")
    first_strat = strategies[0]
    all_facts = list(results[first_strat]["facts_detail"].keys())
    if all_facts:
        fact_header = f"{'Fact':<32}" + "".join(f"{s.upper():>10}" for s in strategies)
        print(fact_header)
        print("-" * len(fact_header))
        for fact in all_facts:
            row = f"{fact[:30]:<32}"
            for s in strategies:
                survived = results[s]["facts_detail"].get(fact, False)
                status = "PASS" if survived else "FAIL"
                row += f"{status:>10}"
            print(row)
    print("=" * 65 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate chunking strategies on a scientific paper.")
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
    parser.add_argument("--chunk-size", type=int, default=500, help="Target chunk size")
    parser.add_argument("--overlap", type=int, default=50, help="Chunk overlap")
    args = parser.parse_args()

    report = run_chunking_evaluation(
        pdf_path=args.pdf,
        ground_truth_path=args.gt,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
    )
    print_scorecard(report)


if __name__ == "__main__":
    main()
