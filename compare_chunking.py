import argparse
import re
import statistics
from collections import Counter
from pathlib import Path

import fitz

from backend.app.services.chunking import chunk_pages

STRATEGIES = ("fixed", "recursive", "semantic")

# Facts that should survive intact inside a single chunk (RNG paper)
CHECKS = {
    "power":      "23.4 mW",
    "clock":      "100 MHz clock rate",
    "chip area":  "116x84",
    "monobit":    "9,654 < X < 10,346",
    "long run":   "run of length 34",
    "email":      "ccwang@ee.nsysu.edu.tw",
    "table row":  "clock (MHz) 33 1 0.5 0.001 100",
}


def normalize(text: str) -> str:
    return " ".join(text.split())


def analyse(chunks: list[dict]) -> dict:
    texts = [normalize(c["text"]) for c in chunks]
    lengths = [len(t) for t in texts]
    return {
        "chunks": len(chunks),
        "min": min(lengths),
        "median": statistics.median(lengths),
        "max": max(lengths),
        "under50": sum(l < 50 for l in lengths),
        "starts_lower": sum(t[:1].islower() for t in texts),
        "ends_label": sum(bool(re.search(r"\b(?:Fig|Eq|Ref)\.$", t)) for t in texts),
        "per_page": dict(sorted(Counter(c["page_number"] for c in chunks).items())),
        "facts": {name: any(fact in t for t in texts) for name, fact in CHECKS.items()},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf_path", help="Path to the PDF file")
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--overlap", type=int, default=50)
    parser.add_argument("--show", type=int, default=0,
                        help="Print the first N chunks of each strategy")
    args = parser.parse_args()

    # Same extraction as the upload endpoint (PyMuPDF plain text per page)
    with fitz.open(args.pdf_path) as document:
        pages = [
            {"page_number": number, "text": page.get_text("text")}
            for number, page in enumerate(document, start=1)
        ]

    results = {}
    for strategy in STRATEGIES:
        chunks = chunk_pages(
            pages=pages,
            paper_id="comparison-paper",
            source_filename=Path(args.pdf_path).name,
            strategy=strategy,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
        )
        results[strategy] = analyse(chunks)

        print(f"\n--- {strategy.upper()} ---")
        print(f"Total chunks: {len(chunks)}")
        print(f"Chunks per page: {results[strategy]['per_page']}")
        for chunk in chunks[: args.show]:
            preview = normalize(chunk["text"])
            print(f"\nPage {chunk['page_number']}: {preview[:180]}...")

    # Side-by-side summary
    print("\n=== SUMMARY ===")
    header = f"{'metric':<22}" + "".join(f"{s:>12}" for s in STRATEGIES)
    print(header)
    print("-" * len(header))
    for key, label in [
        ("chunks", "chunks"),
        ("min", "min length"),
        ("median", "median length"),
        ("max", "max length"),
        ("under50", "chunks < 50 chars"),
        ("starts_lower", "start lowercase"),
        ("ends_label", "end on Fig./Eq."),
    ]:
        print(f"{label:<22}" + "".join(f"{results[s][key]:>12}" for s in STRATEGIES))

    print("\nFact intact in one chunk:")
    for name in CHECKS:
        row = "".join(f"{'yes' if results[s]['facts'][name] else 'NO':>12}" for s in STRATEGIES)
        print(f"{name:<22}{row}")


if __name__ == "__main__":
    main()