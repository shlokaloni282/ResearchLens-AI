import argparse
from collections import Counter

import fitz

from backend.app.services.chunking import chunk_pages


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf_path", help="Path to the PDF file")
    args = parser.parse_args()

    # Extract the same page text that the upload endpoint returns.
    with fitz.open(args.pdf_path) as document:
        pages = [
            {"page_number": number, "text": page.get_text("text")}
            for number, page in enumerate(document, start=1)
        ]

    for strategy in ("fixed", "recursive", "semantic"):
        chunks = chunk_pages(
            pages=pages,
            paper_id="comparison-paper",
            source_filename=args.pdf_path.split("\\")[-1],
            strategy=strategy,
            chunk_size=500,
            overlap=50,
        )

        counts_by_page = Counter(chunk["page_number"] for chunk in chunks)

        print(f"\n--- {strategy.upper()} ---")
        print(f"Total chunks: {len(chunks)}")
        print(f"Chunks per page: {dict(sorted(counts_by_page.items()))}")

        for chunk in chunks[:2]:
            preview = chunk["text"].replace("\n", " ")
            print(f"\nPage {chunk['page_number']}: {preview[:180]}...")


if __name__ == "__main__":
    main()