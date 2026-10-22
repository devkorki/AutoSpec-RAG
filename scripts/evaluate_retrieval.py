import json
from pathlib import Path

from app.ingestion.pdf_loader import load_pdf_pages
from app.ingestion.chunking import chunk_page
from app.retrieval.hybrid import HybridRetriever

PDF_DIR = Path("data/pdfs")
EVAL_FILE = Path("data/eval/questions.jsonl")


def load_chunks():
    chunks = []

    for pdf_path in PDF_DIR.glob("*.pdf"):
        pages = load_pdf_pages(pdf_path)

        for page in pages:
            chunks.extend(
                chunk_page(
                    text=page["text"],
                    source=page["source"],
                    page=page["page"],
                    chunk_size=700,
                    chunk_overlap=100,
                )
            )

    return chunks


def load_questions():
    questions = []

    with EVAL_FILE.open(
        "r",
        encoding="utf-8",
    ) as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))

    return questions


def is_relevant(chunk, example):
    return (
        chunk.source == example["relevant_source"]
        and chunk.page in example["relevant_pages"]
    )


def main():
    chunks = load_chunks()
    questions = load_questions()

    print(f"Chunks: {len(chunks)}")
    print(f"Evaluation questions: {len(questions)}")
    print()

    retriever = HybridRetriever()
    retriever.build(chunks)

    hits_at_1 = 0
    hits_at_3 = 0
    hits_at_5 = 0

    reciprocal_ranks = []

    for example in questions:

        results = retriever.search(
            example["question"],
            top_k=10,
            candidate_k=20,
        )

        first_relevant_rank = None

        for rank, result in enumerate(results, start=1):
            if is_relevant(result["chunk"], example):
                first_relevant_rank = rank
                break

        if first_relevant_rank is not None:
            if first_relevant_rank <= 1:
                hits_at_1 += 1

            if first_relevant_rank <= 3:
                hits_at_3 += 1

            if first_relevant_rank <= 5:
                hits_at_5 += 1

            reciprocal_ranks.append(1 / first_relevant_rank)

        else:
            reciprocal_ranks.append(0.0)

        print("=" * 80)
        print(example["question"])
        print(f"Relevant pages: {example['relevant_pages']}")
        print(f"First relevant rank: {first_relevant_rank}")

        if results:
            print(
                "Top pages:",
                [result["chunk"].page for result in results[:5]],
            )

    n = len(questions)

    hit_1 = hits_at_1 / n
    hit_3 = hits_at_3 / n
    hit_5 = hits_at_5 / n

    mrr = sum(reciprocal_ranks) / n

    print()
    print("=" * 80)
    print("EVALUATION")
    print("=" * 80)

    print(f"Hit@1: {hit_1:.3f}")
    print(f"Hit@3: {hit_3:.3f}")
    print(f"Hit@5: {hit_5:.3f}")
    print(f"MRR:   {mrr:.3f}")


if __name__ == "__main__":
    main()
