from pathlib import Path

from app.ingestion.pdf_loader import load_pdf_pages
from app.ingestion.chunking import chunk_page
from app.retrieval.hybrid import HybridRetriever
from app.services.rag import RAGService

PDF_DIR = Path("data/pdfs")


def load_data():
    chunks = []
    page_lookup = {}

    for pdf_path in PDF_DIR.glob("*.pdf"):
        pages = load_pdf_pages(pdf_path)

        for page in pages:
            key = (
                page["source"],
                page["page"],
            )

            page_lookup[key] = page["text"]

            chunks.extend(
                chunk_page(
                    text=page["text"],
                    source=page["source"],
                    page=page["page"],
                    chunk_size=700,
                    chunk_overlap=100,
                )
            )

    return chunks, page_lookup


def main():
    chunks, page_lookup = load_data()
    retriever = HybridRetriever(
        use_reranker=False,
    )

    retriever.build(chunks)

    rag = RAGService(
        retriever=retriever,
        page_lookup=page_lookup,
    )

    question = "What is the maximum DC charging power " "of the BMW iX1?"
    # question = "What is the battery warranty period of the BMW iX1?"

    response = rag.ask(
        question,
        top_k=5,
    )

    print()
    print("QUESTION")
    print("=" * 80)
    print(question)

    print()
    print("ANSWER")
    print("=" * 80)
    print(response["answer"])

    print()
    print("CITATIONS")
    print("=" * 80)

    for citation in response["citations"]:
        print(f"{citation['source']} - " f"page {citation['page']}")


if __name__ == "__main__":
    main()
