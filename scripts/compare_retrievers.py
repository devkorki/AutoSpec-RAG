from pathlib import Path

from app.ingestion.pdf_loader import load_pdf_pages
from app.ingestion.chunking import chunk_page
from app.retrieval.dense import DenseRetriever
from app.retrieval.lexical import BM25Retriever

pdf_path = Path("data/pdfs/BMW_X1_2022.pdf")

pages = load_pdf_pages(pdf_path)

chunks = []

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


dense = DenseRetriever()
dense.build(chunks)

bm25 = BM25Retriever()
bm25.build(chunks)


query = "What charging speed can I get at a fast charger?"

dense_results = dense.search(query, top_k=5)
bm25_results = bm25.search(query, top_k=5)


print("\n" + "=" * 80)
print("QUERY")
print("=" * 80)
print(query)


print("\n" + "=" * 80)
print("DENSE / MINILM")
print("=" * 80)

for rank, result in enumerate(dense_results, start=1):
    chunk = result["chunk"]

    print(
        f"{rank}. "
        f"score={result['score']:.4f} "
        f"page={chunk.page} "
        f"id={chunk.id}"
    )

    print(chunk.text[:300].replace("\n", " "))
    print()


print("\n" + "=" * 80)
print("BM25")
print("=" * 80)

for rank, result in enumerate(bm25_results, start=1):
    chunk = result["chunk"]

    print(
        f"{rank}. "
        f"score={result['score']:.4f} "
        f"page={chunk.page} "
        f"id={chunk.id}"
    )

    print(chunk.text[:300].replace("\n", " "))
    print()
