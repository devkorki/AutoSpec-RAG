from pathlib import Path

from app.ingestion.pdf_loader import load_pdf_pages
from app.ingestion.chunking import chunk_page
from app.retrieval.dense import DenseRetriever

pdf_path = Path("data/pdfs/BMW_X1_2022.pdf")

pages = load_pdf_pages(pdf_path)

chunks = []

for page in pages:
    page_chunks = chunk_page(
        text=page["text"],
        source=page["source"],
        page=page["page"],
        chunk_size=700,
        chunk_overlap=100,
    )

    chunks.extend(page_chunks)


print(f"Pages: {len(pages)}")
print(f"Chunks: {len(chunks)}")


retriever = DenseRetriever()

print("Building FAISS index...")

retriever.build(chunks)

print("Index ready.")
print()


query = "What charging speed can I get at a fast charger?"
print(f"QUERY: {query}")
print()


results = retriever.search(
    query,
    top_k=5,
)


for rank, result in enumerate(results, start=1):

    chunk = result["chunk"]
    score = result["score"]

    print("=" * 80)
    print(f"RANK: {rank}")
    print(f"SCORE: {score:.4f}")
    print(f"SOURCE: {chunk.source}")
    print(f"PAGE: {chunk.page}")
    print(f"CHUNK: {chunk.id}")
    print()
    print(chunk.text)
    print()
