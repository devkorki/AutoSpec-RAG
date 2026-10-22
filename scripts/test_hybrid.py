from pathlib import Path

from app.ingestion.pdf_loader import load_pdf_pages
from app.ingestion.chunking import chunk_page
from app.retrieval.hybrid import HybridRetriever

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


retriever = HybridRetriever()

print(f"Pages: {len(pages)}")
print(f"Chunks: {len(chunks)}")
print("Building hybrid index...")

retriever.build(chunks)

print("Ready.")


query = "What charging speed can I get at a fast charger?"

print()
print("=" * 80)
print(f"QUERY: {query}")
print("=" * 80)


results = retriever.search(
    query,
    top_k=5,
    candidate_k=20,
)


for rank, result in enumerate(results, start=1):
    chunk = result["chunk"]

    print()
    print("-" * 80)
    print(f"RANK: {rank}")
    print(f"RRF SCORE: {result['score']:.6f}")
    print(f"PAGE: {chunk.page}")
    print(f"CHUNK: {chunk.id}")
    print()
    print(chunk.text)
