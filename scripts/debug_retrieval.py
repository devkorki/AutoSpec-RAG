from pathlib import Path

from app.ingestion.pdf_loader import load_pdf_pages
from app.ingestion.chunking import chunk_page
from app.retrieval.dense import DenseRetriever
from app.retrieval.lexical import BM25Retriever
from app.retrieval.rrf import reciprocal_rank_fusion

PDF_DIR = Path("data/pdfs")

query = "What is the maximum DC charging power of the BMW iX1?"


# --------------------------------------------------
# Load and chunk
# --------------------------------------------------

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


chunks_by_id = {chunk.id: chunk for chunk in chunks}


# --------------------------------------------------
# Build retrievers
# --------------------------------------------------

dense = DenseRetriever()
dense.build(chunks)

bm25 = BM25Retriever()
bm25.build(chunks)


# --------------------------------------------------
# Retrieve
# --------------------------------------------------

dense_results = dense.search(query, top_k=20)
bm25_results = bm25.search(query, top_k=20)


dense_ids = [result["chunk"].id for result in dense_results]

bm25_ids = [result["chunk"].id for result in bm25_results]


fused = reciprocal_rank_fusion([dense_ids, bm25_ids])


# --------------------------------------------------
# Dense
# --------------------------------------------------

print("\n" + "=" * 80)
print("DENSE")
print("=" * 80)

for rank, result in enumerate(dense_results, start=1):

    chunk = result["chunk"]

    print(
        f"{rank:2}. "
        f"score={result['score']:.4f} "
        f"page={chunk.page:<2} "
        f"id={chunk.id}"
    )


# --------------------------------------------------
# BM25
# --------------------------------------------------

print("\n" + "=" * 80)
print("BM25")
print("=" * 80)

for rank, result in enumerate(bm25_results, start=1):

    chunk = result["chunk"]

    print(
        f"{rank:2}. "
        f"score={result['score']:.4f} "
        f"page={chunk.page:<2} "
        f"id={chunk.id}"
    )


# --------------------------------------------------
# RRF
# --------------------------------------------------

print("\n" + "=" * 80)
print("RRF")
print("=" * 80)

for rank, (chunk_id, score) in enumerate(fused[:20], start=1):

    chunk = chunks_by_id[chunk_id]

    dense_rank = dense_ids.index(chunk_id) + 1 if chunk_id in dense_ids else None

    bm25_rank = bm25_ids.index(chunk_id) + 1 if chunk_id in bm25_ids else None

    print(
        f"{rank:2}. "
        f"rrf={score:.6f} "
        f"page={chunk.page:<2} "
        f"dense={str(dense_rank):<4} "
        f"bm25={str(bm25_rank):<4} "
        f"id={chunk.id}"
    )
