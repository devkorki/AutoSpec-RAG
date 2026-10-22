from pathlib import Path

from app.ingestion.pdf_loader import load_pdf_pages
from app.ingestion.chunking import chunk_page

pdf_path = Path("data/pdfs/BMW_X1_2022.pdf")

pages = load_pdf_pages(pdf_path)

all_chunks = []

for page in pages:
    page_chunks = chunk_page(
        text=page["text"],
        source=page["source"],
        page=page["page"],
        chunk_size=700,
        chunk_overlap=100,
    )
    all_chunks.extend(page_chunks)

print(f"Pages: {len(pages)}")
print(f"Chunks: {len(all_chunks)}")
print()

for chunk in all_chunks:
    if chunk.page == 30:
        print("=" * 80)
        print(f"Chunk ID: {chunk.id}")
        print(f"Source: {chunk.source}")
        print(f"Page: {chunk.page}")
        print(f"Characters: {len(chunk.text)}")
        print()
        print(chunk.text)
        print()
