from pathlib import Path

from app.ingestion.pdf_loader import load_pdf_pages

pdf_path = Path("data/pdfs/BMW_X1_2022.pdf")

pages = load_pdf_pages(pdf_path)

print(f"Extracted pages: {len(pages)}")

# for page in pages[:3]:
#     print("=" * 80)
#     print(f"Source: {page['source']}")
#     print(f"Page: {page['page']}")
#     print()
#     print(page["text"][:1500])


for page in pages:
    if "130 kW" in page["text"]:
        print("=" * 80)
        print(f"Source: {page['source']}")
        print(f"Page: {page['page']}")
        print()
        print(page["text"])
