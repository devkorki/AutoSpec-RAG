from pathlib import Path
import pymupdf


def load_pdf_pages(path: Path) -> list[dict]:

    doc = pymupdf.open(path)
    pages: list[dict] = []
    for page_index, page in enumerate(doc):
        text = page.get_text("text").strip()
        if text:
            pages.append(
                {
                    "source": path.name,
                    "page": page_index + 1,
                    "text": text,
                }
            )
    return pages
