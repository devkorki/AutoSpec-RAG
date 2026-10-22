from app.core.models import Chunk


def _split_text(
    text: str,
    chunk_size: int,
    separators: list[str],
) -> list[str]:
    """
    Recursively split text using increasingly smaller separators.
    """

    text = text.strip()

    if not text:
        return []

    # Base case: already small enough.
    if len(text) <= chunk_size:
        return [text]

    # No separators left: hard character split.
    if not separators:
        return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

    separator = separators[0]

    parts = text.split(separator)

    # This separator didn't help.
    if len(parts) == 1:
        return _split_text(
            text,
            chunk_size,
            separators[1:],
        )

    chunks = []
    current = ""

    for part in parts:
        part = part.strip()

        if not part:
            continue

        candidate = f"{current}{separator}{part}" if current else part

        if len(candidate) <= chunk_size:
            current = candidate

        else:
            if current:
                chunks.append(current)

            # The individual part may itself still be oversized.
            if len(part) > chunk_size:
                chunks.extend(
                    _split_text(
                        part,
                        chunk_size,
                        separators[1:],
                    )
                )
                current = ""
            else:
                current = part

    if current:
        chunks.append(current)

    return chunks


def _get_overlap(text: str, overlap_size: int) -> str:
    if overlap_size <= 0:
        return ""

    if len(text) <= overlap_size:
        return text

    tail = text[-overlap_size:]

    # Drop the first partial word.
    first_space = tail.find(" ")

    if first_space != -1:
        tail = tail[first_space + 1 :]

    return tail.strip()


def chunk_page(
    *,
    text: str,
    source: str,
    page: int,
    chunk_size: int = 700,
    chunk_overlap: int = 100,
) -> list[Chunk]:

    base_chunks = _split_text(
        text,
        chunk_size,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
        ],
    )

    chunks: list[Chunk] = []

    for idx, text_chunk in enumerate(base_chunks):

        # Add some context from the previous chunk.
        if idx > 0 and chunk_overlap > 0:
            previous_tail = _get_overlap(
                base_chunks[idx - 1],
                chunk_overlap,
            )

            text_chunk = (previous_tail + "\n" + text_chunk).strip()

        chunks.append(
            Chunk(
                id=f"{source}:p{page}:c{idx}",
                text=text_chunk,
                source=source,
                page=page,
            )
        )

    return chunks
