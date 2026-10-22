import json

from app.generation.ollama import OllamaGenerator
from app.generation.prompt import (
    SYSTEM_PROMPT,
    build_user_prompt,
)


class RAGService:
    def __init__(
        self,
        retriever,
        page_lookup: dict[tuple[str, int], str] | None = None,
        generator: OllamaGenerator | None = None,
    ):
        self.retriever = retriever
        self.page_lookup = page_lookup or {}
        self.generator = generator or OllamaGenerator()

    def ask(
        self,
        question: str,
        top_k: int = 5,
    ):
        results = self.retriever.search(
            question,
            top_k=top_k,
            candidate_k=20,
        )

        if not results:
            return {
                "answer": (
                    "I could not verify that from " "the provided vehicle documents."
                ),
                "citations": [],
            }

        # Expand retrieved chunks to their full source pages.
        context_documents = []
        seen_pages = set()

        for result in results:
            chunk = result["chunk"]

            key = (
                chunk.source,
                chunk.page,
            )

            if key in seen_pages:
                continue

            seen_pages.add(key)

            context_documents.append(
                {
                    "source": chunk.source,
                    "page": chunk.page,
                    "text": self.page_lookup.get(
                        key,
                        chunk.text,
                    ),
                }
            )

        user_prompt = build_user_prompt(
            question,
            context_documents,
        )

        print("\n--- RETRIEVED CONTEXT ---")
        print(user_prompt)

        raw_response = self.generator.generate(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )

        print("\n--- RAW LLM RESPONSE ---")
        print(raw_response)

        data = json.loads(raw_response)

        answer = data["answer"]

        cited_document_ids = data.get(
            "cited_document_ids",
            [],
        )

        citations = []
        seen_citations = set()

        for document_id in cited_document_ids:
            index = document_id - 1

            if index < 0 or index >= len(context_documents):
                continue

            document = context_documents[index]

            key = (
                document["source"],
                document["page"],
            )

            if key in seen_citations:
                continue

            citations.append(
                {
                    "source": document["source"],
                    "page": document["page"],
                }
            )

            seen_citations.add(key)

        return {
            "answer": answer,
            "citations": citations,
        }
