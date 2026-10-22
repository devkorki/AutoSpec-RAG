from langchain_core.prompts import (
    ChatPromptTemplate,
)
from langchain_ollama import ChatOllama


class LangChainRAGService:
    def __init__(
        self,
        retriever,
        page_lookup,
    ):
        self.retriever = retriever
        self.page_lookup = page_lookup

        self.llm = ChatOllama(
            model="qwen3:14b",
            temperature=0,
        )

        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
You are AutoSpec.

Answer only using the supplied
vehicle-document context.

If the context does not contain
enough evidence, say:

"I could not verify that from the
provided vehicle documents."

Be concise.
""",
                ),
                (
                    "human",
                    """
Question:
{question}

Context:
{context}
""",
                ),
            ]
        )

        self.chain = self.prompt | self.llm

    def ask(
        self,
        question: str,
        top_k: int = 5,
    ):
        results = self.retriever.search(
            question,
            top_k=top_k,
        )

        pages = []
        citations = []
        seen = set()

        for result in results:
            chunk = result["chunk"]

            key = (
                chunk.source,
                chunk.page,
            )

            if key in seen:
                continue

            seen.add(key)

            page_text = self.page_lookup.get(
                key,
                chunk.text,
            )

            pages.append(f"""
Source: {chunk.source}
Page: {chunk.page}

{page_text}
""")

            citations.append(
                {
                    "source": chunk.source,
                    "page": chunk.page,
                }
            )

        context = "\n\n".join(pages)

        response = self.chain.invoke(
            {
                "question": question,
                "context": context,
            }
        )

        return {
            "answer": response.content,
            "citations": citations,
        }
