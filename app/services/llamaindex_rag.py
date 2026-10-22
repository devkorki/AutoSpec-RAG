import chromadb

from llama_index.core import (
    Document,
    Settings,
    StorageContext,
    VectorStoreIndex,
)

from llama_index.embeddings.huggingface import (
    HuggingFaceEmbedding,
)

from llama_index.llms.ollama import (
    Ollama,
)

from llama_index.vector_stores.chroma import (
    ChromaVectorStore,
)


class LlamaIndexRAGService:
    def __init__(
        self,
        chunks,
    ):
        Settings.embed_model = HuggingFaceEmbedding(
            model_name=("sentence-transformers/" "all-MiniLM-L6-v2")
        )

        Settings.llm = Ollama(
            model="qwen3:14b",
            request_timeout=120.0,
        )

        client = chromadb.PersistentClient(path="data/" "llamaindex_chroma")

        collection = client.get_or_create_collection("autospec_llamaindex")

        vector_store = ChromaVectorStore(chroma_collection=collection)

        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        documents = [
            Document(
                text=chunk.text,
                metadata={
                    "source": (chunk.source),
                    "page": (chunk.page),
                },
            )
            for chunk in chunks
        ]

        self.index = VectorStoreIndex.from_documents(
            documents,
            storage_context=(storage_context),
        )

        self.query_engine = self.index.as_query_engine(
            similarity_top_k=5,
        )

    def ask(
        self,
        question: str,
        top_k: int = 5,
    ):
        response = self.query_engine.query(question)

        citations = []
        seen = set()

        for node in response.source_nodes or []:
            metadata = node.node.metadata

            key = (
                metadata.get("source"),
                metadata.get("page"),
            )

            if key in seen:
                continue

            seen.add(key)

            citations.append(
                {
                    "source": key[0],
                    "page": key[1],
                }
            )

        return {
            "answer": str(response),
            "citations": citations,
        }
