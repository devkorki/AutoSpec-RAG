class RetrieverRouter:
    def __init__(
        self,
        native_retriever,
        chroma_retriever,
    ):
        self.native = native_retriever
        self.chroma = chroma_retriever

    def get(
        self,
        vector_store: str,
    ):
        if vector_store == "chroma":
            return self.chroma

        return self.native
