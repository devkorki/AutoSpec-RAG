# AutoSpec-RAG Interview Rebuild

A deliberately small automotive RAG project built to understand and demonstrate the concepts relevant to a Junior AI Engineer role: PDF ingestion, chunking, embeddings, hybrid retrieval, reranking, evaluation, grounded generation, REST APIs, JSON, and production-minded debugging.

## Philosophy

This repo intentionally does **not** try to use every AI framework. The core retrieval path is implemented in ordinary Python so each stage can be explained. Optional integrations (MCP, LangChain/LlamaIndex, React) can be added later as separate exercises.

## Target pipeline

PDF -> page extraction -> chunks + metadata -> dense embeddings + BM25 -> RRF fusion -> optional cross-encoder reranking -> evidence check -> LLM answer with citations -> FastAPI JSON response

## Build stages

1. Ingestion and metadata
2. Baseline dense retrieval
3. BM25 + RRF hybrid retrieval
4. Retrieval evaluation
5. Cross-encoder reranking
6. Grounded generation + abstention
7. FastAPI service
8. Tests, logging and failure handling
9. Optional React demo / MCP adapter

Put local vehicle brochures in `data/pdfs/`.
