"""
PulseAgent — RAG Agent
Ingests product documentation into ChromaDB and answers queries:
  - "Is this issue already resolved?"
  - "What does the docs say about X?"

Uses Gemini for embeddings-based Q&A (large context window helps).
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from textwrap import dedent
from typing import Optional

import chromadb
try:
    from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
except ImportError:
    from chromadb.utils import embedding_functions
    DefaultEmbeddingFunction = embedding_functions.DefaultEmbeddingFunction
from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger

from config import get_settings
from llm_factory import get_llm
from models import RAGContext


class DocumentIngester:
    """Loads text files / markdown docs into ChromaDB."""

    CHUNK_SIZE = 800
    CHUNK_OVERLAP = 100

    def __init__(self, collection: chromadb.Collection):
        self.collection = collection

    def _chunk_text(self, text: str) -> list[str]:
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.CHUNK_SIZE
            chunks.append(text[start:end])
            start = end - self.CHUNK_OVERLAP
        return [c.strip() for c in chunks if c.strip()]

    def ingest_file(self, filepath: str | Path) -> int:
        """Ingest a single file. Returns number of chunks added."""
        path = Path(filepath)
        if not path.exists():
            logger.warning(f"[RAG] File not found: {path}")
            return 0

        text = path.read_text(encoding="utf-8", errors="ignore")
        chunks = self._chunk_text(text)

        ids, docs, metas = [], [], []
        for i, chunk in enumerate(chunks):
            chunk_id = hashlib.md5(f"{path.name}:{i}:{chunk[:50]}".encode()).hexdigest()
            ids.append(chunk_id)
            docs.append(chunk)
            metas.append({"source": path.name, "chunk_index": i})

        if ids:
            # Upsert — safe to call multiple times
            self.collection.upsert(ids=ids, documents=docs, metadatas=metas)
            logger.info(f"[RAG] Ingested '{path.name}': {len(ids)} chunks.")

        return len(ids)

    def ingest_directory(self, directory: str | Path, extensions: tuple = (".md", ".txt", ".pdf")) -> int:
        total = 0
        for ext in extensions:
            for f in Path(directory).rglob(f"*{ext}"):
                total += self.ingest_file(f)
        logger.success(f"[RAG] Directory ingestion complete: {total} total chunks.")
        return total


class RAGAgent:
    """
    LangGraph node: given a review cluster, queries ChromaDB for relevant
    product docs and determines if the issue is already addressed.
    Used by ResponseAgent to craft accurate replies.
    """

    QUERY_SYSTEM = dedent("""
    You are a product support specialist with access to internal documentation.
    Given a user complaint and the retrieved documentation excerpts,
    answer these two questions:
    1. Is this issue already resolved/addressed in the documentation?
    2. What relevant information from the docs should be included in a response?

    Return ONLY valid JSON:
    {
      "already_resolved": <bool>,
      "resolution_reference": "<changelog entry or doc section if resolved, else null>",
      "relevant_info": "<key info from docs to use in response, max 3 sentences>"
    }
    """).strip()

    def __init__(self, docs_dir: Optional[str] = None):
        self.settings = get_settings()
        self.llm = get_llm("rag")

        # Ensure chroma persist directory exists
        import os
        os.makedirs(self.settings.chroma_persist_dir, exist_ok=True)

        self.client = chromadb.PersistentClient(path=self.settings.chroma_persist_dir)

        # Use default embedding function (uses local sentence-transformers model)
        ef = DefaultEmbeddingFunction()
        self.collection = self.client.get_or_create_collection(
            name=self.settings.chroma_collection_name,
            embedding_function=ef,
        )

        self.ingester = DocumentIngester(self.collection)

        if docs_dir:
            self.ingest_docs(docs_dir)

    def ingest_docs(self, docs_dir: str) -> int:
        return self.ingester.ingest_directory(docs_dir)

    def ingest_text(self, text: str, source_name: str = "manual_input") -> int:
        chunks = self.ingester._chunk_text(text)
        ids, docs, metas = [], [], []
        for i, chunk in enumerate(chunks):
            cid = hashlib.md5(f"{source_name}:{i}".encode()).hexdigest()
            ids.append(cid)
            docs.append(chunk)
            metas.append({"source": source_name, "chunk_index": i})
        if ids:
            self.collection.upsert(ids=ids, documents=docs, metadatas=metas)
        return len(ids)

    def retrieve(self, query: str, n_results: int = 4) -> list[str]:
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=min(n_results, self.collection.count() or 1),
            )
            return results["documents"][0] if results["documents"] else []
        except Exception as e:
            logger.warning(f"[RAG] Retrieval error: {e}")
            return []

    async def query(self, question: str) -> RAGContext:
        chunks = self.retrieve(question)
        if not chunks:
            return RAGContext(
                query=question,
                retrieved_chunks=[],
                source_files=[],
                already_resolved=False,
            )

        context_text = "\n---\n".join(chunks)
        messages = [
            SystemMessage(content=self.QUERY_SYSTEM),
            HumanMessage(content=f"User complaint: {question}\n\nDocumentation excerpts:\n{context_text}"),
        ]

        try:
            response = await self.llm.ainvoke(messages)
            raw = response.content.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            import json
            parsed = json.loads(raw.strip())
        except Exception as e:
            logger.warning(f"[RAG] LLM query parse error: {e}")
            parsed = {"already_resolved": False, "resolution_reference": None, "relevant_info": ""}

        return RAGContext(
            query=question,
            retrieved_chunks=chunks,
            source_files=[],
            already_resolved=bool(parsed.get("already_resolved", False)),
            resolution_reference=parsed.get("resolution_reference"),
        )