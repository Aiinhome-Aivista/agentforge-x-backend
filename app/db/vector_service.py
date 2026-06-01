"""
app/db/vector_service.py — UPDATED

Now stores embeddings in TWO places:
  1.  ChromaDB           (existing wide-net retrieval — unchanged).
  2.  ArangoDB nodes     (NEW — vectors live on the graph documents
                          themselves so the new graph-grounded RAG layer
                          can use them WITHOUT a separate similarity search
                          round-trip).

After persisting to both stores, this module also triggers the knowledge-
graph indexer, which:
  - Re-embeds every node belonging to the process (one source of truth).
  - Computes the pairwise similarity matrix and stores it in
    `similarity_matrices`.
  - Persists high-similarity edges in `node_similarity` for graph
    traversal queries.
"""

from __future__ import annotations

import logging
import chromadb
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# ✅ Persistent DB
client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_or_create_collection(name="process_docs")

model = SentenceTransformer("all-MiniLM-L6-v2")


def store_embeddings(process_doc, steps, insights):
    """
    Persist embeddings in ChromaDB AND trigger the in-graph indexer.

    The ChromaDB path is preserved exactly so existing code that queries
    `collection.query(...)` keeps working.
    """
    docs, ids, metadatas = [], [], []

    # Process description
    if getattr(process_doc, "description", None):
        docs.append(process_doc.description)
        ids.append(f"process_{process_doc._key}")
        metadatas.append({"type": "process", "process_key": process_doc._key})

    # Steps
    for step in steps:
        text = f"{getattr(step, 'title', '')}: {getattr(step, 'description', '')}".strip(": ").strip()
        if text:
            docs.append(text)
            ids.append(f"step_{step._key}")
            metadatas.append({"type": "step", "process_key": process_doc._key})

    # Insights
    for i, insight in enumerate(insights):
        text = getattr(insight, "text", None) or str(insight)
        if text:
            docs.append(text)
            ids.append(f"insight_{i}_{process_doc._key}")
            metadatas.append({"type": "insight", "process_key": process_doc._key})

    if docs:
        logger.info(f"[vector] storing {len(docs)} docs in ChromaDB for process {process_doc._key}")
        embeddings = model.encode(docs).tolist()
        try:
            collection.add(
                documents=docs,
                embeddings=embeddings,
                ids=ids,
                metadatas=metadatas,
            )
        except Exception as e:
            # Most likely cause: re-running for the same process — delete + add
            logger.warning(f"[vector] add failed, retrying with delete: {e}")
            try:
                collection.delete(ids=ids)
                collection.add(
                    documents=docs,
                    embeddings=embeddings,
                    ids=ids,
                    metadatas=metadatas,
                )
            except Exception as e2:
                logger.error(f"[vector] retry add failed: {e2}")

    # ── NEW: also index this process in the graph (vectors + matrix) ──
    try:
        # Imported lazily so this file stays importable when the new module
        # has not been deployed yet.
        from app.services.knowledge_graph_service import index_process_in_graph
        report = index_process_in_graph(process_doc._key)
        logger.info(f"[vector] graph index report: {report}")
    except Exception as e:
        logger.warning(f"[vector] in-graph indexing skipped: {e}")
