"""
app/services/knowledge_graph_service.py
─────────────────────────────────────────────────────────────────────────────
Knowledge Graph Enhancer

Spec update (this iteration):
    "in the knowledge graph section also save matrix and the vector in
     the graph for more enhance response"

What this module does
─────────────────────
1.  **Vectors in the graph**:  When we ingest a process, every node
    (process · step · suggestion · ERP module · insight) gets its
    embedding vector persisted directly on the ArangoDB document under
    `embedding: [...]`.

2.  **Similarity matrix in the graph**:  We compute pairwise cosine
    similarities between all nodes in the process and persist the
    upper-triangular matrix as a new edge collection (`node_similarity`)
    and a per-process flat matrix doc (`similarity_matrices`).

3.  **Retrieval helpers**:  `retrieve_with_vectors_and_graph(...)` returns
    a ranked, deduplicated set of nodes that combines:
      - vector similarity to the query
      - graph proximity (1-hop neighbours via existing edges)
      - matrix-based cluster expansion (nodes that are tightly correlated
        to high-scoring vector hits, even if not direct neighbours)

The original vector store (ChromaDB) keeps working as a wide-net retriever;
this module adds an authoritative, in-graph copy that the RAG layer prefers
because it inherits ALL the structural relationships from the workflow.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from app.db.arango import get_db, COLLECTIONS, EDGE_COLLECTIONS
from app.db.vector_service import model as _embedder      # already loaded sentence-transformer

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
SIMILARITY_EDGE_COLL  = "node_similarity"
SIMILARITY_MATRIX_COLL = "similarity_matrices"
EMBEDDING_FIELD       = "embedding"
SIM_FIELD             = "similarity"

# A node pair below this threshold is considered noise — skip the edge
SIM_EDGE_THRESHOLD    = 0.45
# How many neighbours to include per node when building cluster expansion
MAX_NEIGHBOURS_PER_NODE = 6


# ─────────────────────────────────────────────────────────────────────────────
# Schema bootstrap — idempotent
# ─────────────────────────────────────────────────────────────────────────────
def ensure_collections() -> None:
    db = get_db().db   # raw ArangoDB connection
    # Node collection for per-process similarity matrices
    if not db.has_collection(SIMILARITY_MATRIX_COLL):
        db.create_collection(SIMILARITY_MATRIX_COLL)
        logger.info(f"[kg] created collection: {SIMILARITY_MATRIX_COLL}")
    # Edge collection for pairwise high-similarity edges
    if not db.has_collection(SIMILARITY_EDGE_COLL):
        db.create_collection(SIMILARITY_EDGE_COLL, edge=True)
        logger.info(f"[kg] created edge collection: {SIMILARITY_EDGE_COLL}")


# ─────────────────────────────────────────────────────────────────────────────
# Embedding helpers
# ─────────────────────────────────────────────────────────────────────────────
def _embed(text: str) -> List[float]:
    if not text or not text.strip():
        return []
    return _embedder.encode([text.strip()])[0].tolist()


def _embed_batch(texts: Sequence[str]) -> List[List[float]]:
    if not texts:
        return []
    clean = [t.strip() if t else "" for t in texts]
    return _embedder.encode(clean).tolist()


def _cosine(v1: Sequence[float], v2: Sequence[float]) -> float:
    if not v1 or not v2:
        return 0.0
    a = np.asarray(v1, dtype=float)
    b = np.asarray(v2, dtype=float)
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# ─────────────────────────────────────────────────────────────────────────────
# Index a single process (called from the analysis pipeline AFTER persistence)
# ─────────────────────────────────────────────────────────────────────────────
def index_process_in_graph(process_key: str) -> Dict[str, Any]:
    """
    Embed every node belonging to `process_key`, persist the vector on the
    document, compute the pairwise similarity matrix, store the matrix and
    the high-similarity edges.

    Returns a small report dict.
    """
    ensure_collections()
    db = get_db()

    # ── 1. Collect every node tied to this process ──────────────────────
    nodes = _collect_process_nodes(process_key)
    if not nodes:
        logger.warning(f"[kg] no nodes found for process {process_key}")
        return {"process_key": process_key, "indexed_nodes": 0, "edges": 0}

    # ── 2. Build the text representation of each node ───────────────────
    texts = [_text_for(node) for node in nodes]

    # ── 3. Embed in batch and persist back to the node documents ────────
    embeddings = _embed_batch(texts)
    updated = 0
    for node, vec in zip(nodes, embeddings):
        try:
            coll = db.collection(node["_coll_"])
            coll.update({
                "_key":          node["_key"],
                EMBEDDING_FIELD: vec,
                "embedded_at":   datetime.utcnow().isoformat() + "Z",
            })
            updated += 1
        except Exception as e:
            logger.warning(f"[kg] embed-persist failed for {node['_coll_']}/{node['_key']}: {e}")

    # ── 4. Pairwise similarity matrix ───────────────────────────────────
    n = len(nodes)
    matrix = np.zeros((n, n), dtype=float)
    arr = np.asarray(embeddings, dtype=float)
    if arr.size > 0 and arr.shape[1] > 0:
        norms = np.linalg.norm(arr, axis=1)
        # guard against zero vectors
        norms[norms == 0] = 1e-12
        normed = arr / norms[:, None]
        matrix = normed @ normed.T

    # ── 5. Persist the matrix as a single document for fast access ──────
    matrix_doc = {
        "_key": f"matrix_{process_key}",
        "process_key": process_key,
        "node_count": n,
        "node_ids":   [f"{x['_coll_']}/{x['_key']}" for x in nodes],
        "node_kinds": [x["_kind_"] for x in nodes],
        "node_labels":[x["_label_"] for x in nodes],
        # Round to keep storage modest — 4 decimals is plenty for retrieval
        "matrix":     [[round(float(v), 4) for v in row] for row in matrix.tolist()],
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "dim":        int(arr.shape[1]) if arr.ndim == 2 else 0,
    }
    mcoll = db.collection(SIMILARITY_MATRIX_COLL)
    try:
        mcoll.insert(matrix_doc, overwrite=True)
    except TypeError:
        # python-arango versions where overwrite kw isn't supported
        try: mcoll.delete(matrix_doc["_key"])
        except Exception: pass
        mcoll.insert(matrix_doc)

    # ── 6. Persist high-similarity edges (so GraphTraversal can use them)
    edge_count = _persist_similarity_edges(process_key, nodes, matrix)

    logger.info(
        f"[kg] indexed process {process_key}: "
        f"{updated}/{n} nodes embedded, {edge_count} similarity edges, "
        f"matrix {n}x{n} stored."
    )
    return {
        "process_key":   process_key,
        "indexed_nodes": updated,
        "matrix_shape":  [n, n],
        "edges":         edge_count,
        "dim":           matrix_doc["dim"],
    }


def _persist_similarity_edges(
    process_key: str,
    nodes: List[Dict[str, Any]],
    matrix: np.ndarray,
) -> int:
    db = get_db()
    ecoll = db.collection(SIMILARITY_EDGE_COLL)

    # Remove prior edges for this process to keep things idempotent
    try:
        db.aql(
            "FOR e IN @@coll FILTER e.process_key == @k REMOVE e IN @@coll",
            {"@coll": SIMILARITY_EDGE_COLL, "k": process_key},
        )
    except Exception as e:
        logger.debug(f"[kg] edge cleanup skipped: {e}")

    n = len(nodes)
    edges: List[Dict[str, Any]] = []
    for i in range(n):
        # Keep top K most-similar neighbours above threshold
        scored: List[Tuple[int, float]] = []
        for j in range(n):
            if i == j: continue
            sim = float(matrix[i][j])
            if sim >= SIM_EDGE_THRESHOLD:
                scored.append((j, sim))
        scored.sort(key=lambda x: -x[1])
        for j, sim in scored[:MAX_NEIGHBOURS_PER_NODE]:
            edges.append({
                "_from":       f"{nodes[i]['_coll_']}/{nodes[i]['_key']}",
                "_to":         f"{nodes[j]['_coll_']}/{nodes[j]['_key']}",
                "process_key": process_key,
                SIM_FIELD:     round(sim, 4),
                "kind_from":   nodes[i]["_kind_"],
                "kind_to":     nodes[j]["_kind_"],
            })

    if not edges:
        return 0
    try:
        ecoll.insert_many(edges)
    except Exception as e:
        # Fall back to one-by-one in case of unique-key conflicts
        logger.warning(f"[kg] insert_many failed, retrying singly: {e}")
        ok = 0
        for edge in edges:
            try:
                ecoll.insert(edge)
                ok += 1
            except Exception as e2:
                logger.debug(f"[kg] edge insert failed: {e2}")
        return ok
    return len(edges)


# ─────────────────────────────────────────────────────────────────────────────
# Node collection — pulls every entity attached to the process
# ─────────────────────────────────────────────────────────────────────────────
def _collect_process_nodes(process_key: str) -> List[Dict[str, Any]]:
    db = get_db()
    nodes: List[Dict[str, Any]] = []

    # Process itself
    try:
        proc_coll = COLLECTIONS["documents"]
        proc = db.collection(proc_coll).get(process_key)
        if proc:
            nodes.append({
                **proc,
                "_coll_":  proc_coll,
                "_kind_":  "process",
                "_label_": proc.get("title") or "Process",
            })
    except Exception as e:
        logger.debug(f"[kg] process lookup failed: {e}")

    # Steps
    try:
        for s in db.aql(
            "FOR s IN process_steps FILTER s.process_key == @k SORT s.step_number RETURN s",
            {"k": process_key},
        ):
            nodes.append({
                **s,
                "_coll_":  "process_steps",
                "_kind_":  "step",
                "_label_": s.get("title") or f"Step {s.get('step_number','')}",
            })
    except Exception as e:
        logger.debug(f"[kg] step lookup failed: {e}")

    # Suggestions
    try:
        for s in db.aql(
            "FOR s IN automation_suggestions FILTER s.process_key == @k RETURN s",
            {"k": process_key},
        ):
            nodes.append({
                **s,
                "_coll_":  "automation_suggestions",
                "_kind_":  "suggestion",
                "_label_": s.get("title") or "Suggestion",
            })
    except Exception as e:
        logger.debug(f"[kg] suggestion lookup failed: {e}")

    # ERP modules
    try:
        for m in db.aql(
            "FOR m IN erp_modules FILTER m.process_key == @k RETURN m",
            {"k": process_key},
        ):
            nodes.append({
                **m,
                "_coll_":  "erp_modules",
                "_kind_":  "module",
                "_label_": m.get("name") or "Module",
            })
    except Exception as e:
        logger.debug(f"[kg] module lookup failed: {e}")

    return nodes


def _text_for(node: Dict[str, Any]) -> str:
    """Build a content string for embedding."""
    kind = node["_kind_"]
    parts: List[str] = [node["_label_"]]
    if kind == "process":
        parts.append(node.get("description", ""))
        parts.append(f"ERP: {node.get('erp', '')}")
    elif kind == "step":
        parts.append(f"Step {node.get('step_number','')}")
        parts.append(node.get("description", ""))
        parts.append(f"Actor: {node.get('actor', '')}")
        parts.append(f"Automation potential: {node.get('automation_potential', 0)}%")
    elif kind == "suggestion":
        parts.append(node.get("description", ""))
        parts.append(node.get("agent_type", ""))
        parts.append(f"ROI: {node.get('roi_impact', '')}")
    elif kind == "module":
        parts.append(node.get("description", ""))
        parts.append(",".join(node.get("entities", []) or []))
    return " | ".join(p for p in parts if p)


# ─────────────────────────────────────────────────────────────────────────────
# Retrieval — combines vector ranking + graph + similarity matrix
# ─────────────────────────────────────────────────────────────────────────────
def retrieve_with_vectors_and_graph(
    query: str,
    process_key: str,
    *,
    top_k: int = 5,
    matrix_expand: int = 3,
) -> Dict[str, Any]:
    """
    Returns:
        {
          "vector_hits":   [...],   # top-K matches ranked by query similarity
          "graph_hits":    [...],   # 1-hop neighbours of the top vector hits
          "matrix_hits":   [...],   # additional nodes pulled from the matrix
                                    # (tightly correlated to the top hits)
          "ranked":        [...],   # final, deduplicated, ranked list
          "context":       "...",   # concatenated text block ready for an LLM
        }
    """
    if not process_key or not query:
        return {"vector_hits": [], "graph_hits": [], "matrix_hits": [],
                "ranked": [], "context": ""}

    db = get_db()

    # Load the matrix once
    try:
        mcoll = db.collection(SIMILARITY_MATRIX_COLL)
        matrix_doc = mcoll.get(f"matrix_{process_key}") or {}
    except Exception:
        matrix_doc = {}

    node_ids: List[str] = matrix_doc.get("node_ids", []) or []
    node_labels: List[str] = matrix_doc.get("node_labels", []) or []
    node_kinds: List[str] = matrix_doc.get("node_kinds", []) or []
    sim_matrix: List[List[float]] = matrix_doc.get("matrix", []) or []

    if not node_ids:
        # Graph hasn't been indexed yet — try to index lazily, then retry once
        try:
            index_process_in_graph(process_key)
            matrix_doc = mcoll.get(f"matrix_{process_key}") or {}
            node_ids = matrix_doc.get("node_ids", []) or []
            node_labels = matrix_doc.get("node_labels", []) or []
            node_kinds = matrix_doc.get("node_kinds", []) or []
            sim_matrix = matrix_doc.get("matrix", []) or []
        except Exception as e:
            logger.warning(f"[kg] lazy index failed: {e}")
            return {"vector_hits": [], "graph_hits": [], "matrix_hits": [],
                    "ranked": [], "context": ""}

    # 1. Vector hits — embed the query and cosine-score against every node
    query_vec = _embed(query)
    if not query_vec:
        return {"vector_hits": [], "graph_hits": [], "matrix_hits": [],
                "ranked": [], "context": ""}

    # Fetch embeddings from the node documents
    node_vecs: List[List[float]] = []
    for nid in node_ids:
        try:
            doc = db.aql(
                "RETURN DOCUMENT(@id)",
                {"id": nid},
            )
            doc_list = list(doc)
            embedding = (doc_list[0] or {}).get(EMBEDDING_FIELD) if doc_list else None
            node_vecs.append(embedding or [])
        except Exception:
            node_vecs.append([])

    scored: List[Tuple[int, float]] = []
    for i, v in enumerate(node_vecs):
        if v:
            scored.append((i, _cosine(query_vec, v)))
    scored.sort(key=lambda x: -x[1])
    top = scored[:top_k]

    vector_hits = [{
        "id":         node_ids[i],
        "label":      node_labels[i] if i < len(node_labels) else "",
        "kind":       node_kinds[i]  if i < len(node_kinds)  else "",
        "score":      round(score, 4),
        "source":     "vector",
    } for i, score in top]

    # 2. Graph hits — 1-hop neighbours of every top vector hit via the real
    # workflow edges (step_sequence, triggers_suggestion, etc.)
    top_ids = [hit["id"] for hit in vector_hits]
    graph_hits: List[Dict[str, Any]] = []
    if top_ids:
        try:
            edge_colls = ", ".join(EDGE_COLLECTIONS.values())
            cursor = db.aql(
                f"""
                FOR start IN @start_ids
                  LET doc = DOCUMENT(start)
                  FILTER doc != null
                  FOR v, e IN 1..1 ANY doc {edge_colls}
                    RETURN DISTINCT {{
                      id:    v._id,
                      label: v.title || v.name || v.label,
                      kind:  e._collection_kind || "",
                      edge:  e._id
                    }}
                """,
                {"start_ids": top_ids},
            )
            graph_hits = [g for g in cursor if g.get("id") not in top_ids]
        except Exception as e:
            logger.debug(f"[kg] graph traversal failed: {e}")

    # 3. Matrix hits — pull additional correlated nodes per top vector hit
    matrix_hits: List[Dict[str, Any]] = []
    if sim_matrix:
        seen_ids = set(top_ids)
        for i, _ in top:
            row = sim_matrix[i] if i < len(sim_matrix) else []
            ranked = sorted(
                ((j, s) for j, s in enumerate(row) if j != i and s >= SIM_EDGE_THRESHOLD),
                key=lambda x: -x[1],
            )[:matrix_expand]
            for j, sim in ranked:
                nid = node_ids[j] if j < len(node_ids) else None
                if not nid or nid in seen_ids:
                    continue
                seen_ids.add(nid)
                matrix_hits.append({
                    "id":     nid,
                    "label":  node_labels[j] if j < len(node_labels) else "",
                    "kind":   node_kinds[j]  if j < len(node_kinds)  else "",
                    "score":  round(sim, 4),
                    "source": "matrix",
                    "origin": node_ids[i],
                })

    # 4. Merge & rank
    seen = set()
    ranked: List[Dict[str, Any]] = []
    for hit in vector_hits + matrix_hits + graph_hits:
        nid = hit.get("id")
        if not nid or nid in seen:
            continue
        seen.add(nid)
        ranked.append(hit)

    # 5. Build a context block by re-reading the actual node text
    context_lines: List[str] = []
    for hit in ranked[:12]:
        try:
            doc_list = list(db.aql("RETURN DOCUMENT(@id)", {"id": hit["id"]}))
            doc = doc_list[0] if doc_list else None
        except Exception:
            doc = None
        if not doc: continue
        label = doc.get("title") or doc.get("name") or doc.get("label") or hit.get("label","")
        desc  = doc.get("description") or doc.get("text") or ""
        kind  = hit.get("kind") or doc.get("_id","").split("/")[0]
        context_lines.append(
            f"[{kind.upper()} · score {hit.get('score','-')}] {label}\n{desc}"
        )

    return {
        "vector_hits": vector_hits,
        "graph_hits":  graph_hits,
        "matrix_hits": matrix_hits,
        "ranked":      ranked,
        "context":     "\n\n".join(context_lines),
    }
