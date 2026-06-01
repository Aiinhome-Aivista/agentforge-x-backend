"""
app/core/rag_service.py — UPDATED
─────────────────────────────────────────────────────────────────────────────
The RAG layer now uses the new graph-grounded retriever that combines:

  • Vector similarity (query → in-graph node embeddings)
  • 1-hop graph traversal (workflow edges)
  • Similarity-matrix expansion (correlated cluster)

The previous flat ChromaDB retrieval is kept as a wide-net fallback so we
still answer when the process has not been indexed in the graph yet.
"""

import logging
from app.db.vector_service import collection
from app.core.mistral_client import get_mistral_client
from app.db.arango import get_graph_context
from app.services.knowledge_graph_service import retrieve_with_vectors_and_graph

logger = logging.getLogger(__name__)

llm = get_mistral_client()


def rag_query(query, process_key=None):
    """
    Returns a synthesised answer string, OR an empty string when no useful
    context could be found.
    """
    # ─── 1. PRIMARY: in-graph vector + matrix + graph retrieval ──────────
    primary_context = ""
    primary_hits    = 0
    if process_key:
        try:
            packet = retrieve_with_vectors_and_graph(
                query=query,
                process_key=process_key,
                top_k=5,
                matrix_expand=3,
            )
            primary_context = packet.get("context") or ""
            primary_hits    = len(packet.get("ranked") or [])
            logger.debug(
                f"[rag] primary retrieval: {primary_hits} hits, "
                f"{len(primary_context)} chars of context"
            )
        except Exception as e:
            logger.warning(f"[rag] graph-aware retrieval failed: {e}")

    # ─── 2. FALLBACK: original ChromaDB wide-net retrieval ───────────────
    enhanced_query = f"ERP process analysis: {query}"
    try:
        results = collection.query(query_texts=[enhanced_query], n_results=10)
        docs = results.get("documents", [[]])[0]
    except Exception as e:
        logger.warning(f"[rag] ChromaDB retrieval failed: {e}")
        docs = []

    seen, fallback_docs = set(), []
    for d in docs:
        if d and len(d.strip()) > 20:
            cd = d.strip()
            if cd not in seen:
                seen.add(cd)
                fallback_docs.append(cd)
    fallback_docs = fallback_docs[:5]
    fallback_context = "\n\n".join(fallback_docs)

    # ─── 3. Direct workflow context (steps) for sequencing questions ─────
    graph_context = ""
    if process_key:
        try:
            graph_data = get_graph_context(process_key)
            step_info = [
                f"Step {s.get('step_number')}: {s.get('title')} - {s.get('description')}"
                for s in graph_data.get("steps", [])
            ]
            graph_context = "\n".join(step_info)
        except Exception as e:
            logger.warning(f"[rag] get_graph_context failed: {e}")

    # ─── 4. Merge — prefer the in-graph context first ────────────────────
    blocks = []
    if primary_context:
        blocks.append(f"GRAPH-GROUNDED CONTEXT (vector + matrix + 1-hop):\n{primary_context}")
    if fallback_context:
        blocks.append(f"VECTOR-STORE CONTEXT (wide net):\n{fallback_context}")
    if graph_context:
        blocks.append(f"WORKFLOW STEPS:\n{graph_context}")
    final_context = "\n\n".join(blocks) or "(no context available)"

    prompt = f"""
You are a senior ERP process analyst.

Use the context below to answer the user's question.  Prefer information
from the GRAPH-GROUNDED CONTEXT when sources disagree, because it is
anchored to the specific process's knowledge graph.

Instructions:
- Identify relationships between steps using the workflow edges.
- Explain root causes using dependencies.
- Give precise, data-driven insights.
- If the context does not cover the question, say so plainly.

Context:
{final_context}

Question:
{query}
""".strip()

    return llm._chat("You are an ERP expert", prompt)
