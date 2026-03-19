from app.db.vector_service import collection
from app.core.mistral_client import get_mistral_client
from app.db.arango import get_graph_context

llm = get_mistral_client()

def rag_query(query, process_key=None):

    # 🔹 Vector Search
    enhanced_query = f"ERP process analysis: {query}"

    results = collection.query(
        query_texts=[enhanced_query],
        n_results=10
    )

    docs = results.get("documents", [[]])[0]

    # ✅ remove duplicates + clean docs
    seen = set()
    filtered_docs = []

    for d in docs:
        if d and len(d.strip()) > 20:
            clean_d = d.strip()

            # remove duplicate
            if clean_d not in seen:
                seen.add(clean_d)
                filtered_docs.append(clean_d)

    # limit top 5
    filtered_docs = filtered_docs[:5]

    vector_context = "\n\n".join(filtered_docs)

    # 🔹 Graph Context
    graph_context = ""

    if process_key:
        graph_data = get_graph_context(process_key)

        step_info = [
            f"Step {s.get('step_number')}: {s.get('title')} - {s.get('description')}"
            for s in graph_data.get("steps", [])
        ]

        graph_context = "\n".join(step_info)

    # 🔥 COMBINE BOTH
    final_context = f"""
    VECTOR CONTEXT:
    {vector_context}

    GRAPH CONTEXT:
    {graph_context}
    """

    print("FINAL CONTEXT:", final_context)

    prompt = f"""
    You are a senior ERP process analyst.

    Use BOTH vector and graph context.

    Instructions:
    - Identify relationships between steps
    - Explain root causes using dependencies
    - Give precise, data-driven insights

    Context:
    {final_context}

    Question:
    {query}
    """

    return llm._chat("You are an ERP expert", prompt)