import chromadb
from sentence_transformers import SentenceTransformer

# ✅ Persistent DB
client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_or_create_collection(name="process_docs")

model = SentenceTransformer("all-MiniLM-L6-v2")


def store_embeddings(process_doc, steps, insights):
    docs = []
    ids = []

    # Process description
    docs.append(process_doc.description)
    ids.append(f"process_{process_doc._key}")

    # Steps
    for step in steps:
        docs.append(f"{step.title}: {step.description}")
        ids.append(f"step_{step._key}")

    # Insights
    for i, insight in enumerate(insights):
        docs.append(insight.text)
        ids.append(f"insight_{i}_{process_doc._key}")

    print("Storing documents in VectorDB:", len(docs))  # ✅ debug

    embeddings = model.encode(docs).tolist()

    collection.add(
        documents=docs,
        embeddings=embeddings,
        ids=ids,
        metadatas=[{"type": "process"}] + 
                  [{"type": "step"} for _ in steps] +
                  [{"type": "insight"} for _ in insights]
    )