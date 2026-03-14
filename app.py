from sentence_transformers import SentenceTransformer
import chromadb
import os
import re
from rank_bm25 import BM25Okapi


# ---------------------------
# Load embedding model
# ---------------------------

model = SentenceTransformer("all-MiniLM-L6-v2")


# ---------------------------
# Initialize Chroma DB
# ---------------------------

client = chromadb.Client()

collection = client.create_collection(
    name="policy_documents"
)


# ---------------------------
# Semantic Chunking Function
# ---------------------------

def chunk_document(text):

    sections = re.split(r'TITLE:', text)

    chunks = []

    for section in sections:

        cleaned = section.strip()

        if len(cleaned) > 50:
            chunks.append(cleaned)

    return chunks


# ---------------------------
# Load Documents
# ---------------------------

data_folder = "data"

documents = []
metadatas = []
ids = []

for filename in os.listdir(data_folder):

    if not filename.endswith(".txt"):
        continue

    path = os.path.join(data_folder, filename)

    with open(path, "r", encoding="utf-8", errors="ignore") as f:

        text = f.read()

    chunks = chunk_document(text)

    for i, chunk in enumerate(chunks):

        documents.append(chunk)

        metadatas.append({
            "source": filename,
            "chunk": i
        })

        ids.append(f"{filename}_{i}")


print("Total chunks loaded:", len(documents))


# ---------------------------
# Create Embeddings
# ---------------------------

embeddings = model.encode(documents).tolist()


collection.add(
    documents=documents,
    embeddings=embeddings,
    metadatas=metadatas,
    ids=ids
)


print("Vector DB indexing complete")


# ---------------------------
# Build BM25 Index
# ---------------------------

tokenized_docs = [doc.split() for doc in documents]

bm25 = BM25Okapi(tokenized_docs)


print("BM25 index built")


# ---------------------------
# Query Loop
# ---------------------------

while True:

    query = input("\nEnter your search query (or type exit): ")

    if query.lower() == "exit":
        break


    # -----------------------
    # Vector Search
    # -----------------------

    query_embedding = model.encode([query]).tolist()

    vector_results = collection.query(
        query_embeddings=query_embedding,
        n_results=5
    )

    vector_docs = vector_results["documents"][0]


    # -----------------------
    # BM25 Keyword Search
    # -----------------------

    tokenized_query = query.split()

    bm25_scores = bm25.get_scores(tokenized_query)

    top_bm25_indices = sorted(
        range(len(bm25_scores)),
        key=lambda i: bm25_scores[i],
        reverse=True
    )[:5]

    bm25_docs = [documents[i] for i in top_bm25_indices]


    # -----------------------
    # Merge Candidate Pool
    # -----------------------

    candidate_docs = list(set(vector_docs + bm25_docs))


    # -----------------------
    # Re-Ranking Step
    # -----------------------

    candidate_embeddings = model.encode(candidate_docs)

    query_vec = model.encode(query)

    scored_results = []

    for i, doc in enumerate(candidate_docs):

        similarity = candidate_embeddings[i] @ query_vec

        scored_results.append((doc, similarity))


    ranked = sorted(scored_results, key=lambda x: x[1], reverse=True)


    # -----------------------
    # Final Output
    # -----------------------

    print("\nFinal Ranked Results:\n")

    for doc, score in ranked[:3]:

        print(doc[:400])
        print("Score:", score)
        print("------------")