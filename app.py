from sentence_transformers import SentenceTransformer, CrossEncoder
import chromadb
import os
import re
from rank_bm25 import BM25Okapi
import requests
import json


# ---------------------------
# Load Models
# ---------------------------

model = SentenceTransformer("all-MiniLM-L6-v2")
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


# ---------------------------
# Initialize Chroma DB
# ---------------------------

client = chromadb.Client()

collection = client.create_collection(
    name="policy_documents"
)


# ---------------------------
# Chunking Function
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
        metadatas.append({"source": filename, "chunk": i})
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
# Ollama Generation Function
# ---------------------------

def generate_answer(query, context_chunks):

    context = "\n\n".join([
        f"[Source {i+1}]\n{chunk}"
        for i, chunk in enumerate(context_chunks)
    ])

    prompt = f"""You are a helpful customer service assistant. Use the context below to answer the question.
The context contains policy information. Interpret it helpfully and connect it to what the user is asking.
Only say you don't know if the context truly contains nothing relevant.
Be concise and direct.

Context:
{context}

Question: {query}

Answer:"""

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3.2",
            "prompt": prompt,
            "stream": False
        }
    )

    result = response.json()
    return result["response"].strip()


# ---------------------------
# Query Loop
# ---------------------------

while True:

    query = input("\nEnter your question (or type exit): ")

    if query.lower() == "exit":
        break


    # -----------------------
    # Vector Search
    # -----------------------

    query_embedding = model.encode([query]).tolist()

    vector_results = collection.query(
        query_embeddings=query_embedding,
        n_results=10
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
    )[:10]

    bm25_docs = [documents[i] for i in top_bm25_indices]


    # -----------------------
    # Merge Candidate Pool
    # -----------------------

    candidate_docs = list(set(vector_docs + bm25_docs))


    # -----------------------
    # Cross-Encoder Re-Ranking
    # -----------------------

    pairs = [(query, doc) for doc in candidate_docs]
    ce_scores = cross_encoder.predict(pairs)
    scored_results = list(zip(ce_scores, candidate_docs))
    ranked = sorted(scored_results, key=lambda x: x[0], reverse=True)

    top_chunks = [doc for _, doc in ranked[:3]]


    # -----------------------
    # Generate Answer
    # -----------------------

    print("\nGenerating answer...\n")
    answer = generate_answer(query, top_chunks)

    print("Answer:")
    print(answer)
    print("\nSources used:")
    for i, chunk in enumerate(top_chunks):
        print(f"\n[{i+1}] {chunk[:200]}...")
        print(f"     Score: {round(float(ranked[i][0]), 4)}")