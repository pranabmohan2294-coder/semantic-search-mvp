from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
import chromadb
import os
import re
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

client = chromadb.Client()
collection = client.create_collection(name="policy_eval")

def chunk_document(text):
    sections = re.split(r'TITLE:', text)
    chunks = []
    for section in sections:
        cleaned = section.strip()
        if len(cleaned) > 50:
            chunks.append(cleaned)
    return chunks

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

embeddings = model.encode(documents).tolist()
collection.add(documents=documents, embeddings=embeddings,
               metadatas=metadatas, ids=ids)

tokenized_docs = [doc.split() for doc in documents]
bm25 = BM25Okapi(tokenized_docs)
print(f"Loaded {len(documents)} chunks\n")


def retrieve(query, k=10, top_n=3):
    query_embedding = model.encode([query]).tolist()
    vector_results = collection.query(
        query_embeddings=query_embedding,
        n_results=k
    )
    vector_docs = vector_results["documents"][0]

    tokenized_query = query.split()
    bm25_scores = bm25.get_scores(tokenized_query)
    top_bm25_indices = sorted(
        range(len(bm25_scores)),
        key=lambda i: bm25_scores[i],
        reverse=True
    )[:k]
    bm25_docs = [documents[i] for i in top_bm25_indices]

    candidate_docs = list(set(vector_docs + bm25_docs))
    pairs = [(query, doc) for doc in candidate_docs]
    ce_scores = cross_encoder.predict(pairs)
    ranked = sorted(zip(ce_scores, candidate_docs), reverse=True)
    return [doc for _, doc in ranked[:top_n]]


def context_recall_score(retrieved_chunks, ground_truth):
    gt_words = set(ground_truth.lower().split())
    retrieved_text = " ".join(retrieved_chunks).lower()
    retrieved_words = set(retrieved_text.split())
    overlap = gt_words.intersection(retrieved_words)
    return round(len(overlap) / len(gt_words), 3)


def context_precision_score(retrieved_chunks, ground_truth):
    gt_embedding = model.encode([ground_truth])
    chunk_embeddings = model.encode(retrieved_chunks)
    scores = []
    for ce in chunk_embeddings:
        score = float(np.dot(gt_embedding[0], ce) /
                     (np.linalg.norm(gt_embedding[0]) * np.linalg.norm(ce)))
        scores.append(score)
    return round(float(np.mean(scores)), 3)


test_data = [
    {
        "question": "What is the refund policy?",
        "ground_truth": "Refunds are issued to the original payment method and partial refunds may apply for missing accessories."
    },
    {
        "question": "How do I get my money back?",
        "ground_truth": "Customers can request a refund which is returned to the original payment method used during purchase."
    },
    {
        "question": "What happens if my refund is rejected?",
        "ground_truth": "Customers may escalate refund disputes to the resolution team if the initial review is rejected."
    },
    {
        "question": "Are refunds automatic for cancelled orders?",
        "ground_truth": "Orders cancelled before shipping are refunded automatically to the original payment method."
    },
    {
        "question": "Can I get a partial refund?",
        "ground_truth": "Partial refunds may apply when returned products are missing accessories or promotional bundles."
    },
]

print("Running evaluation...\n")
print(f"{'Question':<45} {'Recall':>8} {'Precision':>10}")
print("-" * 65)

recall_scores = []
precision_scores = []

for item in test_data:
    q = item["question"]
    gt = item["ground_truth"]
    retrieved = retrieve(q)
    recall = context_recall_score(retrieved, gt)
    precision = context_precision_score(retrieved, gt)
    recall_scores.append(recall)
    precision_scores.append(precision)
    print(f"{q:<45} {recall:>8} {precision:>10}")

print("-" * 65)
avg_recall = round(sum(recall_scores)/len(recall_scores), 3)
avg_precision = round(sum(precision_scores)/len(precision_scores), 3)
print(f"{'AVERAGE':<45} {avg_recall:>8} {avg_precision:>10}")
print()
print("Benchmarks:")
print("  Recall    > 0.6 = good  |  retrieval finding right chunks")
print("  Precision > 0.7 = good  |  retrieved chunks are relevant")
print()
if avg_recall < 0.6:
    print("Action needed: recall low → raise k from 10 to 15, or increase chunk overlap")
if avg_precision < 0.7:
    print("Action needed: precision low → lower k or re-ranking threshold needs tuning")
if avg_recall >= 0.6 and avg_precision >= 0.7:
    print("Pipeline looks healthy. Ready for MVP 2 submission.")
