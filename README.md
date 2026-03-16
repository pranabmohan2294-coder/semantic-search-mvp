# Semantic Search Engine (Hybrid Retrieval MVP)

This project implements a **Semantic Search Engine with Hybrid Retrieval** using vector embeddings and keyword search.

The system retrieves relevant document sections using both **semantic similarity and keyword matching**, then **re-ranks results** to improve retrieval accuracy.

This project is part of a hands-on roadmap to understand how **modern AI retrieval systems and RAG pipelines work**.

---

# Architecture Overview

The system follows a modern retrieval pipeline used in AI-powered search systems.

User Query
↓
Query Embedding
↓
Vector Retrieval (Semantic Search)
+
Keyword Retrieval (BM25)
↓
Candidate Pool
↓
Embedding Re-ranking
↓
Top Relevant Results


---

# Technologies Used

Embedding Model  
SentenceTransformers  
Model: `all-MiniLM-L6-v2`

Vector Database  
Chroma

Keyword Retrieval  
BM25 (rank-bm25)

Language  
Python

---

# Key Concepts Implemented

## 1. Document Chunking

Large documents are split into smaller chunks to improve retrieval precision.

Example:

Refund Policy
→ Refund eligibility  
→ Refund processing time  
→ Refund request method

Chunking improves:

- semantic representation
- retrieval accuracy
- context relevance

---

## 2. Embeddings

Each document chunk is converted into a **dense vector embedding** using SentenceTransformers.

Example:

"Refund eligibility rules"

↓

[0.213, -0.442, 0.129, ...]

Embeddings capture **semantic meaning** instead of exact keywords.

---

## 3. Vector Database

Embeddings are stored in **Chroma**, which enables fast similarity search.

Query embeddings are compared against stored document embeddings to find relevant content.

---

## 4. Hybrid Retrieval

The system combines:

Vector Search (semantic meaning)

+
Keyword Search (BM25)

Vector search works best for:

"how to get money back"

Keyword search works best for:

"refund policy"

Hybrid retrieval improves overall search quality.

---

## 5. Candidate Pool

Results from both retrieval methods are merged into a candidate set before ranking.

Vector Results → Top 5  
BM25 Results → Top 5  

Candidate Pool → Unique merged results

---

## 6. Re-Ranking

Candidate documents are re-ranked using embedding similarity.

Steps:

1. Encode candidate documents
2. Encode query
3. Compute similarity
4. Sort results by similarity score

This improves ranking accuracy.

---

# Example Queries

Example queries the system can answer:

refund policy  
refund eligibility  
how to get money back  
return conditions  

The system retrieves the most relevant document chunks from the policy dataset.

---

# Repository Structure

---

# Retrieval Pipeline

Documents
↓
Chunking
↓
Embeddings
↓
Vector DB (Chroma)
↓
Hybrid Retrieval
↓
Re-ranking
↓
Top Results

---

# Current Capabilities

✔ Semantic search  
✔ Vector similarity retrieval  
✔ Keyword search with BM25  
✔ Hybrid retrieval pipeline  
✔ Candidate pooling  
✔ Embedding-based re-ranking  

---

# Next Steps

Future improvements planned:

- Retrieval-Augmented Generation (RAG)
- LLM integration
- Context-aware answer generation
- Retrieval evaluation metrics
- Cross-encoder re-ranking
- Query expansion

---

# Learning Objective

This project demonstrates the core components behind modern AI search systems used in products like:

- Perplexity
- Notion AI
- ChatGPT Retrieval
- Enterprise knowledge assistants

---

# Author

Pranab Mohan

AI Product Manager Learning Project
