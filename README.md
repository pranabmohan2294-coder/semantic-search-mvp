# Semantic Search Engine — Hybrid Retrieval + Cross-Encoder Re-Ranking

This project implements a **Semantic Search Engine** using vector embeddings,
keyword search, and a **true cross-encoder re-ranker** for production-quality retrieval.

The system is part of a hands-on 30-day roadmap to understand how modern AI
retrieval systems and RAG pipelines work in production.

---

## Architecture
```
User Query
    ↓
Query Embedding (bi-encoder)
    ↓
Vector Retrieval (HNSW semantic search) — top 10
    +
Keyword Retrieval (BM25) — top 10
    ↓
Candidate Pool (merged, deduplicated)
    ↓
Cross-Encoder Re-Ranking (query + chunk read together)
    ↓
Top 3 Results
```

---

## Technologies

| Component | Tool |
|---|---|
| Embedding model | `all-MiniLM-L6-v2` (SentenceTransformers) |
| Vector database | Chroma (local, HNSW index) |
| Keyword search | BM25 (rank-bm25) |
| Re-ranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Language | Python |

---

## Key Concepts

### 1. Document Chunking
Documents are split on `TITLE:` markers into semantically coherent sections.
Each chunk is stored with metadata (source filename + chunk index) for traceability.

### 2. Bi-Encoder Embeddings
Each chunk is converted to a dense vector using `all-MiniLM-L6-v2`.
Embeddings capture semantic meaning — "how to get money back" matches "refund policy."

### 3. Hybrid Retrieval
Two retrieval signals are combined:
- **Vector search** — finds semantically similar chunks via HNSW cosine similarity
- **BM25** — finds exact keyword matches (product names, codes, specific terms)

Hybrid beats either signal alone, especially on domain-specific vocabulary.

### 4. Cross-Encoder Re-Ranking
This is the critical upgrade over a standard bi-encoder pipeline.

**Bi-encoder (old approach):**
- Embeds query and chunk independently
- Score = cosine distance between two separate vectors
- Fast but approximate — no direct query-chunk interaction

**Cross-encoder (current approach):**
- Reads query + chunk together as a single input
- Scores relevance as a pair — understands full context
- Slower, but significantly more accurate on complex queries

Model used: `cross-encoder/ms-marco-MiniLM-L-6-v2` — free, runs fully local.

### 5. Candidate Pool Strategy
Wide net → precise filter:
- Retrieve top 10 from vector search
- Retrieve top 10 from BM25
- Merge and deduplicate into candidate pool
- Cross-encoder scores all candidates
- Return top 3 to user

---

## Example Queries
```
refund policy
how to get money back
return conditions
refund eligibility
```

---

## Repository Structure
```
├── app.py          # Main retrieval pipeline
├── data/           # .txt policy documents
└── README.md
```

---

## Retrieval Pipeline
```
Documents → Chunking → Embeddings → Chroma (HNSW)
                                         ↓
Query → Embed → Vector Search (k=10) ─┐
Query → BM25 Search (k=10) ───────────┤
                                       ↓
                              Candidate Pool
                                       ↓
                         Cross-Encoder Re-Ranking
                                       ↓
                               Top 3 Results
```

---

## Capabilities

- Semantic search via dense vector retrieval
- Keyword search via BM25
- Hybrid retrieval pipeline
- True cross-encoder re-ranking (query-chunk pair scoring)
- Source metadata tracking per result

---

## What Changed in This Commit

Upgraded re-ranking from bi-encoder dot product similarity to a true
cross-encoder (`ms-marco-MiniLM-L-6-v2`). Previous approach re-scored
candidates using the same embedding model — not a real cross-encoder.
Cross-encoder reads query and chunk jointly, producing significantly
more accurate relevance scores on complex queries.

Also widened candidate pool from k=5 to k=10 for both retrieval
signals to give the cross-encoder more candidates to work with.

---

## Next Steps

- RAG pipeline — add Ollama LLM for answer generation (Week 2)
- RAGAS evaluation — faithfulness + answer relevancy scores
- Chunking experiment — compare fixed vs recursive vs semantic
- Query expansion / HyDE

---

## Learning Objective

Demonstrates the core retrieval architecture behind products like
Perplexity, Notion AI, and enterprise knowledge assistants.

---

## Author

Pranab Mohan  
AI Product Manager
