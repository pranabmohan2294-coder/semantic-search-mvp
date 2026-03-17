# RAG Pipeline — Hybrid Retrieval + Cross-Encoder + LLM Generation

A production-quality RAG (Retrieval-Augmented Generation) system built
as part of a 30-day AI PM learning sprint. Week 1 was semantic search.
Week 2 is the full RAG pipeline — retrieval, re-ranking, generation,
and evaluation.

---

## Architecture
```
Documents → Chunking → Embeddings → Chroma (HNSW index)
                                           ↓
User Query → Embed → Vector Search (k=10) ─┐
User Query → BM25 Search (k=10) ───────────┤
                                            ↓
                                   Candidate Pool
                                            ↓
                          Cross-Encoder Re-Ranking
                          (query + chunk scored as pair)
                                            ↓
                                        Top 3 Chunks
                                            ↓
                             Ollama llama3.2 (local LLM)
                                            ↓
                          Natural language answer + citations
```

---

## Technologies

| Component | Tool |
|---|---|
| Embedding model | `all-MiniLM-L6-v2` (SentenceTransformers, local) |
| Vector database | Chroma (local, HNSW index) |
| Keyword search | BM25 (rank-bm25) |
| Re-ranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` (local) |
| LLM generation | Ollama llama3.2 (local, free) |
| Evaluation | Custom recall + precision scoring (evaluate.py) |
| Language | Python 3.9 |

Zero API costs. Fully local. No data leaves the machine.

---

## Evaluation Results

Tested on 5 policy queries with ground truth answers.

| Metric | Score | Benchmark | Status |
|---|---|---|---|
| Context recall | 0.886 | > 0.6 | PASS |
| Context precision | 0.731 | > 0.7 | PASS |

**Weak spot identified:** Colloquial queries ("how do I get my money
back") score recall 0.562 — below benchmark. Root cause: vocabulary
gap between natural language and policy document language. BM25
contributes nothing when there is zero keyword overlap.

**Fix planned for V2:** Query rewriting — rewrite colloquial user
queries into policy language before retrieval using an LLM call.

---

## Key Concepts

### 1. Hybrid Retrieval
Combines two signals:
- Vector search — semantic similarity via HNSW cosine distance
- BM25 — exact keyword matching

Hybrid beats either signal alone. Vector search handles "how do I get
my money back." BM25 handles "Error code E404-B."

### 2. Cross-Encoder Re-Ranking
The critical upgrade from Week 1.

**Bi-encoder (Week 1):** embeds query and chunk independently, scores
via cosine distance. Fast but no direct query-chunk interaction.

**Cross-encoder (Week 2):** reads query + chunk together as a pair.
Scores relevance jointly. Slower but significantly more accurate on
complex queries.

Scores are raw logits — unbounded, can be positive or negative.
What matters is the ranking, not the absolute value.

- High positive score (e.g. 6.78) = strong match, high confidence
- Negative scores clustered close = weak match, low confidence
- All scores very negative and clustered = nothing relevant in corpus

### 3. LLM Generation with Faithfulness Guardrail
Retrieved chunks are passed to Ollama llama3.2 with a prompt that
instructs the model to answer only from provided context. If no
relevant context exists, the system says "I don't have enough
information" rather than hallucinating.

Tested on out-of-scope queries ("car wash policy", "food policy") —
system correctly declined to answer both times.

### 4. Evaluation Framework
`evaluate.py` measures retrieval quality against ground truth answers:
- Context recall — did retrieval find the right information?
- Context precision — are retrieved chunks genuinely relevant?

---

## Failure Cases Documented

1. Colloquial queries (recall 0.562) — vocabulary gap, fix via query rewriting
2. Out-of-scope queries — handled correctly, guardrail working
3. Chunk titles use policy language, not user-intent language — semantic gap
4. No faithfulness score yet — requires LLM-as-judge (V2)
5. Negative cross-encoder scores on weak matches — expected behaviour, not a bug

---

## Repository Structure
```
├── app.py          # Full RAG pipeline — retrieval + generation
├── evaluate.py     # Evaluation framework — recall + precision
├── data/           # .txt policy documents (5 files, 75 chunks)
└── README.md
```

---

## V2 Roadmap

- Query rewriting before retrieval (fix colloquial query recall)
- Faithfulness scoring with Ollama as judge
- Upgrade embedding model to `all-mpnet-base-v2` (768 dims)
- Confidence threshold guardrail (decline when all scores clustered low)
- 50-question test set from real user queries

---

## Week-by-Week Progress

**Week 1 — Semantic Search Engine**
Hybrid retrieval (HNSW + BM25), true cross-encoder re-ranking,
evaluation framework. Scores: recall 0.886, precision 0.731.

**Week 2 — RAG Pipeline**
Added Ollama LLM generation, prompt template with faithfulness
guardrail, full end-to-end RAG pipeline. All components local and free.

---

## Learning Objective

Built as part of a 30-day AI PM sprint to go from AI-aware to
AI-fluent. Demonstrates the core architecture behind products like
Perplexity, Notion AI, and enterprise knowledge assistants.

---

## Author

Pranab Mohan
AI Product Manager — 30-Day Learning Sprint
