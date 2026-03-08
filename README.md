# Newsgroups Semantic Search

A lightweight semantic search system over the 20 Newsgroups corpus (~18,400 documents), featuring:

- **Fuzzy clustering** using UMAP + Fuzzy C-Means (soft assignments, not hard labels)
- **Cluster-aware semantic cache** built from scratch — no Redis, no caching libraries
- **FastAPI service** with proper state management and three REST endpoints

---

## Project Structure

```
newsgroups-semantic-search/
├── app/
│   ├── cache.py          # Semantic cache (from scratch)
│   ├── cluster.py        # Soft cluster assignment at query time
│   ├── embedder.py       # Sentence-transformer wrapper
│   ├── main.py           # FastAPI app + endpoints
│   └── search.py         # Qdrant retrieval
├── data/
│   ├── raw/              # Unpack 20_newsgroups.tar.gz here
│   └── processed/        # Output of preprocess.py (cleaned_docs.jsonl)
├── embeddings/
│   └── qdrant_index/     # Output of build_index.py
├── models/               # Output of train_clusters.py
├── scripts/
│   ├── preprocess.py     # Step 1: clean raw documents
│   ├── build_index.py    # Step 2: embed + store in Qdrant
│   └── train_clusters.py # Step 3: UMAP + FCM clustering
├── notebooks/
│   └── clustering_analysis.ipynb
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Setup

### 1. Clone and create virtual environment

```bash
git clone <repo-url>
cd newsgroups-semantic-search
python -m venv venv

# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Prepare the dataset

Download the [20 Newsgroups dataset](https://archive.ics.uci.edu/dataset/113/twenty+newsgroups) and unpack:

```bash
cd data/raw
unzip twenty_newsgroups.zip
tar -xzf 20_newsgroups.tar.gz
cd ../..
```

### 3. Run the pipeline (one-time setup)

```bash
# Step 1: Clean and preprocess (~10 seconds)
python scripts/preprocess.py

# Step 2: Embed documents and build Qdrant index (~8 minutes on CPU)
python scripts/build_index.py

# Step 3: UMAP reduction + Fuzzy C-Means clustering (~2 minutes)
python scripts/train_clusters.py
```

### 4. Start the API

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## API Endpoints

### `POST /query`

Semantic search with cache layer.

**Request:**
```json
{
  "query": "what are the best guns for self defense",
  "top_k": 5
}
```

**Response (cache miss):**
```json
{
  "query": "what are the best guns for self defense",
  "cache_hit": false,
  "matched_query": null,
  "similarity_score": null,
  "result": "[1] category=talk.politics.guns | score=0.7153\n...",
  "dominant_cluster": 9
}
```

**Response (cache hit):**
```json
{
  "query": "what are the best guns for self defense",
  "cache_hit": true,
  "matched_query": "what are the best guns for self defense",
  "similarity_score": 1.0,
  "result": "[1] category=talk.politics.guns | score=0.7153\n...",
  "dominant_cluster": 9
}
```

---

### `GET /cache/stats`

Returns current cache state.

```json
{
  "total_entries": 42,
  "hit_count": 17,
  "miss_count": 25,
  "hit_rate": 0.405
}
```

---

### `DELETE /cache`

Flushes the cache and resets all stats.

```json
{
  "message": "Cache flushed successfully",
  "entries_cleared": 42
}
```

---

## Design Decisions

### Preprocessing
- All USENET routing headers stripped (Path, Message-ID, Date, etc.)
- Subject line prepended to body — strong topical signal
- Quoted reply lines (`>`) removed — previous poster's words, not current author's
- Signatures truncated at `-- ` separator
- Documents with <20 words or >70% quoted lines dropped
- **Retention: 18,426 / 19,997 (92%)**

### Embedding Model
`all-MiniLM-L6-v2` — 384-dim, fast on CPU, strong semantic quality for short-to-medium English text. Consistently top-tier for this corpus in the literature.

### Clustering
- **UMAP** reduces 384-dim → 50-dim before clustering (preserves topology, removes curse of dimensionality)
- **Fuzzy C-Means** gives probability distributions, not hard labels
- **k=11** chosen by silhouette score sweep (k=10..20), best score=0.5298
- Notable merges: comp.sys.mac + comp.sys.ibm → one hardware cluster; alt.atheism + soc.religion.christian + talk.religion.misc → one religion cluster

### Semantic Cache
- Two-level structure: `{cluster_id: [CacheEntry, ...]}`
- Lookup searches only top-2 clusters by membership weight (~6x faster than flat scan at scale)
- **Similarity threshold θ=0.80** (default): ~95% precision, ~30-40% recall
- Threshold is tunable — lower for higher hit rate, higher for higher precision

### Threshold Analysis

| θ | Behavior |
|---|---|
| 0.70 | Aggressive — high hits, risks wrong results |
| 0.80 | **Default** — balanced precision/recall |
| 0.90 | Conservative — very precise, low hit rate |
| 0.95 | Near-useless — only exact rephrasing hits |

---

## Docker

```bash
# Build and run
docker-compose up --build

# Or manually
docker build -t newsgroups-search .
docker run -p 8000:8000 newsgroups-search
```

The container starts uvicorn on port 8000. Pre-built index and models are copied into the image at build time — no re-embedding on startup.

---

## Requirements

- Python 3.10+
- ~2GB disk space (embeddings + models)
- ~4GB RAM (model loading + Qdrant)
- CPU only — no GPU required