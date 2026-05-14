"""
evaluate.py - End-to-end evaluation of the Newsgroups Semantic Search system

Measures four things:
  1. Search quality    : Precision@1/3/5 and MRR on queries with known categories
  2. Cache paraphrases : Hit rate on semantically equivalent query pairs
  3. Cache dissimilar  : False positive rate on unrelated query pairs
  4. Threshold sweep   : Precision / recall tradeoff across θ = 0.70..0.95

Prerequisites (run in order before this script):
    python scripts/preprocess.py
    python scripts/build_index.py
    python scripts/train_clusters.py

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --top-k 10 --threshold 0.85
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ── Test data ──────────────────────────────────────────────────────────────────

# (query_text, expected_category_substring)
TEST_QUERIES: List[Tuple[str, str]] = [
    ("Does God exist? Arguments against religious belief and atheism", "alt.atheism"),
    ("Christian church doctrine scripture faith prayer", "soc.religion.christian"),
    ("OpenGL 3D rendering techniques graphics programming tutorial", "comp.graphics"),
    ("IBM PC BIOS hardware configuration disk controller problem", "comp.sys.ibm.pc.hardware"),
    ("Mac hardware memory upgrade SIMM DIMM slot", "comp.sys.mac.hardware"),
    ("X Window System server display protocol configuration", "comp.windows.x"),
    ("Windows operating system setup configuration driver", "comp.os.ms-windows.misc"),
    ("RSA public key cryptography encryption algorithm", "sci.crypt"),
    ("heart disease symptoms treatment medication cardiology", "sci.med"),
    ("NASA space shuttle mission orbit launch program", "sci.space"),
    ("electronic circuit resistor capacitor transistor design", "sci.electronics"),
    ("baseball season batting average pitcher ERA statistics", "rec.sport.baseball"),
    ("NHL hockey playoff standings goals saves season", "rec.sport.hockey"),
    ("motorcycle engine repair carburetor maintenance guide", "rec.motorcycles"),
    ("car brake pad oil change transmission service repair", "rec.autos"),
    ("gun control second amendment NRA firearms debate", "talk.politics.guns"),
    ("Israel Palestine Middle East conflict peace talks", "talk.politics.mideast"),
    ("government healthcare policy budget legislation debate", "talk.politics.misc"),
    ("computer equipment for sale asking price offer", "misc.forsale"),
]

# Semantically equivalent pairs — cache should hit
PARAPHRASE_PAIRS: List[Tuple[str, str]] = [
    ("Arguments against the existence of God",
     "Why do atheists not believe in religion?"),
    ("IBM compatible PC hardware problems",
     "Troubleshooting IBM PC hardware issues"),
    ("NASA space exploration rocket mission",
     "Space shuttle orbital flight program"),
    ("gun control second amendment debate",
     "Firearms regulation constitutional rights"),
    ("RSA cryptography and encryption security",
     "Public key encryption algorithm security"),
    ("NHL hockey game season standings",
     "Ice hockey league playoff results"),
    ("motorcycle maintenance and repair guide",
     "Bike engine service upkeep manual"),
    ("heart disease medical treatment therapy",
     "Cardiovascular condition medication treatment"),
]

# Unrelated pairs — cache should NOT hit
DISSIMILAR_PAIRS: List[Tuple[str, str]] = [
    ("baseball batting statistics season",    "NASA space shuttle launch orbit"),
    ("motorcycle engine carburetor repair",   "RSA encryption algorithm security"),
    ("God existence religious debate",        "IBM PC hardware disk configuration"),
    ("heart disease treatment symptoms",      "gun control second amendment"),
    ("NHL hockey playoff standings",          "Mac hardware RAM memory upgrade"),
]


# ── Pipeline loader ────────────────────────────────────────────────────────────

def load_pipeline() -> dict:
    """Load all pipeline components. Exits with a clear message if not built."""
    from app.embedder import get_model
    from app.cluster import load_models as load_cluster_models
    from app.search import get_collection_info

    log.info("Loading embedding model...")
    get_model()

    log.info("Loading cluster models (UMAP + FCM)...")
    load_cluster_models()

    log.info("Connecting to Qdrant index...")
    info = get_collection_info()
    log.info(f"Qdrant ready: {info['points_count']} documents indexed")

    return info


# ── Shared helper ──────────────────────────────────────────────────────────────

def _embed_and_cluster(query: str) -> Tuple[np.ndarray, np.ndarray, List[int], int, float]:
    """
    Embed a query and compute cluster info.
    Returns (embedding, memberships, top_clusters, dominant_cluster, embed_time_s).
    """
    from app.embedder import embed_text
    from app.cluster import get_soft_memberships, get_top_clusters, get_dominant_cluster

    t0 = time.perf_counter()
    embedding = embed_text(query)
    embed_time = time.perf_counter() - t0

    memberships = get_soft_memberships(embedding)
    top_clusters = get_top_clusters(memberships, top_k=2)
    dominant_cluster = get_dominant_cluster(memberships)

    return embedding, memberships, top_clusters, dominant_cluster, embed_time


# ── Section 1: Search quality ──────────────────────────────────────────────────

def evaluate_search_quality(top_k: int = 5) -> Dict:
    """
    For each TEST_QUERY, search Qdrant and check if the expected category
    appears in the top-k results.

    Metrics: Precision@1, @3, @5 and MRR.
    """
    from app.embedder import embed_text
    from app.search import get_client

    client = get_client()

    log.info("")
    log.info("=" * 62)
    log.info(f"  SEARCH QUALITY  (top_k={top_k}, n={len(TEST_QUERIES)} queries)")
    log.info("=" * 62)

    hits_at = {1: 0, 3: 0, 5: 0}
    reciprocal_ranks = []

    for query, expected_cat in TEST_QUERIES:
        embedding = embed_text(query)

        points = client.query_points(
            collection_name="newsgroups",
            query=embedding.tolist(),
            limit=top_k,
            with_payload=True,
        ).points

        categories = [p.payload.get("category", "") for p in points]

        # First rank where the expected category appears
        found_at = next(
            (rank for rank, cat in enumerate(categories, start=1)
             if expected_cat in cat or cat in expected_cat),
            None,
        )

        reciprocal_ranks.append(1.0 / found_at if found_at else 0.0)
        for k_val in [1, 3, 5]:
            if found_at is not None and found_at <= k_val:
                hits_at[k_val] += 1

        status = f"rank={found_at}" if found_at else "MISS  "
        log.info(f"  [{status}]  {expected_cat:<36}  '{query[:45]}'")

    n = len(TEST_QUERIES)
    results = {
        "n_queries":      n,
        "precision_at_1": hits_at[1] / n,
        "precision_at_3": hits_at[3] / n,
        "precision_at_5": hits_at[5] / n,
        "mrr":            float(np.mean(reciprocal_ranks)),
    }

    log.info("")
    log.info(f"  Precision@1  {results['precision_at_1']:>6.1%}")
    log.info(f"  Precision@3  {results['precision_at_3']:>6.1%}")
    log.info(f"  Precision@5  {results['precision_at_5']:>6.1%}")
    log.info(f"  MRR          {results['mrr']:>6.4f}")

    return results


# ── Section 2: Cache paraphrase hit rate ───────────────────────────────────────

def evaluate_cache_paraphrases(threshold: float) -> Dict:
    """
    For each paraphrase pair (q1, q2):
      - Embed both and compute cluster info independently
      - Store q1 in a fresh cache
      - Check if q2 hits the cache (true positive)

    Also records raw cosine similarity and whether cluster routing caused a miss.
    """
    from app.cache import SemanticCache

    log.info("")
    log.info("=" * 62)
    log.info(f"  CACHE PARAPHRASE TEST  (θ={threshold}, n={len(PARAPHRASE_PAIRS)})")
    log.info("=" * 62)

    hits = 0
    cluster_misses = 0

    for q1, q2 in PARAPHRASE_PAIRS:
        emb1, mem1, _, dom1, _ = _embed_and_cluster(q1)
        emb2, _,    top2, _, _ = _embed_and_cluster(q2)

        cache = SemanticCache(threshold=threshold)
        cache.store(q1, emb1, "result", dom1, mem1)

        raw_sim = float(np.dot(emb1, emb2))
        result  = cache.lookup(emb2, top2)
        hit     = result is not None

        if hit:
            hits += 1
            _, score = result
            log.info(f"  HIT  score={score:.4f}  sim={raw_sim:.4f}  '{q1[:35]}' → '{q2[:35]}'")
        else:
            # Distinguish low similarity vs cluster routing mismatch
            if raw_sim >= threshold and dom1 not in top2:
                cluster_misses += 1
                log.info(f"  MISS (cluster mismatch, sim={raw_sim:.4f})  "
                         f"stored_cluster={dom1}, lookup_clusters={top2}")
            else:
                log.info(f"  MISS sim={raw_sim:.4f} < θ={threshold}  '{q1[:35]}' → '{q2[:35]}'")

    hit_rate = hits / len(PARAPHRASE_PAIRS)
    log.info(f"\n  Paraphrase hit rate:     {hits}/{len(PARAPHRASE_PAIRS)} = {hit_rate:.1%}")
    log.info(f"  Cluster-routing misses:  {cluster_misses}")

    return {
        "threshold":           threshold,
        "paraphrase_hit_rate": hit_rate,
        "cluster_misses":      cluster_misses,
    }


# ── Section 3: Cache false positive rate ──────────────────────────────────────

def evaluate_cache_dissimilar(threshold: float) -> Dict:
    """
    For each dissimilar pair (q1, q2):
      - Store q1 in a fresh cache
      - Check if q2 (unrelated topic) incorrectly hits the cache (false positive)
    """
    from app.cache import SemanticCache

    log.info("")
    log.info("=" * 62)
    log.info(f"  CACHE FALSE POSITIVE TEST  (θ={threshold}, n={len(DISSIMILAR_PAIRS)})")
    log.info("=" * 62)

    false_positives = 0

    for q1, q2 in DISSIMILAR_PAIRS:
        emb1, mem1, _, dom1, _ = _embed_and_cluster(q1)
        emb2, _,    top2, _, _ = _embed_and_cluster(q2)

        cache = SemanticCache(threshold=threshold)
        cache.store(q1, emb1, "result", dom1, mem1)

        raw_sim = float(np.dot(emb1, emb2))
        result  = cache.lookup(emb2, top2)
        hit     = result is not None

        if hit:
            false_positives += 1
            _, score = result
            log.info(f"  FALSE POS score={score:.4f}  "
                     f"'{q1[:35]}' → '{q2[:35]}'")
        else:
            log.info(f"  OK  sim={raw_sim:.4f}  '{q1[:35]}' → '{q2[:35]}'")

    fpr = false_positives / len(DISSIMILAR_PAIRS)
    log.info(f"\n  False positive rate: {false_positives}/{len(DISSIMILAR_PAIRS)} = {fpr:.1%}")

    return {"threshold": threshold, "false_positive_rate": fpr}


# ── Section 4: Threshold sweep ─────────────────────────────────────────────────

def threshold_sweep() -> List[Dict]:
    """
    For each threshold in [0.70, 0.75, 0.80, 0.85, 0.90, 0.95], compute:
      - True positive rate  (paraphrase pairs that hit the cache)
      - False positive rate (dissimilar pairs that incorrectly hit the cache)
      - F1 treating TPR as recall and precision = TPR / (TPR + FPR)

    Pre-computes all embeddings once to keep the sweep fast.
    """
    from app.cache import SemanticCache

    log.info("")
    log.info("=" * 62)
    log.info("  THRESHOLD SWEEP")
    log.info("=" * 62)

    log.info("  Pre-computing embeddings...")

    para_items = []
    for q1, q2 in PARAPHRASE_PAIRS:
        emb1, mem1, _, dom1, _ = _embed_and_cluster(q1)
        emb2, _,    top2, _, _ = _embed_and_cluster(q2)
        para_items.append((emb1, mem1, dom1, emb2, top2))

    diss_items = []
    for q1, q2 in DISSIMILAR_PAIRS:
        emb1, mem1, _, dom1, _ = _embed_and_cluster(q1)
        emb2, _,    top2, _, _ = _embed_and_cluster(q2)
        diss_items.append((emb1, mem1, dom1, emb2, top2))

    thresholds = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    results = []

    log.info(f"\n  {'θ':>5} | {'TPR (paraphrase)':>16} | "
             f"{'FPR (dissimilar)':>16} | {'F1':>6}")
    log.info("  " + "-" * 52)

    for θ in thresholds:
        tpr_hits = sum(
            1 for emb1, mem1, dom1, emb2, top2 in para_items
            if SemanticCache(threshold=θ)
               .store("q", emb1, "r", dom1, mem1)
               and False  # trick: use walrus-free approach below
        )
        # Cleaner: just count hits explicitly
        tpr_hits = 0
        for emb1, mem1, dom1, emb2, top2 in para_items:
            c = SemanticCache(threshold=θ)
            c.store("q", emb1, "r", dom1, mem1)
            if c.lookup(emb2, top2) is not None:
                tpr_hits += 1

        fpr_hits = 0
        for emb1, mem1, dom1, emb2, top2 in diss_items:
            c = SemanticCache(threshold=θ)
            c.store("q", emb1, "r", dom1, mem1)
            if c.lookup(emb2, top2) is not None:
                fpr_hits += 1

        tpr = tpr_hits / len(para_items)
        fpr = fpr_hits / len(diss_items)

        precision = tpr / (tpr + fpr) if (tpr + fpr) > 0 else 1.0
        f1 = 2 * precision * tpr / (precision + tpr) if (precision + tpr) > 0 else 0.0

        log.info(f"  {θ:.2f}  |  {tpr:>14.1%}   |  {fpr:>14.1%}   | {f1:.4f}")
        results.append({
            "threshold":          θ,
            "true_positive_rate": tpr,
            "false_positive_rate": fpr,
            "f1":                 f1,
        })

    best = max(results, key=lambda x: x["f1"])
    log.info(f"\n  Best F1={best['f1']:.4f} at θ={best['threshold']}")

    return results


# ── Section 5: Timing ──────────────────────────────────────────────────────────

def evaluate_timing() -> Dict:
    """
    Measure average and p95 latency for:
      - Embedding (sentence-transformer encode)
      - Cache lookup (dot-product scan over stored entries)
    """
    from app.cache import SemanticCache

    n_warm    = 3    # warm-up runs (excluded)
    n_measure = len(TEST_QUERIES)

    log.info("")
    log.info("=" * 62)
    log.info(f"  TIMING  ({n_measure} queries)")
    log.info("=" * 62)

    # Pre-populate cache so lookup has something to scan
    cache = SemanticCache(threshold=0.80)
    embed_times = []

    for i, (query, _) in enumerate(TEST_QUERIES):
        emb, mem, _, dom, etime = _embed_and_cluster(query)
        if i >= n_warm:
            embed_times.append(etime)
        cache.store(query, emb, "result", dom, mem)

    # Now measure cache lookup times
    lookup_times = []
    for query, _ in TEST_QUERIES[n_warm:]:
        emb, _, top_clusters, _, _ = _embed_and_cluster(query)
        t0 = time.perf_counter()
        cache.lookup(emb, top_clusters)
        lookup_times.append(time.perf_counter() - t0)

    results = {
        "embed_avg_ms":    float(np.mean(embed_times))    * 1000,
        "embed_p95_ms":    float(np.percentile(embed_times, 95)) * 1000,
        "lookup_avg_ms":   float(np.mean(lookup_times))   * 1000,
        "lookup_p95_ms":   float(np.percentile(lookup_times, 95)) * 1000,
        "cache_entries":   cache.total_entries,
    }

    log.info(f"  Embedding   avg={results['embed_avg_ms']:.1f}ms  "
             f"p95={results['embed_p95_ms']:.1f}ms")
    log.info(f"  Cache lookup avg={results['lookup_avg_ms']:.3f}ms  "
             f"p95={results['lookup_p95_ms']:.3f}ms  "
             f"(over {results['cache_entries']} entries)")

    return results


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> Dict:
    parser = argparse.ArgumentParser(
        description="Evaluate Newsgroups Semantic Search quality and cache behavior"
    )
    parser.add_argument("--top-k",     type=int,   default=5,    help="Qdrant top-k (default 5)")
    parser.add_argument("--threshold", type=float, default=0.80, help="Cache threshold (default 0.80)")
    args = parser.parse_args()

    try:
        load_pipeline()
    except FileNotFoundError as exc:
        log.error(f"Pipeline not built yet: {exc}")
        sys.exit(1)

    all_results = {}
    all_results["search_quality"]    = evaluate_search_quality(top_k=args.top_k)
    all_results["cache_paraphrases"] = evaluate_cache_paraphrases(threshold=args.threshold)
    all_results["cache_dissimilar"]  = evaluate_cache_dissimilar(threshold=args.threshold)
    all_results["threshold_sweep"]   = threshold_sweep()
    all_results["timing"]            = evaluate_timing()

    # ── Final summary ──────────────────────────────────────────────────────────
    sq  = all_results["search_quality"]
    cp  = all_results["cache_paraphrases"]
    cd  = all_results["cache_dissimilar"]
    ts  = all_results["threshold_sweep"]
    tm  = all_results["timing"]
    best = max(ts, key=lambda x: x["f1"])

    log.info("")
    log.info("=" * 62)
    log.info("  SUMMARY")
    log.info("=" * 62)
    log.info(f"  Search  P@1={sq['precision_at_1']:.1%}  "
             f"P@5={sq['precision_at_5']:.1%}  "
             f"MRR={sq['mrr']:.4f}")
    log.info(f"  Cache   paraphrase_hit_rate={cp['paraphrase_hit_rate']:.1%}  "
             f"false_positive_rate={cd['false_positive_rate']:.1%}  "
             f"(θ={args.threshold})")
    log.info(f"  Best θ  {best['threshold']} → F1={best['f1']:.4f}  "
             f"(TPR={best['true_positive_rate']:.1%}, "
             f"FPR={best['false_positive_rate']:.1%})")
    log.info(f"  Timing  embed={tm['embed_avg_ms']:.1f}ms  "
             f"lookup={tm['lookup_avg_ms']:.3f}ms")

    return all_results


if __name__ == "__main__":
    main()
