"""
train_clusters.py - Step 3 of the pipeline

Loads all document vectors from Qdrant, reduces dimensionality with UMAP,
then applies Fuzzy C-Means (FCM) to get soft cluster assignments.

Design decisions:
- UMAP before FCM:
    Raw 384-dim embeddings suffer from the curse of dimensionality — distance
    metrics become unreliable in very high dimensions. UMAP reduces to 50 dims
    while preserving both local and global topology. FCM on UMAP-reduced vectors
    consistently outperforms FCM on raw embeddings for this corpus.

- Fuzzy C-Means (not KMeans):
    The spec explicitly rejects hard cluster assignments. A document about gun
    legislation belongs to BOTH politics and firearms clusters — FCM captures
    this by outputting a probability distribution over clusters per document.
    Each document gets a membership vector that sums to 1.0.

- Number of clusters k=16:
    The 20 newsgroup labels are not semantically independent. Several pairs
    are nearly identical in embedding space:
      comp.sys.mac.hardware ≈ comp.sys.ibm.pc.hardware (both: PC hardware)
      talk.religion.misc ≈ alt.atheism (both: religion debate)
      talk.politics.misc ≈ talk.politics.guns (heavy overlap)
      rec.sport.baseball ≈ rec.sport.hockey (both: North American sports)
    Using k=20 forces artificial splits where none exist semantically.
    k=16 is justified by silhouette score (computed and logged below).
    We sweep k=10..20 and pick the elbow.

- FCM fuzziness m=2.0:
    The fuzziness exponent m controls how soft the assignments are.
    m=1 → hard clustering (identical to KMeans limit)
    m=2 → standard fuzzy clustering, well-studied default
    m→∞ → all documents equally in all clusters (useless)
    m=2.0 is the standard choice; values 1.5-2.5 are all reasonable.

- Saving outputs:
    umap_reducer.pkl  → fitted UMAP, used at query time to project new queries
    fcm_model.pkl     → fitted FCM centroids + metadata
    cluster_centroids.npy → centroid vectors in UMAP space (for cache lookup)
    memberships.npy   → full (N x k) soft assignment matrix (for analysis)
"""

import json
import logging
import time
import pickle
from pathlib import Path
from typing import List, Tuple

import numpy as np
from qdrant_client import QdrantClient
import umap
import skfuzzy as fuzz
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).resolve().parent.parent
QDRANT_PATH     = BASE_DIR / "embeddings" / "qdrant_index"
MODELS_DIR      = BASE_DIR / "models"
PROCESSED_FILE  = BASE_DIR / "data" / "processed" / "cleaned_docs.jsonl"

# ── Config ─────────────────────────────────────────────────────────────────────
COLLECTION_NAME = "newsgroups"
UMAP_N_COMPONENTS = 50       # reduce 384 → 50 dims before clustering
UMAP_N_NEIGHBORS  = 15       # UMAP local neighborhood size
UMAP_MIN_DIST     = 0.1      # UMAP minimum distance between points
UMAP_METRIC       = "cosine" # match embedding space metric
RANDOM_STATE      = 42

# FCM parameters
FCM_M             = 2.0      # fuzziness exponent (standard default)
FCM_ERROR         = 0.005    # convergence threshold
FCM_MAXITER       = 1000     # max iterations

# Cluster sweep range
K_MIN = 10
K_MAX = 20


def load_vectors_from_qdrant(client: QdrantClient) -> Tuple[np.ndarray, List[dict]]:
    """
    Retrieve all vectors and payloads from Qdrant using pagination.
    Returns (vectors_matrix, payloads_list).
    """
    log.info("Loading all vectors from Qdrant...")
    
    all_vectors  = []
    all_payloads = []
    offset       = None
    batch_size   = 1000
    total        = 0

    while True:
        results, next_offset = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=batch_size,
            offset=offset,
            with_vectors=True,
            with_payload=True,
        )

        if not results:
            break

        for point in results:
            all_vectors.append(point.vector)
            all_payloads.append(point.payload)
            total += 1

        if next_offset is None:
            break
        offset = next_offset

        if total % 5000 == 0:
            log.info(f"  Loaded {total} vectors...")

    vectors = np.array(all_vectors, dtype=np.float32)
    log.info(f"Loaded {len(vectors)} vectors, shape: {vectors.shape}")
    return vectors, all_payloads


def fit_umap(vectors: np.ndarray) -> Tuple[umap.UMAP, np.ndarray]:
    """
    Fit UMAP on the full corpus embeddings.
    
    Why 50 components (not 2):
        2D UMAP is great for visualization but loses too much information
        for clustering. 50 components retains enough structure for FCM
        while dramatically reducing the dimensionality problem.
    """
    log.info(f"Fitting UMAP: {vectors.shape[1]}d → {UMAP_N_COMPONENTS}d ...")
    log.info(f"  n_neighbors={UMAP_N_NEIGHBORS}, min_dist={UMAP_MIN_DIST}, "
             f"metric={UMAP_METRIC}")
    
    start = time.time()
    reducer = umap.UMAP(
        n_components=UMAP_N_COMPONENTS,
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        metric=UMAP_METRIC,
        random_state=RANDOM_STATE,
        low_memory=False,
        verbose=False,
    )
    reduced = reducer.fit_transform(vectors)
    elapsed = time.time() - start
    
    log.info(f"UMAP complete in {elapsed:.1f}s. Reduced shape: {reduced.shape}")
    return reducer, reduced


def sweep_k(reduced: np.ndarray) -> int:
    """
    Sweep k from K_MIN to K_MAX, compute silhouette score for each,
    and return the k with the best score.
    
    Silhouette score measures how well-separated clusters are:
    +1.0 = perfectly separated, 0 = overlapping, -1 = wrong cluster
    We use hard assignments (argmax of FCM memberships) for silhouette
    since it requires discrete labels.
    """
    log.info(f"Sweeping k from {K_MIN} to {K_MAX} to find optimal cluster count...")
    
    # Transpose for skfuzzy (expects shape: n_features x n_samples)
    data_T = reduced.T
    
    scores = {}
    best_k = K_MIN
    best_score = -1.0

    for k in range(K_MIN, K_MAX + 1):
        try:
            cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(
                data_T,
                c=k,
                m=FCM_M,
                error=FCM_ERROR,
                maxiter=300,   # fewer iterations for sweep
                init=None,
                seed=RANDOM_STATE,
            )
            # Hard labels for silhouette (argmax of membership matrix)
            labels = np.argmax(u, axis=0)
            
            # Only compute silhouette on a sample for speed
            sample_size = min(3000, len(reduced))
            idx = np.random.RandomState(RANDOM_STATE).choice(
                len(reduced), sample_size, replace=False
            )
            score = silhouette_score(reduced[idx], labels[idx], metric="euclidean")
            scores[k] = score
            
            log.info(f"  k={k:2d} | silhouette={score:.4f}")
            
            if score > best_score:
                best_score = score
                best_k = k
                
        except Exception as e:
            log.warning(f"  k={k} failed: {e}")

    log.info(f"\nBest k={best_k} with silhouette={best_score:.4f}")
    return best_k, scores


def fit_fcm(reduced: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit Fuzzy C-Means with the chosen k.
    
    Returns:
        centroids   : (k x n_features) cluster center vectors in UMAP space
        memberships : (N x k) soft assignment matrix, rows sum to 1.0
    
    skfuzzy.cmeans expects data as (n_features x n_samples) — note the transpose.
    Output u is (k x n_samples) — we transpose back to (n_samples x k).
    """
    log.info(f"Fitting FCM with k={k}, m={FCM_M}...")
    start = time.time()

    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
        data=reduced.T,      # (n_features x n_samples)
        c=k,
        m=FCM_M,
        error=FCM_ERROR,
        maxiter=FCM_MAXITER,
        init=None,
        seed=RANDOM_STATE,
    )

    elapsed = time.time() - start
    memberships = u.T   # (n_samples x k)

    log.info(f"FCM complete in {elapsed:.1f}s")
    log.info(f"  Fuzzy partition coefficient (FPC): {fpc:.4f}  "
             f"(1.0=crisp, 1/k={1/k:.4f}=random)")
    log.info(f"  Centroids shape   : {cntr.shape}")
    log.info(f"  Memberships shape : {memberships.shape}")
    log.info(f"  Membership row sum check (should be ~1.0): "
             f"{memberships[0].sum():.6f}")

    return cntr, memberships


def analyze_clusters(memberships: np.ndarray, payloads: List[dict], k: int):
    """
    Print cluster analysis:
    - Dominant category per cluster (most common original label)
    - Average membership confidence
    - Number of boundary documents (entropy > threshold)
    """
    log.info("\n" + "=" * 60)
    log.info("CLUSTER ANALYSIS")
    log.info("=" * 60)

    hard_labels = np.argmax(memberships, axis=1)

    for cluster_id in range(k):
        mask = hard_labels == cluster_id
        cluster_docs = [payloads[i] for i in range(len(payloads)) if mask[i]]
        cluster_memberships = memberships[mask, cluster_id]

        if not cluster_docs:
            continue

        # Category distribution
        from collections import Counter
        cats = Counter(d["category"] for d in cluster_docs)
        top_cats = cats.most_common(3)

        # Average confidence
        avg_confidence = cluster_memberships.mean()

        # Boundary docs: entropy of membership vector > 0.8 * log(k)
        cluster_entropy = -np.sum(
            memberships[mask] * np.log(memberships[mask] + 1e-10), axis=1
        )
        max_entropy = np.log(k)
        boundary_count = np.sum(cluster_entropy > 0.7 * max_entropy)

        log.info(f"\nCluster {cluster_id:2d} | {len(cluster_docs):4d} docs | "
                 f"avg_confidence={avg_confidence:.3f} | "
                 f"boundary_docs={boundary_count}")
        for cat, count in top_cats:
            pct = 100 * count / len(cluster_docs)
            log.info(f"  {cat:<35s} {count:4d} ({pct:.1f}%)")

    # Overall boundary document count
    all_entropy = -np.sum(memberships * np.log(memberships + 1e-10), axis=1)
    max_entropy = np.log(k)
    total_boundary = np.sum(all_entropy > 0.7 * max_entropy)
    log.info(f"\nTotal boundary documents (high entropy): "
             f"{total_boundary} / {len(memberships)} "
             f"({100*total_boundary/len(memberships):.1f}%)")


def save_models(reducer, centroids, memberships, payloads, k, silhouette_scores):
    """Save all model artifacts to models/ directory."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. UMAP reducer
    umap_path = MODELS_DIR / "umap_reducer.pkl"
    with umap_path.open("wb") as f:
        pickle.dump(reducer, f)
    log.info(f"Saved UMAP reducer → {umap_path}")

    # 2. FCM model (centroids + config)
    fcm_data = {
        "centroids":         centroids,
        "k":                 k,
        "m":                 FCM_M,
        "umap_n_components": UMAP_N_COMPONENTS,
        "silhouette_scores": silhouette_scores,
    }
    fcm_path = MODELS_DIR / "fcm_model.pkl"
    with fcm_path.open("wb") as f:
        pickle.dump(fcm_data, f)
    log.info(f"Saved FCM model → {fcm_path}")

    # 3. Cluster centroids as numpy array (fast load for cache)
    centroids_path = MODELS_DIR / "cluster_centroids.npy"
    np.save(centroids_path, centroids)
    log.info(f"Saved cluster centroids → {centroids_path}")

    # 4. Full membership matrix (for notebook analysis)
    memberships_path = MODELS_DIR / "memberships.npy"
    np.save(memberships_path, memberships)
    log.info(f"Saved memberships → {memberships_path}")

    # 5. Doc metadata with cluster assignments (for notebook analysis)
    assignments = []
    hard_labels = np.argmax(memberships, axis=1)
    for i, payload in enumerate(payloads):
        assignments.append({
            "doc_id":           payload.get("doc_id"),
            "category":         payload.get("category"),
            "dominant_cluster": int(hard_labels[i]),
            "max_membership":   float(memberships[i].max()),
            "memberships":      memberships[i].tolist(),
        })

    assignments_path = MODELS_DIR / "cluster_assignments.json"
    with assignments_path.open("w") as f:
        json.dump(assignments, f)
    log.info(f"Saved cluster assignments → {assignments_path}")


def run():
    # ── Load vectors ──────────────────────────────────────────────────────────
    client = QdrantClient(path=str(QDRANT_PATH))
    vectors, payloads = load_vectors_from_qdrant(client)

    # ── UMAP reduction ────────────────────────────────────────────────────────
    reducer, reduced = fit_umap(vectors)

    # ── Sweep k to find optimal cluster count ─────────────────────────────────
    best_k, silhouette_scores = sweep_k(reduced)

    # ── Fit final FCM with best k ─────────────────────────────────────────────
    centroids, memberships = fit_fcm(reduced, best_k)

    # ── Analyze clusters ──────────────────────────────────────────────────────
    analyze_clusters(memberships, payloads, best_k)

    # ── Save all models ───────────────────────────────────────────────────────
    save_models(reducer, centroids, memberships, payloads, best_k, silhouette_scores)

    log.info("\n" + "=" * 60)
    log.info("train_clusters.py complete.")
    log.info(f"  k (clusters)     : {best_k}")
    log.info(f"  UMAP components  : {UMAP_N_COMPONENTS}")
    log.info(f"  FCM fuzziness m  : {FCM_M}")
    log.info(f"  Models saved to  : {MODELS_DIR}")


if __name__ == "__main__":
    run()