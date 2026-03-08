"""
cluster.py - Soft cluster assignment for new queries at runtime

Loads the saved UMAP reducer and FCM centroids, and computes
soft cluster membership for any new query vector.

Design decisions:
- Load models once at startup (singleton pattern), same reason as embedder.py.
- UMAP transform (not fit_transform): we use the ALREADY FITTED reducer
  from training to project new query vectors into the same 50-dim space.
  This is critical — the FCM centroids live in that specific UMAP space.
- FCM membership formula: given a query point x and centroids C,
  membership of x in cluster i is computed as:
      u_i = 1 / sum_j( (dist(x,C_i) / dist(x,C_j))^(2/(m-1)) )
  This is the standard FCM prediction formula.
  We implement it from scratch (no skfuzzy dependency at query time).
- top_k_clusters: the cache only searches within the top-k clusters
  by membership weight. Default k=2 — covers the primary and secondary
  cluster, giving good recall while limiting search space.
"""

import logging
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

log = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).resolve().parent.parent
MODELS_DIR   = BASE_DIR / "models"

# ── Singleton model state ──────────────────────────────────────────────────────
_umap_reducer  = None
_fcm_centroids = None   # shape: (k, n_umap_components)
_fcm_m         = 2.0    # fuzziness exponent (loaded from fcm_model.pkl)
_k             = None   # number of clusters


def load_models():
    """
    Load UMAP reducer and FCM model from disk.
    Called once at app startup.
    """
    global _umap_reducer, _fcm_centroids, _fcm_m, _k

    umap_path = MODELS_DIR / "umap_reducer.pkl"
    fcm_path  = MODELS_DIR / "fcm_model.pkl"

    if not umap_path.exists():
        raise FileNotFoundError(
            f"UMAP model not found: {umap_path}\n"
            "Run scripts/train_clusters.py first."
        )
    if not fcm_path.exists():
        raise FileNotFoundError(
            f"FCM model not found: {fcm_path}\n"
            "Run scripts/train_clusters.py first."
        )

    with umap_path.open("rb") as f:
        _umap_reducer = pickle.load(f)
    log.info("UMAP reducer loaded")

    with fcm_path.open("rb") as f:
        fcm_data = pickle.load(f)

    _fcm_centroids = fcm_data["centroids"]   # (k x umap_dims)
    _fcm_m         = fcm_data["m"]
    _k             = fcm_data["k"]
    log.info(f"FCM model loaded: k={_k}, m={_fcm_m}")


def _ensure_loaded():
    """Load models if not already loaded."""
    if _umap_reducer is None or _fcm_centroids is None:
        load_models()


def _fcm_predict(umap_vector: np.ndarray) -> np.ndarray:
    """
    Compute FCM soft membership for a single point in UMAP space.

    Uses the standard FCM prediction formula:
        u_i = 1 / sum_j( (d_i / d_j)^(2/(m-1)) )
    where d_i = Euclidean distance from point to centroid i.

    Returns:
        memberships: np.ndarray of shape (k,), sums to 1.0
    """
    exponent = 2.0 / (_fcm_m - 1.0)

    # Euclidean distances to each centroid
    # _fcm_centroids: (k, dims), umap_vector: (dims,)
    diffs     = _fcm_centroids - umap_vector   # (k, dims)
    distances = np.linalg.norm(diffs, axis=1)  # (k,)

    # Handle exact match with a centroid (distance = 0)
    if np.any(distances == 0):
        memberships = np.zeros(_k)
        memberships[distances == 0] = 1.0
        return memberships

    # Standard FCM formula
    memberships = np.zeros(_k)
    for i in range(_k):
        ratio_sum = np.sum((distances[i] / distances) ** exponent)
        memberships[i] = 1.0 / ratio_sum

    # Normalize to ensure sum = 1 (numerical safety)
    memberships /= memberships.sum()
    return memberships


def get_soft_memberships(embedding: np.ndarray) -> np.ndarray:
    """
    Given a raw 384-dim embedding, return soft cluster memberships.

    Pipeline:
        384-dim embedding → UMAP transform → 50-dim → FCM predict → (k,) memberships

    Args:
        embedding: np.ndarray of shape (384,), L2-normalized

    Returns:
        memberships: np.ndarray of shape (k,), sums to 1.0
    """
    _ensure_loaded()

    # UMAP transform: project into the trained 50-dim space
    # reshape to (1, 384) for transform, then squeeze back
    umap_vector = _umap_reducer.transform(embedding.reshape(1, -1))[0]

    # FCM soft membership prediction
    memberships = _fcm_predict(umap_vector)

    return memberships


def get_top_clusters(memberships: np.ndarray, top_k: int = 2) -> List[int]:
    """
    Return indices of the top-k clusters by membership weight.

    Used by the cache to limit search scope — instead of scanning
    the entire cache, we only search within the most relevant clusters.

    Args:
        memberships: np.ndarray of shape (k,)
        top_k: number of top clusters to return

    Returns:
        List of cluster indices sorted by membership weight (descending)
    """
    top_k = min(top_k, len(memberships))
    return list(np.argsort(memberships)[::-1][:top_k])


def get_dominant_cluster(memberships: np.ndarray) -> int:
    """
    Return the index of the cluster with highest membership weight.
    Used for the API response field 'dominant_cluster'.
    """
    return int(np.argmax(memberships))


def get_cluster_count() -> int:
    """Return the number of clusters k."""
    _ensure_loaded()
    return _k