from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


DEFAULT_N_CLUSTERS = 5
DEFAULT_RANDOM_STATE = 42
DEFAULT_N_INIT = 10


@dataclass(frozen=True)
class ClusteringResult:
    """Store clustered molecules and fitted K-means results."""

    dataframe: pd.DataFrame
    labels: np.ndarray
    cluster_centers: np.ndarray
    inertia: float


def _validate_clustering_input(
    molecules: pd.DataFrame,
    fingerprint_matrix: np.ndarray,
    n_clusters: int,
) -> None:
    """Validate molecule data and fingerprint matrix before clustering."""
    if not isinstance(molecules, pd.DataFrame):
        raise TypeError(
            "Molecules input must be a pandas DataFrame"
        )

    if molecules.empty:
        raise ValueError(
            "Molecules DataFrame must not be empty"
        )

    if not isinstance(fingerprint_matrix, np.ndarray):
        raise TypeError(
            "Fingerprint matrix must be a NumPy array"
        )

    if fingerprint_matrix.ndim != 2:
        raise ValueError(
            "Fingerprint matrix must be two-dimensional"
        )

    if fingerprint_matrix.shape[0] != len(molecules):
        raise ValueError(
            "Fingerprint matrix row count must match "
            "the number of molecule rows: "
            f"matrix_rows={fingerprint_matrix.shape[0]}, "
            f"molecule_rows={len(molecules)}"
        )

    if fingerprint_matrix.shape[1] == 0:
        raise ValueError(
            "Fingerprint matrix must contain at least one feature"
        )

    if not np.issubdtype(
        fingerprint_matrix.dtype,
        np.number,
    ):
        raise TypeError(
            "Fingerprint matrix must contain numeric values"
        )

    if not np.isfinite(
        fingerprint_matrix
    ).all():
        raise ValueError(
            "Fingerprint matrix contains non-finite values"
        )

    if not isinstance(n_clusters, int):
        raise TypeError(
            "Number of clusters must be an integer"
        )

    if n_clusters < 2:
        raise ValueError(
            "Number of clusters must be at least 2"
        )

    if n_clusters > len(molecules):
        raise ValueError(
            "Number of clusters must not exceed "
            "the number of molecules"
        )

    unique_fingerprints = np.unique(
        fingerprint_matrix,
        axis=0,
    )

    if n_clusters > len(unique_fingerprints):
        raise ValueError(
            "Number of clusters must not exceed "
            "the number of unique fingerprints: "
            f"n_clusters={n_clusters}, "
            f"unique_fingerprints={len(unique_fingerprints)}"
        )


def cluster_fingerprints(
    molecules: pd.DataFrame,
    fingerprint_matrix: np.ndarray,
    n_clusters: int = DEFAULT_N_CLUSTERS,
    random_state: int = DEFAULT_RANDOM_STATE,
    n_init: int = DEFAULT_N_INIT,
) -> ClusteringResult:
    """Cluster molecular fingerprints with K-means."""
    _validate_clustering_input(
        molecules=molecules,
        fingerprint_matrix=fingerprint_matrix,
        n_clusters=n_clusters,
    )

    if not isinstance(random_state, int):
        raise TypeError(
            "Random state must be an integer"
        )

    if not isinstance(n_init, int):
        raise TypeError(
            "n_init must be an integer"
        )

    if n_init <= 0:
        raise ValueError(
            "n_init must be greater than zero"
        )

    numeric_matrix = fingerprint_matrix.astype(
        np.float64,
        copy=False,
    )

    model = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=n_init,
    )

    labels = model.fit_predict(
        numeric_matrix
    )

    distances_to_all_centroids = model.transform(
        numeric_matrix
    )

    distance_to_centroid = (
        distances_to_all_centroids[
            np.arange(len(labels)),
            labels,
        ]
    )

    result = molecules.copy()

    result["cluster_id"] = labels.astype(
        np.int64
    )

    result["distance_to_centroid"] = np.round(
        distance_to_centroid,
        6,
    )

    return ClusteringResult(
        dataframe=result,
        labels=labels.astype(
            np.int64,
            copy=False,
        ),
        cluster_centers=model.cluster_centers_,
        inertia=float(model.inertia_),
    )
