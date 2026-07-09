from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lib.molecules.clustering import (
    ClusteringResult,
    cluster_fingerprints,
)
from lib.molecules.fingerprints import (
    DEFAULT_FINGERPRINT_SIZE,
    calculate_fingerprints_dataframe,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

GENERATED_MOLECULES_PATH = (
    PROJECT_ROOT
    / "tests"
    / "fixture"
    / "output"
    / "test001_generated_molecules.csv"
)


def test_cluster_fingerprints():
    molecules = pd.DataFrame(
        {
            "canonical_smiles": [
                "CCO",
                "CCCO",
                "c1ccccc1",
                "Cc1ccccc1",
            ]
        }
    )

    fingerprint_result = (
        calculate_fingerprints_dataframe(
            molecules
        )
    )

    result = cluster_fingerprints(
        molecules=fingerprint_result.dataframe,
        fingerprint_matrix=fingerprint_result.matrix,
        n_clusters=2,
    )

    assert isinstance(
        result,
        ClusteringResult,
    )

    assert len(result.dataframe) == 4

    assert result.labels.shape == (
        4,
    )

    assert result.cluster_centers.shape == (
        2,
        DEFAULT_FINGERPRINT_SIZE,
    )

    assert set(
        result.dataframe["cluster_id"]
    ) == {
        0,
        1,
    }

    assert (
        result.dataframe[
            "distance_to_centroid"
        ]
        >= 0
    ).all()

    assert result.inertia >= 0


def test_cluster_fingerprints_preserves_original_columns():
    molecules = pd.DataFrame(
        {
            "scaffold_id": [0, 1, 2, 3],
            "r_group_id": [4, 5, 6, 7],
            "canonical_smiles": [
                "CCO",
                "CCCO",
                "c1ccccc1",
                "Cc1ccccc1",
            ],
            "mol_weight": [
                46.069,
                60.096,
                78.114,
                92.141,
            ],
        }
    )

    fingerprint_result = (
        calculate_fingerprints_dataframe(
            molecules
        )
    )

    result = cluster_fingerprints(
        molecules=fingerprint_result.dataframe,
        fingerprint_matrix=fingerprint_result.matrix,
        n_clusters=2,
    )

    output = result.dataframe

    assert output["scaffold_id"].tolist() == [
        0,
        1,
        2,
        3,
    ]

    assert output["r_group_id"].tolist() == [
        4,
        5,
        6,
        7,
    ]

    assert "fingerprint" in output.columns
    assert "fingerprint_on_bits" in output.columns
    assert "cluster_id" in output.columns
    assert "distance_to_centroid" in output.columns


def test_cluster_fingerprints_is_deterministic():
    molecules = pd.DataFrame(
        {
            "canonical_smiles": [
                "CCO",
                "CCCO",
                "CCCCO",
                "c1ccccc1",
                "Cc1ccccc1",
                "CCc1ccccc1",
            ]
        }
    )

    fingerprint_result = (
        calculate_fingerprints_dataframe(
            molecules
        )
    )

    first_result = cluster_fingerprints(
        molecules=fingerprint_result.dataframe,
        fingerprint_matrix=fingerprint_result.matrix,
        n_clusters=2,
        random_state=42,
    )

    second_result = cluster_fingerprints(
        molecules=fingerprint_result.dataframe,
        fingerprint_matrix=fingerprint_result.matrix,
        n_clusters=2,
        random_state=42,
    )

    np.testing.assert_array_equal(
        first_result.labels,
        second_result.labels,
    )

    np.testing.assert_allclose(
        first_result.cluster_centers,
        second_result.cluster_centers,
    )

    assert first_result.inertia == pytest.approx(
        second_result.inertia
    )


def test_duplicate_fingerprints_receive_same_cluster():
    molecules = pd.DataFrame(
        {
            "canonical_smiles": [
                "CCO",
                "OCC",
                "c1ccccc1",
                "C1=CC=CC=C1",
            ]
        }
    )

    fingerprint_result = (
        calculate_fingerprints_dataframe(
            molecules
        )
    )

    result = cluster_fingerprints(
        molecules=fingerprint_result.dataframe,
        fingerprint_matrix=fingerprint_result.matrix,
        n_clusters=2,
    )

    assert (
        result.dataframe.loc[0, "cluster_id"]
        == result.dataframe.loc[1, "cluster_id"]
    )

    assert (
        result.dataframe.loc[2, "cluster_id"]
        == result.dataframe.loc[3, "cluster_id"]
    )


def test_cluster_fingerprints_rejects_row_count_mismatch():
    molecules = pd.DataFrame(
        {
            "canonical_smiles": [
                "CCO",
                "CCCO",
            ]
        }
    )

    fingerprint_matrix = np.zeros(
        (
            1,
            DEFAULT_FINGERPRINT_SIZE,
        ),
        dtype=np.uint8,
    )

    with pytest.raises(
        ValueError,
        match=(
            "Fingerprint matrix row count must match"
        ),
    ):
        cluster_fingerprints(
            molecules=molecules,
            fingerprint_matrix=fingerprint_matrix,
            n_clusters=2,
        )


def test_cluster_fingerprints_rejects_one_dimensional_matrix():
    molecules = pd.DataFrame(
        {
            "canonical_smiles": [
                "CCO",
                "CCCO",
            ]
        }
    )

    fingerprint_matrix = np.zeros(
        DEFAULT_FINGERPRINT_SIZE,
        dtype=np.uint8,
    )

    with pytest.raises(
        ValueError,
        match=(
            "Fingerprint matrix must be "
            "two-dimensional"
        ),
    ):
        cluster_fingerprints(
            molecules=molecules,
            fingerprint_matrix=fingerprint_matrix,
            n_clusters=2,
        )


def test_cluster_fingerprints_rejects_too_many_clusters():
    molecules = pd.DataFrame(
        {
            "canonical_smiles": [
                "CCO",
                "CCCO",
            ]
        }
    )

    fingerprint_result = (
        calculate_fingerprints_dataframe(
            molecules
        )
    )

    with pytest.raises(
        ValueError,
        match=(
            "Number of clusters must not exceed "
            "the number of molecules"
        ),
    ):
        cluster_fingerprints(
            molecules=fingerprint_result.dataframe,
            fingerprint_matrix=fingerprint_result.matrix,
            n_clusters=3,
        )


def test_cluster_fingerprints_rejects_more_clusters_than_unique_fingerprints():
    molecules = pd.DataFrame(
        {
            "canonical_smiles": [
                "CCO",
                "OCC",
                "CCO",
            ]
        }
    )

    fingerprint_result = (
        calculate_fingerprints_dataframe(
            molecules
        )
    )

    with pytest.raises(
        ValueError,
        match=(
            "Number of clusters must not exceed "
            "the number of unique fingerprints"
        ),
    ):
        cluster_fingerprints(
            molecules=fingerprint_result.dataframe,
            fingerprint_matrix=fingerprint_result.matrix,
            n_clusters=2,
        )


def test_cluster_fingerprints_rejects_invalid_cluster_count():
    molecules = pd.DataFrame(
        {
            "canonical_smiles": [
                "CCO",
                "CCCO",
            ]
        }
    )

    fingerprint_result = (
        calculate_fingerprints_dataframe(
            molecules
        )
    )

    with pytest.raises(
        ValueError,
        match=(
            "Number of clusters must be at least 2"
        ),
    ):
        cluster_fingerprints(
            molecules=fingerprint_result.dataframe,
            fingerprint_matrix=fingerprint_result.matrix,
            n_clusters=1,
        )


def test_cluster_fingerprints_rejects_empty_dataframe():
    molecules = pd.DataFrame(
        columns=[
            "canonical_smiles",
        ]
    )

    fingerprint_matrix = np.empty(
        (
            0,
            DEFAULT_FINGERPRINT_SIZE,
        ),
        dtype=np.uint8,
    )

    with pytest.raises(
        ValueError,
        match=(
            "Molecules DataFrame must not be empty"
        ),
    ):
        cluster_fingerprints(
            molecules=molecules,
            fingerprint_matrix=fingerprint_matrix,
            n_clusters=2,
        )


def test_cluster_generated_fixture():
    molecules = pd.read_csv(
        GENERATED_MOLECULES_PATH
    )

    fingerprint_result = (
        calculate_fingerprints_dataframe(
            molecules=molecules,
            smiles_column="generated_smiles",
        )
    )

    result = cluster_fingerprints(
        molecules=fingerprint_result.dataframe,
        fingerprint_matrix=fingerprint_result.matrix,
        n_clusters=5,
    )

    assert len(result.dataframe) == 100

    assert result.labels.shape == (
        100,
    )

    assert result.cluster_centers.shape == (
        5,
        DEFAULT_FINGERPRINT_SIZE,
    )

    assert result.dataframe[
        "cluster_id"
    ].nunique() == 5

    assert set(
        result.dataframe["cluster_id"]
    ) == set(
        range(5)
    )

    assert result.dataframe[
        "distance_to_centroid"
    ].notna().all()

    assert (
        result.dataframe[
            "distance_to_centroid"
        ]
        >= 0
    ).all()

    assert result.inertia >= 0
