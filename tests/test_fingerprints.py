from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lib.molecules.fingerprints import (
    DEFAULT_FINGERPRINT_SIZE,
    FINGERPRINT_TYPE,
    FingerprintResult,
    calculate_ecfp4_fingerprint,
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


def test_calculate_ecfp4_fingerprint():
    (
        fingerprint,
        fingerprint_on_bits,
        fingerprint_array,
    ) = calculate_ecfp4_fingerprint(
        "CCO"
    )

    assert len(fingerprint) == (
        DEFAULT_FINGERPRINT_SIZE
    )

    assert set(fingerprint).issubset(
        {"0", "1"}
    )

    assert isinstance(
        fingerprint_on_bits,
        str,
    )

    assert fingerprint_array.shape == (
        DEFAULT_FINGERPRINT_SIZE,
    )

    assert fingerprint_array.dtype == (
        np.uint8
    )

    assert set(
        np.unique(fingerprint_array)
    ).issubset(
        {0, 1}
    )

    assert fingerprint_array.sum() > 0


def test_calculate_ecfp4_fingerprint_is_deterministic():
    first_result = (
        calculate_ecfp4_fingerprint(
            "CCO"
        )
    )

    second_result = (
        calculate_ecfp4_fingerprint(
            "CCO"
        )
    )

    assert first_result[0] == second_result[0]
    assert first_result[1] == second_result[1]

    np.testing.assert_array_equal(
        first_result[2],
        second_result[2],
    )


def test_equivalent_smiles_produce_same_fingerprint():
    first_result = (
        calculate_ecfp4_fingerprint(
            "CCO"
        )
    )

    second_result = (
        calculate_ecfp4_fingerprint(
            "OCC"
        )
    )

    assert first_result[0] == second_result[0]

    np.testing.assert_array_equal(
        first_result[2],
        second_result[2],
    )


def test_different_molecules_produce_different_fingerprints():
    ethanol = calculate_ecfp4_fingerprint(
        "CCO"
    )

    benzene = calculate_ecfp4_fingerprint(
        "c1ccccc1"
    )

    assert ethanol[0] != benzene[0]

    assert not np.array_equal(
        ethanol[2],
        benzene[2],
    )


def test_calculate_ecfp4_fingerprint_rejects_blank_smiles():
    with pytest.raises(
        ValueError,
        match="SMILES must not be blank",
    ):
        calculate_ecfp4_fingerprint(
            "   "
        )


def test_calculate_ecfp4_fingerprint_rejects_invalid_smiles():
    with pytest.raises(
        ValueError,
        match="Invalid SMILES",
    ):
        calculate_ecfp4_fingerprint(
            "not_a_smiles"
        )


def test_calculate_ecfp4_fingerprint_rejects_non_string():
    with pytest.raises(
        TypeError,
        match="SMILES must be a string",
    ):
        calculate_ecfp4_fingerprint(
            123
        )


def test_calculate_ecfp4_fingerprint_rejects_invalid_size():
    with pytest.raises(
        ValueError,
        match=(
            "Fingerprint size must be "
            "greater than zero"
        ),
    ):
        calculate_ecfp4_fingerprint(
            smiles="CCO",
            fingerprint_size=0,
        )


def test_calculate_fingerprints_dataframe():
    molecules = pd.DataFrame(
        {
            "scaffold_id": [0, 1],
            "r_group_id": [0, 0],
            "canonical_smiles": [
                "CCO",
                "c1ccccc1",
            ],
        }
    )

    result = calculate_fingerprints_dataframe(
        molecules
    )

    assert isinstance(
        result,
        FingerprintResult,
    )

    assert len(result.dataframe) == 2

    assert result.matrix.shape == (
        2,
        DEFAULT_FINGERPRINT_SIZE,
    )

    assert result.matrix.dtype == np.uint8

    assert (
        result.dataframe[
            "fingerprint_type"
        ]
        == FINGERPRINT_TYPE
    ).all()

    assert (
        result.dataframe[
            "fingerprint"
        ].str.len()
        == DEFAULT_FINGERPRINT_SIZE
    ).all()

    assert (
        result.dataframe[
            "fingerprint_on_bits"
        ].str.len()
        > 0
    ).all()


def test_calculate_fingerprints_dataframe_preserves_columns():
    molecules = pd.DataFrame(
        {
            "scaffold_id": [4],
            "r_group_id": [7],
            "generated_smiles": ["CCO"],
            "canonical_smiles": ["CCO"],
            "mol_weight": [46.069],
        }
    )

    result = calculate_fingerprints_dataframe(
        molecules
    )

    output = result.dataframe

    assert output.loc[0, "scaffold_id"] == 4
    assert output.loc[0, "r_group_id"] == 7

    assert (
        output.loc[
            0,
            "generated_smiles",
        ]
        == "CCO"
    )

    assert output.loc[
        0,
        "mol_weight",
    ] == pytest.approx(
        46.069
    )


def test_duplicate_molecules_have_same_fingerprint():
    molecules = pd.DataFrame(
        {
            "canonical_smiles": [
                "CCO",
                "CCO",
                "c1ccccc1",
            ]
        }
    )

    result = calculate_fingerprints_dataframe(
        molecules
    )

    assert (
        result.dataframe.loc[
            0,
            "fingerprint",
        ]
        == result.dataframe.loc[
            1,
            "fingerprint",
        ]
    )

    np.testing.assert_array_equal(
        result.matrix[0],
        result.matrix[1],
    )

    assert not np.array_equal(
        result.matrix[0],
        result.matrix[2],
    )


def test_equivalent_smiles_are_canonicalized():
    molecules = pd.DataFrame(
        {
            "canonical_smiles": [
                "CCO",
                "OCC",
            ]
        }
    )

    result = calculate_fingerprints_dataframe(
        molecules
    )

    assert (
        result.dataframe[
            "canonical_smiles"
        ].nunique()
        == 1
    )

    np.testing.assert_array_equal(
        result.matrix[0],
        result.matrix[1],
    )


def test_fingerprint_on_bits_matches_matrix():
    molecules = pd.DataFrame(
        {
            "canonical_smiles": [
                "CCO",
            ]
        }
    )

    result = calculate_fingerprints_dataframe(
        molecules
    )

    expected_on_bits = {
        int(value)
        for value in (
            result.dataframe.loc[
                0,
                "fingerprint_on_bits",
            ].split()
        )
    }

    actual_on_bits = set(
        np.flatnonzero(
            result.matrix[0]
        ).tolist()
    )

    assert expected_on_bits == actual_on_bits


def test_calculate_fingerprints_dataframe_rejects_missing_column():
    molecules = pd.DataFrame(
        {
            "generated_smiles": [
                "CCO",
            ]
        }
    )

    with pytest.raises(
        ValueError,
        match=(
            "Molecules DataFrame must contain "
            "the 'canonical_smiles' column"
        ),
    ):
        calculate_fingerprints_dataframe(
            molecules
        )


def test_calculate_fingerprints_dataframe_rejects_empty_dataframe():
    molecules = pd.DataFrame(
        columns=[
            "canonical_smiles",
        ]
    )

    with pytest.raises(
        ValueError,
        match=(
            "Molecules DataFrame must not be empty"
        ),
    ):
        calculate_fingerprints_dataframe(
            molecules
        )


def test_calculate_fingerprints_dataframe_rejects_non_dataframe():
    with pytest.raises(
        TypeError,
        match=(
            "Molecules input must be "
            "a pandas DataFrame"
        ),
    ):
        calculate_fingerprints_dataframe(
            ["CCO"]
        )


def test_calculate_fingerprints_dataframe_reports_failed_row():
    molecules = pd.DataFrame(
        {
            "canonical_smiles": [
                "CCO",
                "not_a_smiles",
            ]
        }
    )

    with pytest.raises(
        ValueError,
        match=(
            "Failed to calculate fingerprint "
            "for row_position=1"
        ),
    ):
        calculate_fingerprints_dataframe(
            molecules
        )


def test_calculate_fingerprints_for_generated_fixture():
    molecules = pd.read_csv(
        GENERATED_MOLECULES_PATH
    )

    result = calculate_fingerprints_dataframe(
        molecules=molecules,
        smiles_column="generated_smiles",
    )

    assert len(result.dataframe) == 100

    assert result.matrix.shape == (
        100,
        DEFAULT_FINGERPRINT_SIZE,
    )

    assert result.matrix.dtype == np.uint8

    assert result.dataframe[
        "fingerprint"
    ].notna().all()

    assert (
        result.dataframe[
            "fingerprint"
        ].str.len()
        == DEFAULT_FINGERPRINT_SIZE
    ).all()

    assert (
        result.matrix.sum(axis=1) > 0
    ).all()
