from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from lib.molecules.properties import (
    PROPERTY_COLUMNS,
    calculate_molecular_properties,
    calculate_properties_dataframe,
    check_lipinski_rule,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

EXPECTED_MOLECULES_PATH = (
    PROJECT_ROOT
    / "tests"
    / "fixture"
    / "output"
    / "test001_generated_molecules.csv"
)


def test_calculate_molecular_properties_for_ethanol():
    properties = calculate_molecular_properties(
        "CCO"
    )

    assert properties["canonical_smiles"] == "CCO"
    assert properties["mol_formula"] == "C2H6O"

    assert properties["mol_weight"] == pytest.approx(
        46.069,
        abs=0.001,
    )

    assert properties["log_p"] == pytest.approx(
        -0.0014,
        abs=0.001,
    )

    assert properties["tpsa"] == pytest.approx(
        20.23,
        abs=0.01,
    )

    assert properties["hba"] == 1
    assert properties["hbd"] == 1
    assert properties["rotatable_bonds"] == 0
    assert properties["aromatic_rings"] == 0
    assert properties["heavy_atom_count"] == 3
    assert properties["hetero_atom_count"] == 1
    assert properties["ring_count"] == 0

    assert properties["fraction_csp3"] == pytest.approx(
        1.0
    )

    assert properties["formal_charge"] == 0

    assert 0 <= properties["qed"] <= 1
    assert properties["lipinski_pass"] is True


def test_calculate_molecular_properties_returns_all_columns():
    properties = calculate_molecular_properties(
        "c1ccccc1"
    )

    assert list(properties.keys()) == PROPERTY_COLUMNS


def test_calculate_molecular_properties_strips_whitespace():
    properties = calculate_molecular_properties(
        "  CCO  "
    )

    assert properties["canonical_smiles"] == "CCO"


def test_calculate_molecular_properties_rejects_blank_smiles():
    with pytest.raises(
        ValueError,
        match="SMILES must not be blank",
    ):
        calculate_molecular_properties(
            "   "
        )


def test_calculate_molecular_properties_rejects_invalid_smiles():
    with pytest.raises(
        ValueError,
        match="Invalid SMILES",
    ):
        calculate_molecular_properties(
            "not_a_smiles"
        )


def test_calculate_molecular_properties_rejects_non_string():
    with pytest.raises(
        TypeError,
        match="SMILES must be a string",
    ):
        calculate_molecular_properties(
            123
        )


def test_check_lipinski_rule_passes():
    result = check_lipinski_rule(
        mol_weight=450,
        log_p=4.5,
        hba=8,
        hbd=3,
    )

    assert result is True


def test_check_lipinski_rule_fails_for_high_molecular_weight():
    result = check_lipinski_rule(
        mol_weight=501,
        log_p=4,
        hba=8,
        hbd=3,
    )

    assert result is False


def test_check_lipinski_rule_accepts_boundary_values():
    result = check_lipinski_rule(
        mol_weight=500,
        log_p=5,
        hba=10,
        hbd=5,
    )

    assert result is True


def test_calculate_properties_dataframe():
    molecules = pd.DataFrame(
        {
            "scaffold_id": [0, 1],
            "r_group_id": [0, 0],
            "generated_smiles": [
                "CCO",
                "c1ccccc1",
            ],
        }
    )

    result = calculate_properties_dataframe(
        molecules
    )

    assert len(result) == 2

    assert list(result.columns) == [
        "scaffold_id",
        "r_group_id",
        "generated_smiles",
        *PROPERTY_COLUMNS,
    ]

    assert result.loc[
        0,
        "mol_formula",
    ] == "C2H6O"

    assert result.loc[
        1,
        "mol_formula",
    ] == "C6H6"

    assert result["lipinski_pass"].all()


def test_calculate_properties_dataframe_preserves_original_columns():
    molecules = pd.DataFrame(
        {
            "scaffold_id": [4],
            "r_group_id": [7],
            "scaffold_smiles": ["CC[*:1]"],
            "r_group_smiles": ["[*:1]O"],
            "generated_smiles": ["CCO"],
        }
    )

    result = calculate_properties_dataframe(
        molecules
    )

    assert result.loc[0, "scaffold_id"] == 4
    assert result.loc[0, "r_group_id"] == 7
    assert (
        result.loc[0, "scaffold_smiles"]
        == "CC[*:1]"
    )
    assert (
        result.loc[0, "r_group_smiles"]
        == "[*:1]O"
    )
    assert (
        result.loc[0, "generated_smiles"]
        == "CCO"
    )


def test_calculate_properties_dataframe_rejects_missing_smiles_column():
    molecules = pd.DataFrame(
        {
            "molecule": ["CCO"],
        }
    )

    with pytest.raises(
        ValueError,
        match=(
            "Molecules DataFrame must contain the "
            "'generated_smiles' column"
        ),
    ):
        calculate_properties_dataframe(
            molecules
        )


def test_calculate_properties_dataframe_rejects_empty_dataframe():
    molecules = pd.DataFrame(
        columns=[
            "generated_smiles",
        ]
    )

    with pytest.raises(
        ValueError,
        match="Molecules DataFrame must not be empty",
    ):
        calculate_properties_dataframe(
            molecules
        )


def test_calculate_properties_dataframe_rejects_non_dataframe():
    with pytest.raises(
        TypeError,
        match=(
            "Molecules input must be a pandas DataFrame"
        ),
    ):
        calculate_properties_dataframe(
            ["CCO"]
        )


def test_calculate_properties_dataframe_reports_failed_row():
    molecules = pd.DataFrame(
        {
            "generated_smiles": [
                "CCO",
                "not_a_smiles",
            ]
        }
    )

    with pytest.raises(
        ValueError,
        match=(
            "Failed to calculate molecular properties "
            "for row_index=1"
        ),
    ):
        calculate_properties_dataframe(
            molecules
        )


def test_calculate_properties_for_generated_fixture():
    molecules = pd.read_csv(
        EXPECTED_MOLECULES_PATH
    )

    result = calculate_properties_dataframe(
        molecules
    )

    assert len(result) == 100

    assert all(
        column in result.columns
        for column in PROPERTY_COLUMNS
    )

    assert result["canonical_smiles"].notna().all()
    assert result["mol_formula"].notna().all()

    assert (
        result["mol_weight"] > 0
    ).all()

    assert (
        result["heavy_atom_count"] > 0
    ).all()

    assert result["qed"].between(
        0,
        1,
        inclusive="both",
    ).all()

    assert result["lipinski_pass"].dtype == bool
