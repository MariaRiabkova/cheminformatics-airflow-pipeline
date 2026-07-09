from pathlib import Path

import pandas as pd

from dags.lib.molecules.pipeline import (
    generate_molecules_dataframe,
    generate_molecules_from_csv_bytes,
)
from dags.lib.molecules.smiles_parser import (
    dataframe_to_csv_bytes,
    parse_smiles_csv,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

FIXTURE_INPUT_DIRECTORY = (
    PROJECT_ROOT
    / "tests"
    / "fixture"
    / "input"
)

DATASET_ID = "test001"

SCAFFOLDS_PATH = (
    FIXTURE_INPUT_DIRECTORY
    / f"{DATASET_ID}_scaffolds.csv"
)

R_GROUPS_PATH = (
    FIXTURE_INPUT_DIRECTORY
    / f"{DATASET_ID}_r_groups.csv"
)

EXPECTED_OUTPUT_PATH = (
    PROJECT_ROOT
    / "tests"
    / "fixture"
    / "output"
    / f"{DATASET_ID}_generated_molecules.csv"
)

def test_parse_scaffolds_fixture():
    dataframe = parse_smiles_csv(
        raw=SCAFFOLDS_PATH.read_bytes(),
        source_name=str(SCAFFOLDS_PATH),
    )

    assert list(dataframe.columns) == ["smiles"]
    assert len(dataframe) == 10
    assert dataframe["smiles"].str.strip().ne("").all()


def test_parse_r_groups_fixture():
    dataframe = parse_smiles_csv(
        raw=R_GROUPS_PATH.read_bytes(),
        source_name=str(R_GROUPS_PATH),
    )

    assert list(dataframe.columns) == ["smiles"]
    assert len(dataframe) == 10
    assert dataframe["smiles"].str.strip().ne("").all()


def test_generate_molecules_dataframe_from_fixtures():
    scaffolds = parse_smiles_csv(
        raw=SCAFFOLDS_PATH.read_bytes(),
        source_name=str(SCAFFOLDS_PATH),
    )

    r_groups = parse_smiles_csv(
        raw=R_GROUPS_PATH.read_bytes(),
        source_name=str(R_GROUPS_PATH),
    )

    result = generate_molecules_dataframe(
        scaffolds=scaffolds,
        r_groups=r_groups,
    )

    assert len(result) == 100

    assert list(result.columns) == [
        "scaffold_id",
        "r_group_id",
        "scaffold_smiles",
        "r_group_smiles",
        "generated_smiles",
    ]

    assert result["scaffold_id"].nunique() == 10
    assert result["r_group_id"].nunique() == 10
    assert result["generated_smiles"].str.strip().ne("").all()

    assert (
        result[
            [
                "scaffold_id",
                "r_group_id",
            ]
        ]
        .drop_duplicates()
        .shape[0]
        == 100
    )


def test_generate_molecules_from_csv_bytes():
    result = generate_molecules_from_csv_bytes(
        scaffolds_raw=SCAFFOLDS_PATH.read_bytes(),
        r_groups_raw=R_GROUPS_PATH.read_bytes(),
        scaffolds_source_name=str(SCAFFOLDS_PATH),
        r_groups_source_name=str(R_GROUPS_PATH),
    )

    assert len(result) == 100
    assert result["generated_smiles"].str.strip().ne("").all()


def test_generated_dataframe_can_be_serialized():
    result = generate_molecules_from_csv_bytes(
        scaffolds_raw=SCAFFOLDS_PATH.read_bytes(),
        r_groups_raw=R_GROUPS_PATH.read_bytes(),
        scaffolds_source_name=str(SCAFFOLDS_PATH),
        r_groups_source_name=str(R_GROUPS_PATH),
    )

    csv_bytes = dataframe_to_csv_bytes(result)

    assert isinstance(csv_bytes, bytes)
    assert csv_bytes.startswith(
        b"scaffold_id,r_group_id,"
    )


def test_generated_output_can_be_written_and_read(
    tmp_path: Path,
):
    result = generate_molecules_from_csv_bytes(
        scaffolds_raw=SCAFFOLDS_PATH.read_bytes(),
        r_groups_raw=R_GROUPS_PATH.read_bytes(),
        scaffolds_source_name=str(SCAFFOLDS_PATH),
        r_groups_source_name=str(R_GROUPS_PATH),
    )

    output_path = (
        tmp_path
        / f"{DATASET_ID}_generated_molecules.csv"
    )

    output_path.write_bytes(
        dataframe_to_csv_bytes(result)
    )

    saved_dataframe = pd.read_csv(output_path)

    assert output_path.is_file()
    assert len(saved_dataframe) == 100

def test_pipeline_matches_expected_fixture():
    actual = generate_molecules_from_csv_bytes(
        scaffolds_raw=SCAFFOLDS_PATH.read_bytes(),
        r_groups_raw=R_GROUPS_PATH.read_bytes(),
        scaffolds_source_name=str(SCAFFOLDS_PATH),
        r_groups_source_name=str(R_GROUPS_PATH),
    )

    expected = pd.read_csv(EXPECTED_OUTPUT_PATH)

    pd.testing.assert_frame_equal(
        actual.reset_index(drop=True),
        expected.reset_index(drop=True),
    )