from __future__ import annotations

import pytest

from lib.molecules.dataset_discovery import (
    discover_complete_datasets,
    discover_datasets_to_process,
    discover_processed_datasets,
    extract_dataset_id,
    normalize_prefix,
    select_datasets_to_process,
)


def test_normalize_prefix():
    assert normalize_prefix("input") == "input/"
    assert normalize_prefix("/input/") == "input/"
    assert normalize_prefix("") == ""


def test_normalize_prefix_rejects_non_string():
    with pytest.raises(
        TypeError,
        match="S3 prefix must be a string",
    ):
        normalize_prefix(123)


def test_extract_dataset_id():
    result = extract_dataset_id(
        key="input/test001_scaffolds.csv",
        prefix="input",
        suffix="_scaffolds.csv",
    )

    assert result == "test001"


def test_extract_dataset_id_returns_none_for_wrong_prefix():
    result = extract_dataset_id(
        key="archive/test001_scaffolds.csv",
        prefix="input",
        suffix="_scaffolds.csv",
    )

    assert result is None


def test_extract_dataset_id_returns_none_for_wrong_suffix():
    result = extract_dataset_id(
        key="input/test001.csv",
        prefix="input",
        suffix="_scaffolds.csv",
    )

    assert result is None


def test_extract_dataset_id_ignores_nested_folders():
    result = extract_dataset_id(
        key="input/archive/test001_scaffolds.csv",
        prefix="input",
        suffix="_scaffolds.csv",
    )

    assert result is None


def test_extract_dataset_id_returns_none_for_empty_dataset_id():
    result = extract_dataset_id(
        key="input/_scaffolds.csv",
        prefix="input",
        suffix="_scaffolds.csv",
    )

    assert result is None


def test_extract_dataset_id_rejects_non_string_key():
    with pytest.raises(
        TypeError,
        match="S3 object key must be a string",
    ):
        extract_dataset_id(
            key=123,
            prefix="input",
            suffix="_scaffolds.csv",
        )


def test_discover_complete_datasets():
    input_keys = [
        "input/test001_scaffolds.csv",
        "input/test001_r_groups.csv",
        "input/test002_scaffolds.csv",
        "input/test002_r_groups.csv",
    ]

    result = discover_complete_datasets(
        input_keys=input_keys,
        input_prefix="input",
    )

    assert result == [
        "test001",
        "test002",
    ]


def test_discover_complete_datasets_skips_incomplete_pairs():
    input_keys = [
        "input/test001_scaffolds.csv",
        "input/test001_r_groups.csv",
        "input/test002_scaffolds.csv",
        "input/test003_r_groups.csv",
    ]

    result = discover_complete_datasets(
        input_keys=input_keys,
        input_prefix="input",
    )

    assert result == ["test001"]


def test_discover_complete_datasets_ignores_unrelated_files():
    input_keys = [
        "input/test001_scaffolds.csv",
        "input/test001_r_groups.csv",
        "input/readme.txt",
        "output/test002_scaffolds.csv",
        "input/archive/test003_scaffolds.csv",
    ]

    result = discover_complete_datasets(
        input_keys=input_keys,
        input_prefix="input",
    )

    assert result == ["test001"]


def test_discover_complete_datasets_removes_duplicates():
    input_keys = [
        "input/test001_scaffolds.csv",
        "input/test001_scaffolds.csv",
        "input/test001_r_groups.csv",
        "input/test001_r_groups.csv",
    ]

    result = discover_complete_datasets(
        input_keys=input_keys,
        input_prefix="input",
    )

    assert result == ["test001"]


def test_discover_complete_datasets_returns_sorted_ids():
    input_keys = [
        "input/test010_scaffolds.csv",
        "input/test002_r_groups.csv",
        "input/test010_r_groups.csv",
        "input/test002_scaffolds.csv",
    ]

    result = discover_complete_datasets(
        input_keys=input_keys,
        input_prefix="input",
    )

    assert result == [
        "test002",
        "test010",
    ]


def test_discover_complete_datasets_rejects_non_list():
    with pytest.raises(
        TypeError,
        match="Input keys must be provided as a list",
    ):
        discover_complete_datasets(
            input_keys="input/test001_scaffolds.csv",
        )


def test_discover_processed_datasets():
    output_keys = [
        "output/test001_clustered_molecules.csv",
        "output/test002_clustered_molecules.csv",
        "output/test003_fingerprints.csv",
    ]

    result = discover_processed_datasets(
        output_keys=output_keys,
        output_prefix="output",
    )

    assert result == {
        "test001",
        "test002",
    }


def test_discover_processed_datasets_ignores_other_outputs():
    output_keys = [
        "output/test001_generated_molecules.csv",
        "output/test001_molecular_properties.csv",
        "output/test001_fingerprints.csv",
    ]

    result = discover_processed_datasets(
        output_keys=output_keys,
        output_prefix="output",
    )

    assert result == set()


def test_discover_processed_datasets_rejects_non_list():
    with pytest.raises(
        TypeError,
        match="Output keys must be provided as a list",
    ):
        discover_processed_datasets(
            output_keys="output/test001_clustered_molecules.csv",
        )


def test_select_datasets_to_process_without_overwrite():
    result = select_datasets_to_process(
        complete_dataset_ids=[
            "test001",
            "test002",
        ],
        output_keys=[
            "output/test001_clustered_molecules.csv",
        ],
        overwrite=False,
        output_prefix="output",
    )

    assert result == ["test002"]


def test_select_datasets_to_process_with_overwrite():
    result = select_datasets_to_process(
        complete_dataset_ids=[
            "test002",
            "test001",
            "test001",
        ],
        output_keys=[
            "output/test001_clustered_molecules.csv",
        ],
        overwrite=True,
        output_prefix="output",
    )

    assert result == [
        "test001",
        "test002",
    ]


def test_select_datasets_to_process_rejects_non_boolean_overwrite():
    with pytest.raises(
        TypeError,
        match="Overwrite must be a boolean",
    ):
        select_datasets_to_process(
            complete_dataset_ids=["test001"],
            output_keys=[],
            overwrite="false",
        )


def test_select_datasets_to_process_rejects_non_list_dataset_ids():
    with pytest.raises(
        TypeError,
        match="Dataset IDs must be provided as a list",
    ):
        select_datasets_to_process(
            complete_dataset_ids="test001",
            output_keys=[],
        )


def test_discover_datasets_to_process_without_overwrite():
    input_keys = [
        "input/test001_scaffolds.csv",
        "input/test001_r_groups.csv",
        "input/test002_scaffolds.csv",
        "input/test002_r_groups.csv",
        "input/test003_scaffolds.csv",
    ]

    output_keys = [
        "output/test001_clustered_molecules.csv",
    ]

    result = discover_datasets_to_process(
        input_keys=input_keys,
        output_keys=output_keys,
        overwrite=False,
        input_prefix="input",
        output_prefix="output",
    )

    assert result == ["test002"]


def test_discover_datasets_to_process_with_overwrite():
    input_keys = [
        "input/test001_scaffolds.csv",
        "input/test001_r_groups.csv",
        "input/test002_scaffolds.csv",
        "input/test002_r_groups.csv",
    ]

    output_keys = [
        "output/test001_clustered_molecules.csv",
    ]

    result = discover_datasets_to_process(
        input_keys=input_keys,
        output_keys=output_keys,
        overwrite=True,
        input_prefix="input",
        output_prefix="output",
    )

    assert result == [
        "test001",
        "test002",
    ]


def test_discover_datasets_to_process_returns_empty_list():
    result = discover_datasets_to_process(
        input_keys=[
            "input/test001_scaffolds.csv",
        ],
        output_keys=[],
        overwrite=False,
    )

    assert result == []
