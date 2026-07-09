from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from lib.molecules.dataset_discovery import (
    discover_changed_datasets,
    discover_complete_datasets,
    discover_datasets_to_process,
    discover_processed_datasets,
    extract_dataset_id,
    normalize_prefix,
)


INTERVAL_START = datetime(
    2026,
    7,
    6,
    3,
    0,
    tzinfo=timezone.utc,
)

INTERVAL_END = INTERVAL_START + timedelta(
    days=7
)


def build_object(
    key: str,
    last_modified: datetime,
) -> dict[str, object]:
    """Build one S3 object metadata record for tests."""
    return {
        "key": key,
        "last_modified": last_modified,
    }


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


def test_discover_changed_datasets():
    input_objects = [
        build_object(
            "input/test001_scaffolds.csv",
            INTERVAL_START + timedelta(days=1),
        ),
        build_object(
            "input/test001_r_groups.csv",
            INTERVAL_START - timedelta(days=1),
        ),
        build_object(
            "input/test002_scaffolds.csv",
            INTERVAL_START - timedelta(days=2),
        ),
        build_object(
            "input/test002_r_groups.csv",
            INTERVAL_END,
        ),
        build_object(
            "input/readme.txt",
            INTERVAL_START + timedelta(days=2),
        ),
    ]

    result = discover_changed_datasets(
        input_objects=input_objects,
        interval_start=INTERVAL_START,
        interval_end=INTERVAL_END,
        input_prefix="input",
    )

    assert result == {"test001"}


def test_discover_changed_datasets_includes_interval_start():
    input_objects = [
        build_object(
            "input/test001_scaffolds.csv",
            INTERVAL_START,
        ),
    ]

    result = discover_changed_datasets(
        input_objects=input_objects,
        interval_start=INTERVAL_START,
        interval_end=INTERVAL_END,
    )

    assert result == {"test001"}


def test_discover_changed_datasets_excludes_interval_end():
    input_objects = [
        build_object(
            "input/test001_scaffolds.csv",
            INTERVAL_END,
        ),
    ]

    result = discover_changed_datasets(
        input_objects=input_objects,
        interval_start=INTERVAL_START,
        interval_end=INTERVAL_END,
    )

    assert result == set()


def test_discover_changed_datasets_rejects_invalid_interval():
    with pytest.raises(
        ValueError,
        match="Interval start must be earlier than interval end",
    ):
        discover_changed_datasets(
            input_objects=[],
            interval_start=INTERVAL_END,
            interval_end=INTERVAL_START,
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


def test_discover_datasets_to_process_without_overwrite():
    input_objects = [
        build_object(
            "input/test001_scaffolds.csv",
            INTERVAL_START + timedelta(days=1),
        ),
        build_object(
            "input/test001_r_groups.csv",
            INTERVAL_START + timedelta(days=1),
        ),
        build_object(
            "input/test002_scaffolds.csv",
            INTERVAL_START + timedelta(days=2),
        ),
        build_object(
            "input/test002_r_groups.csv",
            INTERVAL_START - timedelta(days=2),
        ),
        build_object(
            "input/test003_scaffolds.csv",
            INTERVAL_START + timedelta(days=3),
        ),
    ]

    output_keys = [
        "output/test001_clustered_molecules.csv",
    ]

    result = discover_datasets_to_process(
        input_objects=input_objects,
        output_keys=output_keys,
        interval_start=INTERVAL_START,
        interval_end=INTERVAL_END,
        overwrite=False,
        input_prefix="input",
        output_prefix="output",
    )

    assert result == ["test002"]


def test_discover_datasets_to_process_with_overwrite():
    input_objects = [
        build_object(
            "input/test001_scaffolds.csv",
            INTERVAL_START - timedelta(days=10),
        ),
        build_object(
            "input/test001_r_groups.csv",
            INTERVAL_START - timedelta(days=10),
        ),
        build_object(
            "input/test002_scaffolds.csv",
            INTERVAL_START - timedelta(days=10),
        ),
        build_object(
            "input/test002_r_groups.csv",
            INTERVAL_START - timedelta(days=10),
        ),
    ]

    output_keys = [
        "output/test001_clustered_molecules.csv",
    ]

    result = discover_datasets_to_process(
        input_objects=input_objects,
        output_keys=output_keys,
        interval_start=INTERVAL_START,
        interval_end=INTERVAL_END,
        overwrite=True,
        input_prefix="input",
        output_prefix="output",
    )

    assert result == [
        "test001",
        "test002",
    ]


def test_discover_datasets_to_process_requires_complete_pair():
    input_objects = [
        build_object(
            "input/test001_scaffolds.csv",
            INTERVAL_START + timedelta(days=1),
        ),
    ]

    result = discover_datasets_to_process(
        input_objects=input_objects,
        output_keys=[],
        interval_start=INTERVAL_START,
        interval_end=INTERVAL_END,
        overwrite=False,
    )

    assert result == []


def test_discover_datasets_to_process_rejects_non_boolean_overwrite():
    with pytest.raises(
        TypeError,
        match="Overwrite must be a boolean",
    ):
        discover_datasets_to_process(
            input_objects=[],
            output_keys=[],
            interval_start=INTERVAL_START,
            interval_end=INTERVAL_END,
            overwrite="false",
        )
