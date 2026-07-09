from __future__ import annotations

from datetime import datetime, timedelta, timezone
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lib.molecules import storage_pipeline


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


def test_normalize_dataset_id():
    result = storage_pipeline.normalize_dataset_id(
        "  test001  "
    )

    assert result == "test001"


def test_normalize_dataset_id_rejects_blank_value():
    with pytest.raises(
        ValueError,
        match="Dataset ID must not be blank",
    ):
        storage_pipeline.normalize_dataset_id("   ")


def test_normalize_dataset_id_rejects_non_string():
    with pytest.raises(
        TypeError,
        match="Dataset ID must be a string",
    ):
        storage_pipeline.normalize_dataset_id(123)


def test_normalize_dataset_id_rejects_forward_slash():
    with pytest.raises(
        ValueError,
        match="Dataset ID must not contain '/'",
    ):
        storage_pipeline.normalize_dataset_id(
            "folder/test001"
        )


def test_normalize_dataset_id_rejects_backslash():
    with pytest.raises(
        ValueError,
        match="Dataset ID must not contain",
    ):
        storage_pipeline.normalize_dataset_id(
            r"folder\test001"
        )


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (5, 5),
        ("5", 5),
        (" 5 ", 5),
    ],
)
def test_normalize_n_clusters(
    value: int | str,
    expected: int,
):
    assert (
        storage_pipeline.normalize_n_clusters(value)
        == expected
    )


@pytest.mark.parametrize(
    "value",
    [
        0,
        1,
        "1",
    ],
)
def test_normalize_n_clusters_rejects_small_values(
    value: int | str,
):
    with pytest.raises(
        ValueError,
        match="Number of clusters must be at least 2",
    ):
        storage_pipeline.normalize_n_clusters(value)


@pytest.mark.parametrize(
    "value",
    [
        None,
        "",
        "abc",
        True,
    ],
)
def test_normalize_n_clusters_rejects_invalid_values(
    value: object,
):
    with pytest.raises(
        TypeError,
        match="Number of clusters must be an integer",
    ):
        storage_pipeline.normalize_n_clusters(value)


def test_build_dataset_keys():
    result = storage_pipeline.build_dataset_keys(
        dataset_id="test001",
        input_prefix="input",
        output_prefix="output",
    )

    assert result == (
        "input/test001_scaffolds.csv",
        "input/test001_r_groups.csv",
        "output/test001_generated_molecules.csv",
    )


def test_build_dataset_keys_strips_prefix_slashes():
    result = storage_pipeline.build_dataset_keys(
        dataset_id="test001",
        input_prefix="/incoming/",
        output_prefix="/results/",
    )

    assert result == (
        "incoming/test001_scaffolds.csv",
        "incoming/test001_r_groups.csv",
        "results/test001_generated_molecules.csv",
    )


def test_build_dataset_keys_supports_empty_prefixes():
    result = storage_pipeline.build_dataset_keys(
        dataset_id="test001",
        input_prefix="",
        output_prefix="",
    )

    assert result == (
        "test001_scaffolds.csv",
        "test001_r_groups.csv",
        "test001_generated_molecules.csv",
    )


def test_build_properties_keys():
    assert storage_pipeline.build_properties_keys(
        dataset_id="test001",
        output_prefix="output",
    ) == (
        "output/test001_generated_molecules.csv",
        "output/test001_molecular_properties.csv",
    )


def test_build_fingerprints_keys():
    assert storage_pipeline.build_fingerprints_keys(
        dataset_id="test001",
        output_prefix="output",
    ) == (
        "output/test001_molecular_properties.csv",
        "output/test001_fingerprints.csv",
    )


def test_build_clustering_keys():
    assert storage_pipeline.build_clustering_keys(
        dataset_id="test001",
        output_prefix="output",
    ) == (
        "output/test001_fingerprints.csv",
        "output/test001_clustered_molecules.csv",
    )


def test_require_s3_object_when_object_exists():
    calls: list[dict[str, str]] = []

    def fake_object_exists(
        key: str,
        bucket_name: str,
        aws_conn_id: str,
    ) -> bool:
        calls.append(
            {
                "key": key,
                "bucket_name": bucket_name,
                "aws_conn_id": aws_conn_id,
            }
        )
        return True

    storage_pipeline.require_s3_object(
        key="input/test001_scaffolds.csv",
        bucket_name="test-bucket",
        aws_conn_id="test-connection",
        object_exists=fake_object_exists,
    )

    assert calls == [
        {
            "key": "input/test001_scaffolds.csv",
            "bucket_name": "test-bucket",
            "aws_conn_id": "test-connection",
        }
    ]


def test_require_s3_object_raises_when_object_is_missing():
    def fake_object_exists(
        key: str,
        bucket_name: str,
        aws_conn_id: str,
    ) -> bool:
        return False

    with pytest.raises(
        FileNotFoundError,
        match=(
            "Required S3 object does not exist: "
            "s3://test-bucket/input/test001_scaffolds.csv"
        ),
    ):
        storage_pipeline.require_s3_object(
            key="input/test001_scaffolds.csv",
            bucket_name="test-bucket",
            aws_conn_id="test-connection",
            object_exists=fake_object_exists,
        )


def test_fingerprint_strings_to_matrix():
    dataframe = pd.DataFrame(
        {
            "fingerprint": [
                "0101",
                "1100",
            ]
        }
    )

    matrix = storage_pipeline.fingerprint_strings_to_matrix(
        dataframe
    )

    assert matrix.dtype == np.uint8
    assert matrix.shape == (2, 4)
    assert matrix.tolist() == [
        [0, 1, 0, 1],
        [1, 1, 0, 0],
    ]


def test_fingerprint_strings_to_matrix_rejects_invalid_bits():
    dataframe = pd.DataFrame(
        {
            "fingerprint": [
                "0101",
                "01x1",
            ]
        }
    )

    with pytest.raises(
        ValueError,
        match="only '0' and '1'",
    ):
        storage_pipeline.fingerprint_strings_to_matrix(
            dataframe
        )


def test_process_s3_dataset():
    scaffolds_raw = SCAFFOLDS_PATH.read_bytes()
    r_groups_raw = R_GROUPS_PATH.read_bytes()

    checked_keys: list[str] = []
    downloaded_keys: list[str] = []
    uploaded_objects: list[dict[str, object]] = []

    def fake_object_exists(
        key: str,
        bucket_name: str,
        aws_conn_id: str,
    ) -> bool:
        checked_keys.append(key)
        return True

    def fake_download_object(
        key: str,
        bucket_name: str,
        aws_conn_id: str,
    ) -> bytes:
        downloaded_keys.append(key)

        if key == "input/test001_scaffolds.csv":
            return scaffolds_raw

        if key == "input/test001_r_groups.csv":
            return r_groups_raw

        raise AssertionError(
            f"Unexpected S3 key: {key}"
        )

    def fake_upload_bytes(
        data: bytes,
        key: str,
        bucket_name: str,
        aws_conn_id: str,
        replace: bool,
    ) -> None:
        uploaded_objects.append(
            {
                "data": data,
                "key": key,
                "bucket_name": bucket_name,
                "aws_conn_id": aws_conn_id,
                "replace": replace,
            }
        )

    output_key = storage_pipeline.process_s3_dataset(
        dataset_id="test001",
        object_exists=fake_object_exists,
        download_object=fake_download_object,
        upload_bytes=fake_upload_bytes,
        bucket_name="test-bucket",
        aws_conn_id="test-connection",
        input_prefix="input",
        output_prefix="output",
        replace=True,
    )

    assert output_key == (
        "output/test001_generated_molecules.csv"
    )

    assert checked_keys == [
        "input/test001_scaffolds.csv",
        "input/test001_r_groups.csv",
    ]

    assert downloaded_keys == [
        "input/test001_scaffolds.csv",
        "input/test001_r_groups.csv",
    ]

    assert len(uploaded_objects) == 1

    uploaded_dataframe = pd.read_csv(
        BytesIO(uploaded_objects[0]["data"])
    )

    assert len(uploaded_dataframe) == 100

    assert list(uploaded_dataframe.columns) == [
        "scaffold_id",
        "r_group_id",
        "scaffold_smiles",
        "r_group_smiles",
        "generated_smiles",
    ]


def test_process_properties_s3_dataset():
    input_dataframe = pd.DataFrame(
        {
            "scaffold_id": [1, 2],
            "r_group_id": [1, 2],
            "generated_smiles": [
                "CCO",
                "c1ccccc1",
            ],
        }
    )

    uploaded_objects: list[dict[str, object]] = []

    def fake_object_exists(
        key: str,
        bucket_name: str,
        aws_conn_id: str,
    ) -> bool:
        return True

    def fake_download_object(
        key: str,
        bucket_name: str,
        aws_conn_id: str,
    ) -> bytes:
        assert key == (
            "output/test001_generated_molecules.csv"
        )
        return input_dataframe.to_csv(
            index=False
        ).encode("utf-8")

    def fake_upload_bytes(
        data: bytes,
        key: str,
        bucket_name: str,
        aws_conn_id: str,
        replace: bool,
    ) -> None:
        uploaded_objects.append(
            {
                "data": data,
                "key": key,
                "replace": replace,
            }
        )

    output_key = (
        storage_pipeline.process_properties_s3_dataset(
            dataset_id="test001",
            object_exists=fake_object_exists,
            download_object=fake_download_object,
            upload_bytes=fake_upload_bytes,
            bucket_name="test-bucket",
            aws_conn_id="test-connection",
            output_prefix="output",
            replace=True,
        )
    )

    assert output_key == (
        "output/test001_molecular_properties.csv"
    )
    assert len(uploaded_objects) == 1
    assert uploaded_objects[0]["key"] == output_key
    assert uploaded_objects[0]["replace"] is True

    output_dataframe = pd.read_csv(
        BytesIO(uploaded_objects[0]["data"])
    )

    assert len(output_dataframe) == 2
    assert "canonical_smiles" in output_dataframe.columns
    assert "mol_weight" in output_dataframe.columns
    assert "lipinski_pass" in output_dataframe.columns


def test_process_fingerprints_s3_dataset():
    input_dataframe = pd.DataFrame(
        {
            "scaffold_id": [1, 2],
            "r_group_id": [1, 2],
            "generated_smiles": [
                "CCO",
                "c1ccccc1",
            ],
            "canonical_smiles": [
                "CCO",
                "c1ccccc1",
            ],
        }
    )

    uploaded_objects: list[dict[str, object]] = []

    def fake_object_exists(
        key: str,
        bucket_name: str,
        aws_conn_id: str,
    ) -> bool:
        return True

    def fake_download_object(
        key: str,
        bucket_name: str,
        aws_conn_id: str,
    ) -> bytes:
        assert key == (
            "output/test001_molecular_properties.csv"
        )
        return input_dataframe.to_csv(
            index=False
        ).encode("utf-8")

    def fake_upload_bytes(
        data: bytes,
        key: str,
        bucket_name: str,
        aws_conn_id: str,
        replace: bool,
    ) -> None:
        uploaded_objects.append(
            {
                "data": data,
                "key": key,
            }
        )

    output_key = (
        storage_pipeline.process_fingerprints_s3_dataset(
            dataset_id="test001",
            object_exists=fake_object_exists,
            download_object=fake_download_object,
            upload_bytes=fake_upload_bytes,
            bucket_name="test-bucket",
            aws_conn_id="test-connection",
            output_prefix="output",
        )
    )

    assert output_key == (
        "output/test001_fingerprints.csv"
    )

    output_dataframe = pd.read_csv(
        BytesIO(uploaded_objects[0]["data"]),
        dtype={
            "fingerprint": str,
            "fingerprint_on_bits": str,
        },
    )

    assert len(output_dataframe) == 2
    assert "fingerprint_type" in output_dataframe.columns
    assert "fingerprint" in output_dataframe.columns
    assert "fingerprint_on_bits" in output_dataframe.columns
    assert output_dataframe[
        "fingerprint"
    ].str.len().eq(2048).all()


def test_process_clustering_s3_dataset():
    fingerprint_a = "0" * 2047 + "1"
    fingerprint_b = "0" * 2046 + "10"
    fingerprint_c = "1" + "0" * 2047
    fingerprint_d = "01" + "0" * 2046

    input_dataframe = pd.DataFrame(
        {
            "canonical_smiles": [
                "CCO",
                "CCN",
                "c1ccccc1",
                "c1ccncc1",
            ],
            "fingerprint": [
                fingerprint_a,
                fingerprint_b,
                fingerprint_c,
                fingerprint_d,
            ],
        }
    )

    uploaded_objects: list[dict[str, object]] = []

    def fake_object_exists(
        key: str,
        bucket_name: str,
        aws_conn_id: str,
    ) -> bool:
        return True

    def fake_download_object(
        key: str,
        bucket_name: str,
        aws_conn_id: str,
    ) -> bytes:
        assert key == (
            "output/test001_fingerprints.csv"
        )
        return input_dataframe.to_csv(
            index=False
        ).encode("utf-8")

    def fake_upload_bytes(
        data: bytes,
        key: str,
        bucket_name: str,
        aws_conn_id: str,
        replace: bool,
    ) -> None:
        uploaded_objects.append(
            {
                "data": data,
                "key": key,
            }
        )

    output_key = (
        storage_pipeline.process_clustering_s3_dataset(
            dataset_id="test001",
            object_exists=fake_object_exists,
            download_object=fake_download_object,
            upload_bytes=fake_upload_bytes,
            n_clusters="2",
            bucket_name="test-bucket",
            aws_conn_id="test-connection",
            output_prefix="output",
        )
    )

    assert output_key == (
        "output/test001_clustered_molecules.csv"
    )

    output_dataframe = pd.read_csv(
        BytesIO(uploaded_objects[0]["data"])
    )

    assert len(output_dataframe) == 4
    assert "cluster_id" in output_dataframe.columns
    assert "distance_to_centroid" in output_dataframe.columns
    assert output_dataframe["cluster_id"].nunique() == 2


def test_processing_stops_when_input_is_missing():
    download_was_called = False
    upload_was_called = False

    def fake_object_exists(
        key: str,
        bucket_name: str,
        aws_conn_id: str,
    ) -> bool:
        return False

    def fake_download_object(
        key: str,
        bucket_name: str,
        aws_conn_id: str,
    ) -> bytes:
        nonlocal download_was_called
        download_was_called = True
        return b""

    def fake_upload_bytes(
        data: bytes,
        key: str,
        bucket_name: str,
        aws_conn_id: str,
        replace: bool,
    ) -> None:
        nonlocal upload_was_called
        upload_was_called = True

    with pytest.raises(
        FileNotFoundError,
        match="test001_generated_molecules.csv",
    ):
        storage_pipeline.process_properties_s3_dataset(
            dataset_id="test001",
            object_exists=fake_object_exists,
            download_object=fake_download_object,
            upload_bytes=fake_upload_bytes,
            bucket_name="test-bucket",
            aws_conn_id="test-connection",
            output_prefix="output",
        )

    assert download_was_called is False
    assert upload_was_called is False

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


def test_discover_s3_datasets_to_process():
    calls: list[tuple[str, str, str]] = []

    def fake_list_objects(
        prefix: str,
        bucket_name: str,
        aws_conn_id: str,
    ) -> list[dict[str, object]]:
        calls.append(
            (
                prefix,
                bucket_name,
                aws_conn_id,
            )
        )

        if prefix == "input":
            return [
                {
                    "key": "input/test001_scaffolds.csv",
                    "last_modified": (
                        INTERVAL_START
                        + timedelta(days=1)
                    ),
                },
                {
                    "key": "input/test001_r_groups.csv",
                    "last_modified": (
                        INTERVAL_START
                        + timedelta(days=1)
                    ),
                },
                {
                    "key": "input/test002_scaffolds.csv",
                    "last_modified": (
                        INTERVAL_START
                        + timedelta(days=2)
                    ),
                },
                {
                    "key": "input/test002_r_groups.csv",
                    "last_modified": (
                        INTERVAL_START
                        - timedelta(days=2)
                    ),
                },
                {
                    "key": "input/test003_scaffolds.csv",
                    "last_modified": (
                        INTERVAL_START
                        + timedelta(days=3)
                    ),
                },
            ]

        if prefix == "output":
            return [
                {
                    "key": (
                        "output/"
                        "test001_clustered_molecules.csv"
                    ),
                    "last_modified": (
                        INTERVAL_START
                        - timedelta(days=5)
                    ),
                },
            ]

        raise AssertionError(
            f"Unexpected prefix: {prefix}"
        )

    result = (
        storage_pipeline.discover_s3_datasets_to_process(
            list_objects=fake_list_objects,
            interval_start=INTERVAL_START,
            interval_end=INTERVAL_END,
            overwrite=False,
            bucket_name="test-bucket",
            aws_conn_id="test-connection",
            input_prefix="input",
            output_prefix="output",
        )
    )

    assert result == ["test002"]

    assert calls == [
        (
            "input",
            "test-bucket",
            "test-connection",
        ),
        (
            "output",
            "test-bucket",
            "test-connection",
        ),
    ]


def test_discover_s3_datasets_to_process_with_overwrite():
    def fake_list_objects(
        prefix: str,
        bucket_name: str,
        aws_conn_id: str,
    ) -> list[dict[str, object]]:
        if prefix == "input":
            return [
                {
                    "key": "input/test001_scaffolds.csv",
                    "last_modified": (
                        INTERVAL_START
                        - timedelta(days=10)
                    ),
                },
                {
                    "key": "input/test001_r_groups.csv",
                    "last_modified": (
                        INTERVAL_START
                        - timedelta(days=10)
                    ),
                },
                {
                    "key": "input/test002_scaffolds.csv",
                    "last_modified": (
                        INTERVAL_START
                        - timedelta(days=10)
                    ),
                },
                {
                    "key": "input/test002_r_groups.csv",
                    "last_modified": (
                        INTERVAL_START
                        - timedelta(days=10)
                    ),
                },
            ]

        if prefix == "output":
            return [
                {
                    "key": (
                        "output/"
                        "test001_clustered_molecules.csv"
                    ),
                    "last_modified": (
                        INTERVAL_START
                        - timedelta(days=5)
                    ),
                },
            ]

        raise AssertionError(
            f"Unexpected prefix: {prefix}"
        )

    result = (
        storage_pipeline.discover_s3_datasets_to_process(
            list_objects=fake_list_objects,
            interval_start=INTERVAL_START,
            interval_end=INTERVAL_END,
            overwrite=True,
            bucket_name="test-bucket",
            aws_conn_id="test-connection",
            input_prefix="input",
            output_prefix="output",
        )
    )

    assert result == [
        "test001",
        "test002",
    ]


def test_process_complete_s3_dataset(
    monkeypatch,
):
    calls: list[dict[str, object]] = []

    def fake_process_s3_dataset(**kwargs):
        calls.append(
            {
                "stage": "generation",
                **kwargs,
            }
        )
        return "output/test001_generated_molecules.csv"

    def fake_process_properties_s3_dataset(**kwargs):
        calls.append(
            {
                "stage": "properties",
                **kwargs,
            }
        )
        return "output/test001_molecular_properties.csv"

    def fake_process_fingerprints_s3_dataset(**kwargs):
        calls.append(
            {
                "stage": "fingerprints",
                **kwargs,
            }
        )
        return "output/test001_fingerprints.csv"

    def fake_process_clustering_s3_dataset(**kwargs):
        calls.append(
            {
                "stage": "clustering",
                **kwargs,
            }
        )
        return "output/test001_clustered_molecules.csv"

    monkeypatch.setattr(
        storage_pipeline,
        "process_s3_dataset",
        fake_process_s3_dataset,
    )
    monkeypatch.setattr(
        storage_pipeline,
        "process_properties_s3_dataset",
        fake_process_properties_s3_dataset,
    )
    monkeypatch.setattr(
        storage_pipeline,
        "process_fingerprints_s3_dataset",
        fake_process_fingerprints_s3_dataset,
    )
    monkeypatch.setattr(
        storage_pipeline,
        "process_clustering_s3_dataset",
        fake_process_clustering_s3_dataset,
    )

    def fake_object_exists(
        key: str,
        bucket_name: str,
        aws_conn_id: str,
    ) -> bool:
        return True

    def fake_download_object(
        key: str,
        bucket_name: str,
        aws_conn_id: str,
    ) -> bytes:
        return b""

    def fake_upload_bytes(
        data: bytes,
        key: str,
        bucket_name: str,
        aws_conn_id: str,
        replace: bool,
    ) -> None:
        return None

    result = storage_pipeline.process_complete_s3_dataset(
        dataset_id="test001",
        object_exists=fake_object_exists,
        download_object=fake_download_object,
        upload_bytes=fake_upload_bytes,
        overwrite=True,
        n_clusters="5",
        bucket_name="test-bucket",
        aws_conn_id="test-connection",
        input_prefix="input",
        output_prefix="output",
    )

    assert result == {
        "generated_molecules": (
            "output/test001_generated_molecules.csv"
        ),
        "molecular_properties": (
            "output/test001_molecular_properties.csv"
        ),
        "fingerprints": (
            "output/test001_fingerprints.csv"
        ),
        "clustered_molecules": (
            "output/test001_clustered_molecules.csv"
        ),
    }

    assert [
        call["stage"]
        for call in calls
    ] == [
        "generation",
        "properties",
        "fingerprints",
        "clustering",
    ]

    assert all(
        call["dataset_id"] == "test001"
        for call in calls
    )

    assert all(
        call["replace"] is True
        for call in calls
    )

    assert calls[0]["input_prefix"] == "input"
    assert calls[0]["output_prefix"] == "output"
    assert calls[3]["n_clusters"] == "5"


def test_process_complete_s3_dataset_passes_overwrite_false(
    monkeypatch,
):
    replace_values: list[bool] = []

    def fake_stage(**kwargs):
        replace_values.append(
            kwargs["replace"]
        )
        return "output/result.csv"

    monkeypatch.setattr(
        storage_pipeline,
        "process_s3_dataset",
        fake_stage,
    )
    monkeypatch.setattr(
        storage_pipeline,
        "process_properties_s3_dataset",
        fake_stage,
    )
    monkeypatch.setattr(
        storage_pipeline,
        "process_fingerprints_s3_dataset",
        fake_stage,
    )
    monkeypatch.setattr(
        storage_pipeline,
        "process_clustering_s3_dataset",
        fake_stage,
    )

    storage_pipeline.process_complete_s3_dataset(
        dataset_id="test001",
        object_exists=lambda *args: True,
        download_object=lambda *args: b"",
        upload_bytes=lambda *args: None,
        overwrite=False,
    )

    assert replace_values == [
        False,
        False,
        False,
        False,
    ]

