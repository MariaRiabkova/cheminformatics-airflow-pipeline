from __future__ import annotations

import os
from collections.abc import Callable

import numpy as np
import pandas as pd

from lib.molecules.clustering import cluster_fingerprints
from lib.molecules.fingerprints import calculate_fingerprints_dataframe
from lib.molecules.pipeline import generate_molecules_from_csv_bytes
from lib.molecules.properties import calculate_properties_dataframe
from lib.molecules.smiles_parser import dataframe_to_csv_bytes, read_csv_bytes
from lib.molecules.dataset_discovery import discover_datasets_to_process


BUCKET_NAME = os.getenv(
    "MOLECULES_BUCKET_NAME",
    "bronze",
)

AWS_CONN_ID = os.getenv(
    "MOLECULES_AWS_CONN_ID",
    "aws_s3",
)

INPUT_PREFIX = os.getenv(
    "MOLECULES_INPUT_PREFIX",
    "input",
).strip("/")

OUTPUT_PREFIX = os.getenv(
    "MOLECULES_OUTPUT_PREFIX",
    "output",
).strip("/")

DEFAULT_N_CLUSTERS = int(
    os.getenv(
        "MOLECULES_DEFAULT_N_CLUSTERS",
        "5",
    )
)


ObjectExists = Callable[[str, str, str], bool]
DownloadObject = Callable[[str, str, str], bytes]
UploadBytes = Callable[[bytes, str, str, str, bool], None]
ListObjectKeys = Callable[[str, str, str], list[str]]


def normalize_dataset_id(
    dataset_id: str,
) -> str:
    """Validate and normalize a dataset identifier."""
    if not isinstance(dataset_id, str):
        raise TypeError(
            "Dataset ID must be a string"
        )

    normalized_dataset_id = dataset_id.strip()

    if not normalized_dataset_id:
        raise ValueError(
            "Dataset ID must not be blank"
        )

    if "/" in normalized_dataset_id:
        raise ValueError(
            "Dataset ID must not contain '/'"
        )

    if "\\" in normalized_dataset_id:
        raise ValueError(
            "Dataset ID must not contain '\\'"
        )

    return normalized_dataset_id


def normalize_n_clusters(
    n_clusters: int | str,
) -> int:
    """Validate and normalize the number of clusters."""
    if isinstance(n_clusters, bool):
        raise TypeError(
            "Number of clusters must be an integer"
        )

    try:
        normalized_n_clusters = int(n_clusters)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "Number of clusters must be an integer"
        ) from exc

    if normalized_n_clusters < 2:
        raise ValueError(
            "Number of clusters must be at least 2"
        )

    return normalized_n_clusters


def build_output_key(
    dataset_id: str,
    suffix: str,
    output_prefix: str = OUTPUT_PREFIX,
) -> str:
    """Build one output S3 key for a dataset."""
    normalized_dataset_id = normalize_dataset_id(
        dataset_id
    )

    if not isinstance(suffix, str):
        raise TypeError(
            "Output suffix must be a string"
        )

    normalized_suffix = suffix.strip().strip("_")

    if not normalized_suffix:
        raise ValueError(
            "Output suffix must not be blank"
        )

    normalized_output_prefix = output_prefix.strip("/")

    filename = (
        f"{normalized_dataset_id}_{normalized_suffix}.csv"
    )

    if normalized_output_prefix:
        return (
            f"{normalized_output_prefix}/{filename}"
        )

    return filename


def build_dataset_keys(
    dataset_id: str,
    input_prefix: str = INPUT_PREFIX,
    output_prefix: str = OUTPUT_PREFIX,
) -> tuple[str, str, str]:
    """Build scaffold, R-group, and generated-molecule S3 keys."""
    normalized_dataset_id = normalize_dataset_id(
        dataset_id
    )

    normalized_input_prefix = input_prefix.strip("/")

    scaffolds_filename = (
        f"{normalized_dataset_id}_scaffolds.csv"
    )
    r_groups_filename = (
        f"{normalized_dataset_id}_r_groups.csv"
    )

    scaffolds_key = (
        f"{normalized_input_prefix}/{scaffolds_filename}"
        if normalized_input_prefix
        else scaffolds_filename
    )

    r_groups_key = (
        f"{normalized_input_prefix}/{r_groups_filename}"
        if normalized_input_prefix
        else r_groups_filename
    )

    output_key = build_output_key(
        dataset_id=normalized_dataset_id,
        suffix="generated_molecules",
        output_prefix=output_prefix,
    )

    return (
        scaffolds_key,
        r_groups_key,
        output_key,
    )


def build_properties_keys(
    dataset_id: str,
    output_prefix: str = OUTPUT_PREFIX,
) -> tuple[str, str]:
    """Build molecular-properties input and output S3 keys."""
    input_key = build_output_key(
        dataset_id=dataset_id,
        suffix="generated_molecules",
        output_prefix=output_prefix,
    )

    output_key = build_output_key(
        dataset_id=dataset_id,
        suffix="molecular_properties",
        output_prefix=output_prefix,
    )

    return input_key, output_key


def build_fingerprints_keys(
    dataset_id: str,
    output_prefix: str = OUTPUT_PREFIX,
) -> tuple[str, str]:
    """Build fingerprint input and output S3 keys."""
    input_key = build_output_key(
        dataset_id=dataset_id,
        suffix="molecular_properties",
        output_prefix=output_prefix,
    )

    output_key = build_output_key(
        dataset_id=dataset_id,
        suffix="fingerprints",
        output_prefix=output_prefix,
    )

    return input_key, output_key


def build_clustering_keys(
    dataset_id: str,
    output_prefix: str = OUTPUT_PREFIX,
) -> tuple[str, str]:
    """Build clustering input and output S3 keys."""
    input_key = build_output_key(
        dataset_id=dataset_id,
        suffix="fingerprints",
        output_prefix=output_prefix,
    )

    output_key = build_output_key(
        dataset_id=dataset_id,
        suffix="clustered_molecules",
        output_prefix=output_prefix,
    )

    return input_key, output_key


def require_s3_object(
    key: str,
    bucket_name: str,
    aws_conn_id: str,
    object_exists: ObjectExists,
) -> None:
    """Raise an error if a required S3 object does not exist."""
    if not object_exists(
        key,
        bucket_name,
        aws_conn_id,
    ):
        raise FileNotFoundError(
            "Required S3 object does not exist: "
            f"s3://{bucket_name}/{key}"
        )


def download_csv_dataframe(
    key: str,
    bucket_name: str,
    aws_conn_id: str,
    object_exists: ObjectExists,
    download_object: DownloadObject,
) -> pd.DataFrame:
    """Download and parse one CSV object from S3."""
    require_s3_object(
        key=key,
        bucket_name=bucket_name,
        aws_conn_id=aws_conn_id,
        object_exists=object_exists,
    )

    raw = download_object(
        key,
        bucket_name,
        aws_conn_id,
    )

    return read_csv_bytes(
        raw=raw,
        source_name=f"s3://{bucket_name}/{key}",
    )


def upload_dataframe(
    dataframe: pd.DataFrame,
    key: str,
    bucket_name: str,
    aws_conn_id: str,
    upload_bytes: UploadBytes,
    replace: bool,
) -> None:
    """Serialize and upload one DataFrame to S3."""
    raw = dataframe_to_csv_bytes(
        dataframe
    )

    upload_bytes(
        raw,
        key,
        bucket_name,
        aws_conn_id,
        replace,
    )


def fingerprint_strings_to_matrix(
    dataframe: pd.DataFrame,
    fingerprint_column: str = "fingerprint",
) -> np.ndarray:
    """Convert fingerprint bit strings into a NumPy matrix."""
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError(
            "Expected a pandas DataFrame"
        )

    if fingerprint_column not in dataframe.columns:
        raise ValueError(
            f"Missing required column: {fingerprint_column!r}"
        )

    if dataframe.empty:
        raise ValueError(
            "Fingerprint DataFrame must not be empty"
        )

    fingerprints = (
        dataframe[fingerprint_column]
        .astype(str)
        .str.strip()
    )

    if fingerprints.eq("").any():
        raise ValueError(
            "Fingerprint column contains blank values"
        )

    fingerprint_lengths = fingerprints.str.len()

    if fingerprint_lengths.nunique() != 1:
        raise ValueError(
            "All fingerprints must have the same length"
        )

    invalid_mask = ~fingerprints.str.fullmatch(
        r"[01]+"
    )

    if invalid_mask.any():
        invalid_index = fingerprints.index[
            invalid_mask
        ][0]

        raise ValueError(
            "Fingerprint must contain only '0' and '1' "
            f"at row index {invalid_index}"
        )

    return np.vstack(
        [
            np.fromiter(
                (
                    int(bit)
                    for bit in fingerprint
                ),
                dtype=np.uint8,
                count=len(fingerprint),
            )
            for fingerprint in fingerprints
        ]
    )


def process_s3_dataset(
    dataset_id: str,
    object_exists: ObjectExists,
    download_object: DownloadObject,
    upload_bytes: UploadBytes,
    bucket_name: str = BUCKET_NAME,
    aws_conn_id: str = AWS_CONN_ID,
    input_prefix: str = INPUT_PREFIX,
    output_prefix: str = OUTPUT_PREFIX,
    replace: bool = True,
) -> str:
    """Generate molecules for one dataset stored in S3."""
    (
        scaffolds_key,
        r_groups_key,
        output_key,
    ) = build_dataset_keys(
        dataset_id=dataset_id,
        input_prefix=input_prefix,
        output_prefix=output_prefix,
    )

    require_s3_object(
        key=scaffolds_key,
        bucket_name=bucket_name,
        aws_conn_id=aws_conn_id,
        object_exists=object_exists,
    )

    require_s3_object(
        key=r_groups_key,
        bucket_name=bucket_name,
        aws_conn_id=aws_conn_id,
        object_exists=object_exists,
    )

    scaffolds_raw = download_object(
        scaffolds_key,
        bucket_name,
        aws_conn_id,
    )

    r_groups_raw = download_object(
        r_groups_key,
        bucket_name,
        aws_conn_id,
    )

    generated_dataframe = (
        generate_molecules_from_csv_bytes(
            scaffolds_raw=scaffolds_raw,
            r_groups_raw=r_groups_raw,
            scaffolds_source_name=(
                f"s3://{bucket_name}/{scaffolds_key}"
            ),
            r_groups_source_name=(
                f"s3://{bucket_name}/{r_groups_key}"
            ),
        )
    )

    upload_dataframe(
        dataframe=generated_dataframe,
        key=output_key,
        bucket_name=bucket_name,
        aws_conn_id=aws_conn_id,
        upload_bytes=upload_bytes,
        replace=replace,
    )

    return output_key


def process_properties_s3_dataset(
    dataset_id: str,
    object_exists: ObjectExists,
    download_object: DownloadObject,
    upload_bytes: UploadBytes,
    bucket_name: str = BUCKET_NAME,
    aws_conn_id: str = AWS_CONN_ID,
    output_prefix: str = OUTPUT_PREFIX,
    replace: bool = True,
) -> str:
    """Calculate molecular properties for one S3 dataset."""
    input_key, output_key = build_properties_keys(
        dataset_id=dataset_id,
        output_prefix=output_prefix,
    )

    molecules = download_csv_dataframe(
        key=input_key,
        bucket_name=bucket_name,
        aws_conn_id=aws_conn_id,
        object_exists=object_exists,
        download_object=download_object,
    )

    properties = calculate_properties_dataframe(
        molecules,
        smiles_column="generated_smiles",
    )

    upload_dataframe(
        dataframe=properties,
        key=output_key,
        bucket_name=bucket_name,
        aws_conn_id=aws_conn_id,
        upload_bytes=upload_bytes,
        replace=replace,
    )

    return output_key


def process_fingerprints_s3_dataset(
    dataset_id: str,
    object_exists: ObjectExists,
    download_object: DownloadObject,
    upload_bytes: UploadBytes,
    bucket_name: str = BUCKET_NAME,
    aws_conn_id: str = AWS_CONN_ID,
    output_prefix: str = OUTPUT_PREFIX,
    replace: bool = True,
) -> str:
    """Calculate ECFP4 fingerprints for one S3 dataset."""
    input_key, output_key = build_fingerprints_keys(
        dataset_id=dataset_id,
        output_prefix=output_prefix,
    )

    molecules = download_csv_dataframe(
        key=input_key,
        bucket_name=bucket_name,
        aws_conn_id=aws_conn_id,
        object_exists=object_exists,
        download_object=download_object,
    )

    fingerprint_result = (
        calculate_fingerprints_dataframe(
            molecules,
            smiles_column="canonical_smiles",
        )
    )

    upload_dataframe(
        dataframe=fingerprint_result.dataframe,
        key=output_key,
        bucket_name=bucket_name,
        aws_conn_id=aws_conn_id,
        upload_bytes=upload_bytes,
        replace=replace,
    )

    return output_key


def process_clustering_s3_dataset(
    dataset_id: str,
    object_exists: ObjectExists,
    download_object: DownloadObject,
    upload_bytes: UploadBytes,
    n_clusters: int | str = DEFAULT_N_CLUSTERS,
    bucket_name: str = BUCKET_NAME,
    aws_conn_id: str = AWS_CONN_ID,
    output_prefix: str = OUTPUT_PREFIX,
    replace: bool = True,
) -> str:
    """Cluster molecular fingerprints for one S3 dataset."""
    normalized_n_clusters = normalize_n_clusters(
        n_clusters
    )

    input_key, output_key = build_clustering_keys(
        dataset_id=dataset_id,
        output_prefix=output_prefix,
    )

    molecules = download_csv_dataframe(
        key=input_key,
        bucket_name=bucket_name,
        aws_conn_id=aws_conn_id,
        object_exists=object_exists,
        download_object=download_object,
    )

    fingerprint_matrix = fingerprint_strings_to_matrix(
        dataframe=molecules,
        fingerprint_column="fingerprint",
    )

    clustering_result = cluster_fingerprints(
        molecules,
        fingerprint_matrix,
        n_clusters=normalized_n_clusters,
    )

    upload_dataframe(
        dataframe=clustering_result.dataframe,
        key=output_key,
        bucket_name=bucket_name,
        aws_conn_id=aws_conn_id,
        upload_bytes=upload_bytes,
        replace=replace,
    )

    return output_key

def discover_s3_datasets_to_process(
    list_object_keys: ListObjectKeys,
    overwrite: bool = False,
    bucket_name: str = BUCKET_NAME,
    aws_conn_id: str = AWS_CONN_ID,
    input_prefix: str = INPUT_PREFIX,
    output_prefix: str = OUTPUT_PREFIX,
) -> list[str]:
    """Discover complete S3 datasets that require processing."""
    input_keys = list_object_keys(
        input_prefix,
        bucket_name,
        aws_conn_id,
    )

    output_keys = list_object_keys(
        output_prefix,
        bucket_name,
        aws_conn_id,
    )

    return discover_datasets_to_process(
        input_keys=input_keys,
        output_keys=output_keys,
        overwrite=overwrite,
        input_prefix=input_prefix,
        output_prefix=output_prefix,
    )


def process_complete_s3_dataset(
    dataset_id: str,
    object_exists: ObjectExists,
    download_object: DownloadObject,
    upload_bytes: UploadBytes,
    overwrite: bool = False,
    n_clusters: int | str = DEFAULT_N_CLUSTERS,
    bucket_name: str = BUCKET_NAME,
    aws_conn_id: str = AWS_CONN_ID,
    input_prefix: str = INPUT_PREFIX,
    output_prefix: str = OUTPUT_PREFIX,
) -> dict[str, str]:
    """Run the complete molecular pipeline for one S3 dataset."""
    generated_key = process_s3_dataset(
        dataset_id=dataset_id,
        object_exists=object_exists,
        download_object=download_object,
        upload_bytes=upload_bytes,
        bucket_name=bucket_name,
        aws_conn_id=aws_conn_id,
        input_prefix=input_prefix,
        output_prefix=output_prefix,
        replace=overwrite,
    )

    properties_key = process_properties_s3_dataset(
        dataset_id=dataset_id,
        object_exists=object_exists,
        download_object=download_object,
        upload_bytes=upload_bytes,
        bucket_name=bucket_name,
        aws_conn_id=aws_conn_id,
        output_prefix=output_prefix,
        replace=overwrite,
    )

    fingerprints_key = process_fingerprints_s3_dataset(
        dataset_id=dataset_id,
        object_exists=object_exists,
        download_object=download_object,
        upload_bytes=upload_bytes,
        bucket_name=bucket_name,
        aws_conn_id=aws_conn_id,
        output_prefix=output_prefix,
        replace=overwrite,
    )

    clustered_key = process_clustering_s3_dataset(
        dataset_id=dataset_id,
        object_exists=object_exists,
        download_object=download_object,
        upload_bytes=upload_bytes,
        n_clusters=n_clusters,
        bucket_name=bucket_name,
        aws_conn_id=aws_conn_id,
        output_prefix=output_prefix,
        replace=overwrite,
    )

    return {
        "generated_molecules": generated_key,
        "molecular_properties": properties_key,
        "fingerprints": fingerprints_key,
        "clustered_molecules": clustered_key,
    }

