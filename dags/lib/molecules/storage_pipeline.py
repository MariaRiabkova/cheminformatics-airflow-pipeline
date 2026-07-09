from __future__ import annotations

import os
from collections.abc import Callable

from lib.molecules.pipeline import (
    generate_molecules_from_csv_bytes,
)
from lib.molecules.smiles_parser import (
    dataframe_to_csv_bytes,
)


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


ObjectExists = Callable[[str, str, str], bool]
DownloadObject = Callable[[str, str, str], bytes]
UploadBytes = Callable[[bytes, str, str, str, bool], None]


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


def build_dataset_keys(
    dataset_id: str,
    input_prefix: str = INPUT_PREFIX,
    output_prefix: str = OUTPUT_PREFIX,
) -> tuple[str, str, str]:
    """Build scaffold, R-group, and output S3 keys."""
    normalized_dataset_id = normalize_dataset_id(
        dataset_id
    )

    normalized_input_prefix = input_prefix.strip("/")
    normalized_output_prefix = output_prefix.strip("/")

    scaffolds_filename = (
        f"{normalized_dataset_id}_scaffolds.csv"
    )

    r_groups_filename = (
        f"{normalized_dataset_id}_r_groups.csv"
    )

    output_filename = (
        f"{normalized_dataset_id}"
        "_generated_molecules.csv"
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

    output_key = (
        f"{normalized_output_prefix}/{output_filename}"
        if normalized_output_prefix
        else output_filename
    )

    return (
        scaffolds_key,
        r_groups_key,
        output_key,
    )


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
            f"Required S3 object does not exist: "
            f"s3://{bucket_name}/{key}"
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
    """Process one scaffold and R-group dataset stored in S3."""
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

    output_raw = dataframe_to_csv_bytes(
        generated_dataframe
    )

    upload_bytes(
        output_raw,
        output_key,
        bucket_name,
        aws_conn_id,
        replace,
    )

    return output_key
