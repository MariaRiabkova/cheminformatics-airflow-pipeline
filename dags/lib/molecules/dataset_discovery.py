from __future__ import annotations

from datetime import datetime
from typing import TypedDict


SCAFFOLDS_SUFFIX = "_scaffolds.csv"
R_GROUPS_SUFFIX = "_r_groups.csv"
FINAL_OUTPUT_SUFFIX = "_clustered_molecules.csv"


class S3ObjectMetadata(TypedDict):
    """Metadata required for S3 dataset discovery."""

    key: str
    last_modified: datetime


def normalize_prefix(prefix: str) -> str:
    """Normalize an S3 prefix."""
    if not isinstance(prefix, str):
        raise TypeError(
            "S3 prefix must be a string"
        )

    normalized_prefix = prefix.strip("/")

    if normalized_prefix:
        return f"{normalized_prefix}/"

    return ""


def extract_dataset_id(
    key: str,
    prefix: str,
    suffix: str,
) -> str | None:
    """Extract a dataset ID from one matching S3 key."""
    if not isinstance(key, str):
        raise TypeError(
            "S3 object key must be a string"
        )

    normalized_prefix = normalize_prefix(
        prefix
    )

    if not key.startswith(normalized_prefix):
        return None

    filename = key[len(normalized_prefix):]

    if "/" in filename:
        return None

    if not filename.endswith(suffix):
        return None

    dataset_id = filename[:-len(suffix)]

    if not dataset_id:
        return None

    return dataset_id


def discover_complete_datasets(
    input_keys: list[str],
    input_prefix: str = "input",
) -> list[str]:
    """Return dataset IDs with both scaffold and R-group files."""
    if not isinstance(input_keys, list):
        raise TypeError(
            "Input keys must be provided as a list"
        )

    scaffold_datasets: set[str] = set()
    r_group_datasets: set[str] = set()

    for key in input_keys:
        scaffold_dataset_id = extract_dataset_id(
            key=key,
            prefix=input_prefix,
            suffix=SCAFFOLDS_SUFFIX,
        )

        if scaffold_dataset_id is not None:
            scaffold_datasets.add(
                scaffold_dataset_id
            )
            continue

        r_group_dataset_id = extract_dataset_id(
            key=key,
            prefix=input_prefix,
            suffix=R_GROUPS_SUFFIX,
        )

        if r_group_dataset_id is not None:
            r_group_datasets.add(
                r_group_dataset_id
            )

    return sorted(
        scaffold_datasets
        & r_group_datasets
    )


def discover_changed_datasets(
    input_objects: list[S3ObjectMetadata],
    interval_start: datetime,
    interval_end: datetime,
    input_prefix: str = "input",
) -> set[str]:
    """Return dataset IDs changed during the current data interval."""
    if not isinstance(input_objects, list):
        raise TypeError(
            "Input objects must be provided as a list"
        )

    if not isinstance(interval_start, datetime):
        raise TypeError(
            "Interval start must be a datetime"
        )

    if not isinstance(interval_end, datetime):
        raise TypeError(
            "Interval end must be a datetime"
        )

    if interval_start >= interval_end:
        raise ValueError(
            "Interval start must be earlier than interval end"
        )

    changed_datasets: set[str] = set()

    for object_metadata in input_objects:
        key = object_metadata["key"]
        last_modified = object_metadata[
            "last_modified"
        ]

        if not (
            interval_start
            <= last_modified
            < interval_end
        ):
            continue

        for suffix in (
            SCAFFOLDS_SUFFIX,
            R_GROUPS_SUFFIX,
        ):
            dataset_id = extract_dataset_id(
                key=key,
                prefix=input_prefix,
                suffix=suffix,
            )

            if dataset_id is not None:
                changed_datasets.add(
                    dataset_id
                )
                break

    return changed_datasets


def discover_processed_datasets(
    output_keys: list[str],
    output_prefix: str = "output",
) -> set[str]:
    """Return dataset IDs with a final clustered output."""
    if not isinstance(output_keys, list):
        raise TypeError(
            "Output keys must be provided as a list"
        )

    processed_datasets: set[str] = set()

    for key in output_keys:
        dataset_id = extract_dataset_id(
            key=key,
            prefix=output_prefix,
            suffix=FINAL_OUTPUT_SUFFIX,
        )

        if dataset_id is not None:
            processed_datasets.add(
                dataset_id
            )

    return processed_datasets


def discover_datasets_to_process(
    input_objects: list[S3ObjectMetadata],
    output_keys: list[str],
    interval_start: datetime,
    interval_end: datetime,
    overwrite: bool = False,
    input_prefix: str = "input",
    output_prefix: str = "output",
) -> list[str]:
    """Discover complete datasets that should be processed."""
    if not isinstance(overwrite, bool):
        raise TypeError(
            "Overwrite must be a boolean"
        )

    input_keys = [
        object_metadata["key"]
        for object_metadata in input_objects
    ]

    complete_datasets = set(
        discover_complete_datasets(
            input_keys=input_keys,
            input_prefix=input_prefix,
        )
    )

    if overwrite:
        return sorted(
            complete_datasets
        )

    changed_datasets = discover_changed_datasets(
        input_objects=input_objects,
        interval_start=interval_start,
        interval_end=interval_end,
        input_prefix=input_prefix,
    )

    processed_datasets = discover_processed_datasets(
        output_keys=output_keys,
        output_prefix=output_prefix,
    )

    return sorted(
        complete_datasets
        & changed_datasets
        - processed_datasets
    )
