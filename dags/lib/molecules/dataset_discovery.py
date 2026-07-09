from __future__ import annotations


SCAFFOLDS_SUFFIX = "_scaffolds.csv"
R_GROUPS_SUFFIX = "_r_groups.csv"
FINAL_OUTPUT_SUFFIX = "_clustered_molecules.csv"


def normalize_prefix(
    prefix: str,
) -> str:
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

    # Ignore objects located in nested folders.
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


def select_datasets_to_process(
    complete_dataset_ids: list[str],
    output_keys: list[str],
    overwrite: bool = False,
    output_prefix: str = "output",
) -> list[str]:
    """Select complete datasets that require processing."""
    if not isinstance(
        complete_dataset_ids,
        list,
    ):
        raise TypeError(
            "Dataset IDs must be provided as a list"
        )

    if not isinstance(overwrite, bool):
        raise TypeError(
            "Overwrite must be a boolean"
        )

    normalized_dataset_ids = sorted(
        set(complete_dataset_ids)
    )

    if overwrite:
        return normalized_dataset_ids

    processed_datasets = (
        discover_processed_datasets(
            output_keys=output_keys,
            output_prefix=output_prefix,
        )
    )

    return [
        dataset_id
        for dataset_id in normalized_dataset_ids
        if dataset_id not in processed_datasets
    ]


def discover_datasets_to_process(
    input_keys: list[str],
    output_keys: list[str],
    overwrite: bool = False,
    input_prefix: str = "input",
    output_prefix: str = "output",
) -> list[str]:
    """Discover complete datasets that should be processed."""
    complete_dataset_ids = (
        discover_complete_datasets(
            input_keys=input_keys,
            input_prefix=input_prefix,
        )
    )

    return select_datasets_to_process(
        complete_dataset_ids=complete_dataset_ids,
        output_keys=output_keys,
        overwrite=overwrite,
        output_prefix=output_prefix,
    )
