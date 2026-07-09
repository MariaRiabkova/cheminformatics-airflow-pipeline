from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta

import pendulum
from airflow.providers.standard.operators.empty import EmptyOperator
from airflow.providers.standard.operators.python import PythonOperator
from airflow.sdk import DAG, Param

from lib.molecules.storage_pipeline import (
    discover_s3_datasets_to_process,
    process_complete_s3_dataset,
)
from lib.utils.s3 import (
    download_object,
    list_objects,
    object_exists,
    upload_bytes,
)


logger = logging.getLogger(__name__)

DAG_TIMEZONE = pendulum.timezone("Asia/Yerevan")

DEFAULT_N_CLUSTERS = int(
    os.getenv(
        "MOLECULES_DEFAULT_N_CLUSTERS",
        "5",
    )
)


def normalize_interval_datetime(
    value: datetime | str,
) -> datetime:
    """Convert an Airflow interval value to a timezone-aware datetime."""
    if isinstance(value, datetime):
        return value

    if isinstance(value, str):
        parsed_value = pendulum.parse(value)

        if parsed_value.tzinfo is None:
            return DAG_TIMEZONE.convert(parsed_value)

        return parsed_value

    raise TypeError(
        "Data interval value must be a datetime or ISO datetime string"
    )


def discover_datasets(
    overwrite: bool,
    data_interval_start: datetime | str,
    data_interval_end: datetime | str,
) -> list[str]:
    """Discover complete S3 datasets that require processing."""
    interval_start = normalize_interval_datetime(
        data_interval_start
    )
    interval_end = normalize_interval_datetime(
        data_interval_end
    )

    logger.info(
        "Starting dataset discovery. overwrite=%s, "
        "data_interval_start=%s, data_interval_end=%s",
        overwrite,
        interval_start,
        interval_end,
    )

    dataset_ids = discover_s3_datasets_to_process(
        list_objects=list_objects,
        interval_start=interval_start,
        interval_end=interval_end,
        overwrite=overwrite,
    )

    logger.info(
        "Dataset discovery completed. Found %s dataset(s): %s",
        len(dataset_ids),
        dataset_ids,
    )

    return dataset_ids


def process_datasets(
    dataset_ids: list[str],
    overwrite: bool,
    n_clusters: int,
) -> list[dict[str, str]]:
    """Process every dataset returned by the discovery task."""
    if not dataset_ids:
        logger.info(
            "No new complete datasets were found"
        )
        return []

    results: list[dict[str, str]] = []

    for dataset_id in dataset_ids:
        logger.info(
            "Starting complete processing for dataset_id=%s",
            dataset_id,
        )

        result = process_complete_s3_dataset(
            dataset_id=dataset_id,
            object_exists=object_exists,
            download_object=download_object,
            upload_bytes=upload_bytes,
            overwrite=overwrite,
            n_clusters=n_clusters,
        )

        results.append(result)

        logger.info(
            "Complete processing finished for dataset_id=%s. "
            "Outputs: %s",
            dataset_id,
            result,
        )

    logger.info(
        "All discovered datasets were processed. "
        "Processed dataset count: %s",
        len(results),
    )

    return results


with DAG(
    dag_id="molecule_generation_dag",
    schedule="0 3 * * 1",
    start_date=pendulum.datetime(
        2026,
        7,
        13,
        3,
        0,
        tz=DAG_TIMEZONE,
    ),
    catchup=False,
    render_template_as_native_obj=True,
    tags=[
        "cheminformatics",
        "de_school",
        "molecule_pipeline",
        "weekly",
    ],
    params={
        "overwrite": Param(
            default=False,
            type="boolean",
            description=(
                "Reprocess all complete datasets and overwrite "
                "existing output files."
            ),
        ),
        "n_clusters": Param(
            default=DEFAULT_N_CLUSTERS,
            type="integer",
            minimum=2,
            description="Number of K-means clusters.",
        ),
    },
    dagrun_timeout=timedelta(minutes=60),
    default_args={
        "owner": "data-platform",
        "retries": 1,
        "retry_delay": timedelta(minutes=1),
        "retry_exponential_backoff": True,
        "max_retry_delay": timedelta(minutes=10),
    },
) as dag:
    start_op = EmptyOperator(
        task_id="start",
    )

    discover_datasets_op = PythonOperator(
        task_id="discover_datasets",
        python_callable=discover_datasets,
        op_kwargs={
            "overwrite": "{{ params.overwrite }}",
            "data_interval_start": "{{ data_interval_start }}",
            "data_interval_end": "{{ data_interval_end }}",
        },
    )

    process_datasets_op = PythonOperator(
        task_id="process_datasets",
        python_callable=process_datasets,
        op_kwargs={
            "dataset_ids": (
                "{{ ti.xcom_pull(task_ids='discover_datasets') }}"
            ),
            "overwrite": "{{ params.overwrite }}",
            "n_clusters": "{{ params.n_clusters }}",
        },
    )

    finish_op = EmptyOperator(
        task_id="finish",
    )

    (
        start_op
        >> discover_datasets_op
        >> process_datasets_op
        >> finish_op
    )
