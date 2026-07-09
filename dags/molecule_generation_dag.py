from __future__ import annotations

import logging
from datetime import timedelta

from airflow.providers.standard.operators.empty import EmptyOperator
from airflow.providers.standard.operators.python import PythonOperator
from airflow.sdk import DAG, Param

from lib.molecules.storage_pipeline import (
    process_s3_dataset,
)
from lib.utils.s3 import (
    download_object,
    object_exists,
    upload_bytes,
)


logger = logging.getLogger(__name__)


def generate_molecules(
    dataset_id: str,
) -> str:
    """
    Generate molecules for one dataset stored in S3-compatible storage.
    """
    logger.info(
        "Starting molecule generation for dataset_id=%s",
        dataset_id,
    )

    output_key = process_s3_dataset(
        dataset_id=dataset_id,
        object_exists=object_exists,
        download_object=download_object,
        upload_bytes=upload_bytes,
        replace=True,
    )

    logger.info(
        "Molecule generation completed for dataset_id=%s. "
        "Output key: %s",
        dataset_id,
        output_key,
    )

    return output_key


with DAG(
    dag_id="molecule_generation_dag",
    schedule=None,
    start_date=None,
    catchup=False,
    tags=[
        "cheminformatics",
        "de_school",
        "step1",
    ],
    params={
        "dataset_id": Param(
            default="test001",
            type="string",
            minLength=1,
            description=(
                "Dataset identifier used to locate matching "
                "scaffold and R-group CSV files."
            ),
        ),
    },
    dagrun_timeout=timedelta(minutes=30),
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

    generate_molecules_op = PythonOperator(
        task_id="generate_molecules",
        python_callable=generate_molecules,
        op_kwargs={
            "dataset_id": "{{ params.dataset_id }}",
        },
    )

    finish_op = EmptyOperator(
        task_id="finish",
    )

    start_op >> generate_molecules_op >> finish_op

