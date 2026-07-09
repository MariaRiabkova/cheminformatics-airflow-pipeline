from __future__ import annotations

import logging
import os
from datetime import timedelta

from airflow.providers.standard.operators.empty import EmptyOperator
from airflow.providers.standard.operators.python import PythonOperator
from airflow.sdk import DAG, Param

from lib.molecules.storage_pipeline import (
    process_clustering_s3_dataset,
    process_fingerprints_s3_dataset,
    process_properties_s3_dataset,
    process_s3_dataset,
)
from lib.utils.s3 import (
    download_object,
    object_exists,
    upload_bytes,
)


logger = logging.getLogger(__name__)

DEFAULT_N_CLUSTERS = int(
    os.getenv(
        "MOLECULES_DEFAULT_N_CLUSTERS",
        "5",
    )
)


def generate_molecules(
    dataset_id: str,
) -> str:
    """Generate molecules for one dataset stored in S3-compatible storage."""
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


def calculate_properties(
    dataset_id: str,
) -> str:
    """Calculate molecular properties for one generated dataset."""
    logger.info(
        "Starting molecular properties calculation for dataset_id=%s",
        dataset_id,
    )

    output_key = process_properties_s3_dataset(
        dataset_id=dataset_id,
        object_exists=object_exists,
        download_object=download_object,
        upload_bytes=upload_bytes,
        replace=True,
    )

    logger.info(
        "Molecular properties calculation completed for dataset_id=%s. "
        "Output key: %s",
        dataset_id,
        output_key,
    )

    return output_key


def calculate_fingerprints(
    dataset_id: str,
) -> str:
    """Calculate ECFP4 fingerprints for one molecular dataset."""
    logger.info(
        "Starting fingerprint calculation for dataset_id=%s",
        dataset_id,
    )

    output_key = process_fingerprints_s3_dataset(
        dataset_id=dataset_id,
        object_exists=object_exists,
        download_object=download_object,
        upload_bytes=upload_bytes,
        replace=True,
    )

    logger.info(
        "Fingerprint calculation completed for dataset_id=%s. "
        "Output key: %s",
        dataset_id,
        output_key,
    )

    return output_key


def cluster_molecules(
    dataset_id: str,
    n_clusters: int | str,
) -> str:
    """Cluster molecules using ECFP4 fingerprints and K-means."""
    logger.info(
        "Starting molecule clustering for dataset_id=%s "
        "with n_clusters=%s",
        dataset_id,
        n_clusters,
    )

    output_key = process_clustering_s3_dataset(
        dataset_id=dataset_id,
        object_exists=object_exists,
        download_object=download_object,
        upload_bytes=upload_bytes,
        n_clusters=n_clusters,
        replace=True,
    )

    logger.info(
        "Molecule clustering completed for dataset_id=%s. "
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
        "molecule_pipeline",
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
        "n_clusters": Param(
            default=DEFAULT_N_CLUSTERS,
            type="integer",
            minimum=2,
            description="Number of K-means clusters.",
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

    calculate_properties_op = PythonOperator(
        task_id="calculate_properties",
        python_callable=calculate_properties,
        op_kwargs={
            "dataset_id": "{{ params.dataset_id }}",
        },
    )

    calculate_fingerprints_op = PythonOperator(
        task_id="calculate_fingerprints",
        python_callable=calculate_fingerprints,
        op_kwargs={
            "dataset_id": "{{ params.dataset_id }}",
        },
    )

    cluster_molecules_op = PythonOperator(
        task_id="cluster_molecules",
        python_callable=cluster_molecules,
        op_kwargs={
            "dataset_id": "{{ params.dataset_id }}",
            "n_clusters": "{{ params.n_clusters }}",
        },
    )

    finish_op = EmptyOperator(
        task_id="finish",
    )

    (
        start_op
        >> generate_molecules_op
        >> calculate_properties_op
        >> calculate_fingerprints_op
        >> cluster_molecules_op
        >> finish_op
    )
