# Cheminformatics Airflow Pipeline

Current branch: `feature/schedule-processing-step2`

## Goal

Implement an Airflow pipeline for weekly cheminformatics processing of scaffold and R-group datasets stored in S3-compatible object storage.

The pipeline:

1. discovers new input files added or updated since the previous scheduled run;
2. identifies complete dataset pairs;
3. generates molecules for every scaffold × R-group combination;
4. validates generated molecules with RDKit;
5. calculates molecular properties;
6. creates ECFP4 fingerprints;
7. clusters molecules with K-means;
8. writes every processing stage to S3-compatible storage.

## Step 2 requirements

The second iteration adds:

- a weekly Airflow schedule;
- processing of new S3 files since the previous launch;
- an `overwrite` DAG parameter with the default value `False`;
- reprocessing and replacement of existing output files when `overwrite=True`.

## Weekly schedule

The DAG runs every Monday at `03:00` in the `Asia/Yerevan` timezone.

```python
schedule="0 3 * * 1"
```

The timezone is set explicitly in the DAG:

```python
DAG_TIMEZONE = pendulum.timezone("Asia/Yerevan")
```

The DAG uses:

```python
catchup=False
```

Therefore, Airflow does not create historical weekly runs before the DAG is enabled.

## Detection of new files

For every object under the input prefix, the S3 helper returns:

```text
key
last_modified
```

Airflow provides the current scheduled interval:

```text
data_interval_start
data_interval_end
```

A file is considered new or updated for the current run when:

```python
data_interval_start <= last_modified < data_interval_end
```

For a weekly run at Monday `03:00 Asia/Yerevan`, the interval normally represents:

```text
previous Monday 03:00
≤ LastModified <
current Monday 03:00
```

The discovery logic then:

1. lists all objects under `input/`;
2. selects files changed during the current Airflow data interval;
3. extracts `dataset_id` values;
4. verifies that both required input files currently exist;
5. excludes already completed datasets when `overwrite=False`;
6. returns the datasets that must be processed.

A dataset is complete only when both files exist:

```text
input/<dataset_id>_scaffolds.csv
input/<dataset_id>_r_groups.csv
```

If one file was uploaded earlier and the second file appeared during the current interval, the dataset is processed because the pair is now complete and at least one input file changed during the interval.

## Overwrite behavior

### `overwrite=False`

Default behavior:

```json
{
  "overwrite": false
}
```

The DAG:

- processes only complete datasets changed during the current data interval;
- skips datasets that already have the final clustered output;
- writes new output objects without replacing existing ones.

A dataset is treated as already completed when this object exists:

```text
output/<dataset_id>_clustered_molecules.csv
```

### `overwrite=True`

Manual reprocessing behavior:

```json
{
  "overwrite": true
}
```

The DAG:

- ignores the `LastModified` interval filter;
- processes every complete dataset pair under `input/`;
- reprocesses datasets that already have outputs;
- passes `replace=True` to every S3 upload;
- overwrites all existing stage outputs.

## Input files

For a dataset with:

```text
dataset_id = test001
```

the pipeline expects:

```text
bronze/input/test001_scaffolds.csv
bronze/input/test001_r_groups.csv
```

The bucket and prefixes are configurable through environment variables.

## Input format

Expected CSV format:

```csv
smiles
CCC*
...
```

The parser supports:

- a column named `smiles`;
- common SMILES column aliases;
- a single column with another name;
- a headerless single-column CSV.

Files with multiple columns and no recognized SMILES column are rejected.

## Supported attachment-point formats

This iteration supports exactly one attachment point in each input molecule.

Supported examples:

```text
CC* + *C
CC[*:1] + [*:1]C
```

Rules:

- each scaffold must contain exactly one dummy atom;
- each R-group must contain exactly one dummy atom;
- the dummy atom must have exactly one neighbour;
- both attachment points must either be unnumbered or use the same atom-map number;
- multiple attachment points are rejected.

Examples that fail:

```text
CC[*:1] + [*:2]C
C([*:1])([*:2]) + [*:1]C
```

## Molecule generation

For one scaffold and one R-group, the pipeline:

1. parses both SMILES with RDKit;
2. locates the attachment atoms;
3. validates attachment-point compatibility;
4. combines the RDKit molecules;
5. creates a bond between attachment neighbours;
6. removes dummy atoms;
7. validates the generated molecule;
8. returns canonical SMILES.

For multiple rows, the full Cartesian product is generated.

Example:

```text
10 scaffolds × 10 R-groups = 100 generated molecules
```

## Output validation contract

A generated molecule is valid only if it:

- passes RDKit sanitization;
- contains no dummy atoms;
- contains no unresolved attachment points;
- contains no atom-map numbers;
- contains exactly one connected fragment;
- can be converted to canonical SMILES;
- can be parsed again from canonical SMILES.

Invalid molecules fail the task instead of being silently skipped.

## Processing stages

### 1. Generated molecules

Output:

```text
output/<dataset_id>_generated_molecules.csv
```

Columns:

```text
scaffold_id
r_group_id
scaffold_smiles
r_group_smiles
generated_smiles
```

### 2. Molecular properties

Input:

```text
output/<dataset_id>_generated_molecules.csv
```

Output:

```text
output/<dataset_id>_molecular_properties.csv
```

Added columns:

```text
canonical_smiles
mol_formula
mol_weight
log_p
tpsa
hba
hbd
rotatable_bonds
aromatic_rings
heavy_atom_count
hetero_atom_count
ring_count
fraction_csp3
formal_charge
qed
lipinski_pass
```

### 3. ECFP4 fingerprints

Input:

```text
output/<dataset_id>_molecular_properties.csv
```

Output:

```text
output/<dataset_id>_fingerprints.csv
```

Configuration:

```text
fingerprint type: ECFP4
Morgan radius: 2
fingerprint size: 2048 bits
```

Added columns:

```text
fingerprint_type
fingerprint
fingerprint_on_bits
```

### 4. K-means clustering

Input:

```text
output/<dataset_id>_fingerprints.csv
```

Output:

```text
output/<dataset_id>_clustered_molecules.csv
```

Added columns:

```text
cluster_id
distance_to_centroid
```

Default configuration:

```text
n_clusters = 5
random_state = 42
n_init = 10
```

## Airflow DAG

DAG ID:

```text
molecule_generation_dag
```

Task flow:

```text
start
→ discover_datasets
→ process_datasets
→ finish
```

### `discover_datasets`

The task:

- receives `data_interval_start` and `data_interval_end` from Airflow;
- lists S3 input and output objects;
- detects changed input datasets;
- validates complete input pairs;
- applies `overwrite` behavior;
- returns a list of dataset IDs through XCom.

### `process_datasets`

The task processes every discovered dataset through the complete pipeline:

```text
molecule generation
→ molecular properties
→ ECFP4 fingerprints
→ K-means clustering
```

When no datasets require processing, the task returns an empty list and the DAG finishes successfully.

## DAG parameters

Default parameters:

```json
{
  "overwrite": false,
  "n_clusters": 5
}
```

`dataset_id` is no longer supplied manually in Step 2. Dataset IDs are discovered from S3 object names.

## Output files

For `test001`, the complete pipeline writes:

```text
bronze/output/test001_generated_molecules.csv
bronze/output/test001_molecular_properties.csv
bronze/output/test001_fingerprints.csv
bronze/output/test001_clustered_molecules.csv
```

## Project structure

```text
dags/
├── molecule_generation_dag.py
└── lib/
    ├── molecules/
    │   ├── __init__.py
    │   ├── clustering.py
    │   ├── dataset_discovery.py
    │   ├── fingerprints.py
    │   ├── generation.py
    │   ├── pipeline.py
    │   ├── properties.py
    │   ├── smiles_parser.py
    │   └── storage_pipeline.py
    └── utils/
        └── s3.py

tests/
├── conftest.py
├── fixture/
│   ├── input/
│   │   ├── test001_scaffolds.csv
│   │   └── test001_r_groups.csv
│   └── output/
│       └── test001_generated_molecules.csv
├── test_clustering.py
├── test_dataset_discovery.py
├── test_fingerprints.py
├── test_generation.py
├── test_pipeline.py
├── test_properties.py
└── test_storage_pipeline.py
```

## Module responsibilities

### `dataset_discovery.py`

Contains pure discovery logic:

```text
S3 object metadata
+ Airflow data interval
+ complete input pairs
+ overwrite policy
→ dataset IDs to process
```

The module has no Airflow or S3 client dependency and can be unit tested locally.

### `storage_pipeline.py`

Contains S3 storage orchestration:

```text
list input/output objects
→ discover datasets
→ build S3 keys
→ download inputs
→ run processing stages
→ upload outputs
```

S3 functions are injected into this layer so the orchestration can be tested without a live Airflow or MinIO environment.

### `s3.py`

Contains generic helpers for S3-compatible storage:

- upload bytes;
- download objects;
- check object existence;
- list object keys;
- list objects with `LastModified` metadata.

Object listing uses an S3 paginator and therefore supports more than one response page.

### `molecule_generation_dag.py`

Contains the Airflow wrapper:

- weekly schedule;
- explicit `Asia/Yerevan` timezone;
- Airflow parameters;
- Airflow data interval handling;
- S3 discovery task;
- complete dataset-processing task.

Cheminformatics business logic remains outside the DAG.

## Environment variables

```env
MINIO_BUCKET=bronze

MOLECULES_AWS_CONN_ID=aws_s3
MOLECULES_INPUT_PREFIX=input
MOLECULES_OUTPUT_PREFIX=output
MOLECULES_DEFAULT_N_CLUSTERS=5
```

The Airflow connection is supplied through:

```text
AIRFLOW_CONN_AWS_S3
```

The MinIO endpoint inside the Docker network is:

```text
http://storage:9000
```

Credentials must not be committed to Git.

## Dependencies

Main project dependencies:

```text
apache-airflow-providers-postgres>=6.0.0
apache-airflow-providers-amazon>=9.0.0
pandas>=2.2.0
rdkit>=2023.9.1
pandera>=0.20.0
soda-core-postgres>=3.3.0
scikit-learn>=1.5.0
pendulum
```

After dependency changes:

```powershell
docker compose build
docker compose up -d
```

## Tests

Run Step 2 tests:

```powershell
python -m pytest `
    tests/test_dataset_discovery.py `
    tests/test_storage_pipeline.py `
    -v
```

Run the complete test suite:

```powershell
python -m pytest tests -v
```

Check syntax:

```powershell
python -m py_compile `
    dags/molecule_generation_dag.py `
    dags/lib/molecules/dataset_discovery.py `
    dags/lib/molecules/storage_pipeline.py `
    dags/lib/utils/s3.py
```

Check Airflow import errors:

```powershell
docker compose exec airflow-scheduler airflow dags list-import-errors
```

List DAGs:

```powershell
docker compose exec airflow-scheduler airflow dags list
```

## Step 2 verification status

The Step 2 implementation has been verified successfully.

- the complete test suite passes;
- the DAG imports without Airflow errors;
- the weekly schedule works;
- S3 objects are filtered by `LastModified`;
- complete scaffold and R-group pairs are detected correctly;
- `overwrite=False` skips completed datasets;
- `overwrite=True` reprocesses all complete datasets and replaces outputs;
- the full DAG runs successfully through Airflow and MinIO.

## Step 2 verification scenarios

### Scenario 1: new dataset, `overwrite=False`

1. Upload a new complete pair:

```text
input/test002_scaffolds.csv
input/test002_r_groups.csv
```

2. Ensure at least one file has `LastModified` within the current Airflow data interval.
3. Run the DAG with:

```json
{
  "overwrite": false,
  "n_clusters": 5
}
```

4. Verify that four output files are created.
5. Run again with `overwrite=False`.
6. Verify that the completed dataset is skipped.

### Scenario 2: full reprocessing, `overwrite=True`

Run the DAG with:

```json
{
  "overwrite": true,
  "n_clusters": 5
}
```

Verify that:

- every complete dataset pair is processed;
- existing outputs are replaced;
- all four stage files receive updated modification timestamps.

## Current implementation status

Completed:

- [x] Molecule generation
- [x] Explicit molecule validation
- [x] CSV parsing and normalization
- [x] Molecular-property calculation
- [x] ECFP4 fingerprint generation
- [x] K-means clustering
- [x] S3-compatible storage pipeline
- [x] Weekly DAG schedule
- [x] Explicit `Asia/Yerevan` timezone
- [x] S3 `LastModified` listing with pagination
- [x] Airflow data-interval filtering
- [x] Complete-pair discovery
- [x] `overwrite=False` behavior
- [x] `overwrite=True` behavior
- [x] Unit tests for discovery and storage orchestration

Validation completed:

- [x] Complete local test suite passed
- [x] Airflow DAG imports without errors
- [x] Weekly DAG schedule is configured
- [x] New S3 datasets are discovered by `LastModified`
- [x] `overwrite=False` behavior works
- [x] `overwrite=True` behavior works
- [x] Step 2 end-to-end verification completed successfully
- [x] DAG works correctly with Airflow and MinIO


## Branch policy

```text
feature/* → dev → master
```

Direct commits to `dev` and `master` should be avoided.
