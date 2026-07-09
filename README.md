# Feature: Molecule Generation — Step 1

Branch: `feature/molecule-generation-step1`

## Goal

Implement the first iteration of the cheminformatics pipeline:

1. Accept a `dataset_id`.
2. Read two input files with the same dataset id:
   - `<dataset_id>_scaffolds.csv`
   - `<dataset_id>_r_groups.csv`
3. Generate molecules from all scaffold × R-group combinations.
4. Validate generated molecules before they are passed to downstream property calculation.

## Input format

Both files must contain exactly one column named `smiles`.

Example scaffold file:

```csv
smiles
c1ccccc1[*:1]
CC*
```

Example R-group file:

```csv
smiles
[*:1]C
*Cl
```

## Supported attachment-point formats

This iteration supports exactly one attachment point per input molecule.

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
- multiple attachment points are not supported in this iteration.

Examples that must fail:

```text
CC[*:1] + [*:2]C
C([*:1])([*:2]) + [*:1]C
```

## Output validation contract

RDKit parseability alone is not considered sufficient validation.

A generated molecule is valid only if it:

- passes RDKit sanitization;
- contains no dummy atoms;
- contains no unresolved attachment points;
- contains no atom-map numbers;
- contains exactly one connected fragment;
- can be converted to canonical SMILES;
- can be parsed again from the canonical SMILES.

Examples that must be rejected:

```text
C*C
CC*
CC.CC
```

Only validated molecules may be used by downstream steps such as:

- molecular property calculation;
- fingerprint generation;
- clustering;
- prediction.

## Current implementation status

Completed:

- [x] Create `dags/lib/molecules/generation.py`
- [x] Implement single scaffold + single R-group generation
- [x] Validate scaffold and R-group attachment points
- [x] Reject mismatched attachment-point numbers
- [x] Add unit tests for basic generation
- [x] Commit molecule generation core

In progress:

- [ ] Add explicit output validation
- [ ] Add tests for unresolved dummy atoms
- [ ] Add tests for disconnected fragments
- [ ] Add tests for canonical SMILES round-trip validation

Next:

- [ ] Generate all scaffold × R-group combinations from DataFrames
- [ ] Read input CSV files from MinIO
- [ ] Write generated molecules back to MinIO
- [ ] Add a thin Airflow DAG wrapper with `dataset_id` parameter
- [ ] Run end-to-end test with `test001`
- [ ] Open pull request: `feature/molecule-generation-step1` → `dev`

## Development order

```text
1. Core molecule generation
2. Input validation
3. Output validation
4. DataFrame-level generation
5. MinIO integration
6. Airflow DAG wrapper
7. End-to-end test
8. Pull request to dev
9. Promotion from dev to master
```

## Local test command

```powershell
python -m pytest tests\test_generation.py -v
```

## Branch policy

```text
feature/* -> dev -> master
```

Direct commits to `dev` and `master` should be avoided.
