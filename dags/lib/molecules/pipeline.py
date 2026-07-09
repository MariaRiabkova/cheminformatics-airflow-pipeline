from __future__ import annotations

from itertools import product

import pandas as pd

from dags.lib.molecules.generation import generate_molecule
from dags.lib.molecules.smiles_parser import parse_smiles_csv


SMILES_COLUMN = "smiles"

OUTPUT_COLUMNS = [
    "scaffold_id",
    "r_group_id",
    "scaffold_smiles",
    "r_group_smiles",
    "generated_smiles",
]


def generate_molecules_dataframe(
    scaffolds: pd.DataFrame,
    r_groups: pd.DataFrame,
) -> pd.DataFrame:
    """Generate molecules for every scaffold and R-group combination."""
    if SMILES_COLUMN not in scaffolds.columns:
        raise ValueError(
            f"Scaffold DataFrame must contain the "
            f"'{SMILES_COLUMN}' column"
        )

    if SMILES_COLUMN not in r_groups.columns:
        raise ValueError(
            f"R-group DataFrame must contain the "
            f"'{SMILES_COLUMN}' column"
        )

    if scaffolds.empty:
        raise ValueError(
            "Scaffold DataFrame must not be empty"
        )

    if r_groups.empty:
        raise ValueError(
            "R-group DataFrame must not be empty"
        )

    scaffold_values = scaffolds[SMILES_COLUMN].tolist()
    r_group_values = r_groups[SMILES_COLUMN].tolist()

    records: list[dict[str, int | str]] = []

    for (
        scaffold_id,
        scaffold_smiles,
    ), (
        r_group_id,
        r_group_smiles,
    ) in product(
        enumerate(scaffold_values),
        enumerate(r_group_values),
    ):
        try:
            generated_smiles = generate_molecule(
                scaffold_smiles=scaffold_smiles,
                r_group_smiles=r_group_smiles,
            )
        except Exception as exc:
            raise ValueError(
                "Failed to generate molecule for "
                f"scaffold_id={scaffold_id}, "
                f"r_group_id={r_group_id}, "
                f"scaffold={scaffold_smiles!r}, "
                f"r_group={r_group_smiles!r}: {exc}"
            ) from exc

        records.append(
            {
                "scaffold_id": scaffold_id,
                "r_group_id": r_group_id,
                "scaffold_smiles": scaffold_smiles,
                "r_group_smiles": r_group_smiles,
                "generated_smiles": generated_smiles,
            }
        )

    return pd.DataFrame.from_records(
        records,
        columns=OUTPUT_COLUMNS,
    )


def generate_molecules_from_csv_bytes(
    scaffolds_raw: bytes,
    r_groups_raw: bytes,
    scaffolds_source_name: str = "scaffolds CSV",
    r_groups_source_name: str = "R-groups CSV",
) -> pd.DataFrame:
    """Parse two CSV files and generate all molecule combinations."""
    scaffolds = parse_smiles_csv(
        raw=scaffolds_raw,
        source_name=scaffolds_source_name,
    )

    r_groups = parse_smiles_csv(
        raw=r_groups_raw,
        source_name=r_groups_source_name,
    )

    return generate_molecules_dataframe(
        scaffolds=scaffolds,
        r_groups=r_groups,
    )

