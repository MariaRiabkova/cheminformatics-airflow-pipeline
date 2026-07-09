from __future__ import annotations

from io import BytesIO

import pandas as pd
from rdkit import Chem


STANDARD_SMILES_COLUMN = "smiles"

SMILES_COLUMN_ALIASES = {
    "smiles",
    "smile",
    "canonical_smiles",
    "canonical smiles",
}


def normalize_column_name(column_name: object) -> str:
    """Normalize a column name for comparison."""
    return (
        str(column_name)
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
    )


def is_attachment_smiles(value: object) -> bool:
    """Check whether a value is a valid SMILES with one attachment point."""
    smiles = str(value).strip()

    if not smiles:
        return False

    molecule = Chem.MolFromSmiles(smiles)

    if molecule is None:
        return False

    attachment_atoms = [
        atom
        for atom in molecule.GetAtoms()
        if atom.GetAtomicNum() == 0
    ]

    if len(attachment_atoms) != 1:
        return False

    return len(attachment_atoms[0].GetNeighbors()) == 1


def read_csv_bytes(
    raw: bytes,
    source_name: str,
    header: int | None = 0,
) -> pd.DataFrame:
    """Read CSV bytes into a string-valued DataFrame."""
    if not isinstance(raw, bytes):
        raise TypeError(
            f"{source_name} must be provided as bytes"
        )

    if not raw:
        raise ValueError(
            f"{source_name} is empty"
        )

    try:
        return pd.read_csv(
            BytesIO(raw),
            sep=",",
            header=header,
            dtype=str,
            keep_default_na=False,
            encoding="utf-8-sig",
        )
    except Exception as exc:
        raise ValueError(
            f"Failed to parse {source_name}: {exc}"
        ) from exc


def find_named_smiles_column(
    dataframe: pd.DataFrame,
) -> object | None:
    """Find a column with a known SMILES column name."""
    matching_columns = [
        column
        for column in dataframe.columns
        if normalize_column_name(column)
        in {
            normalize_column_name(alias)
            for alias in SMILES_COLUMN_ALIASES
        }
    ]

    if len(matching_columns) > 1:
        raise ValueError(
            "Multiple SMILES columns were found: "
            f"{matching_columns}"
        )

    if matching_columns:
        return matching_columns[0]

    return None


def normalize_smiles_dataframe(
    dataframe: pd.DataFrame,
    smiles_column: object,
    source_name: str,
) -> pd.DataFrame:
    """Select, validate, and normalize a SMILES column."""
    normalized = dataframe[[smiles_column]].copy()

    normalized.columns = [STANDARD_SMILES_COLUMN]

    normalized[STANDARD_SMILES_COLUMN] = (
        normalized[STANDARD_SMILES_COLUMN]
        .astype(str)
        .str.strip()
    )

    if normalized.empty:
        raise ValueError(
            f"{source_name} contains no SMILES rows"
        )

    if normalized[STANDARD_SMILES_COLUMN].eq("").any():
        raise ValueError(
            f"{source_name} contains blank SMILES values"
        )

    for row_index, smiles in enumerate(
        normalized[STANDARD_SMILES_COLUMN],
        start=1,
    ):
        if not is_attachment_smiles(smiles):
            raise ValueError(
                f"{source_name} contains invalid attachment-point "
                f"SMILES at data row {row_index}: {smiles!r}"
            )

    return normalized.reset_index(drop=True)


def parse_smiles_csv(
    raw: bytes,
    source_name: str = "SMILES CSV",
) -> pd.DataFrame:
    """
    Parse CSV bytes and return a DataFrame with a standard smiles column.

    Supported inputs:
    - a column named smiles or a known alias;
    - one column with any other header;
    - a headerless one-column CSV.
    """
    dataframe = read_csv_bytes(
        raw=raw,
        source_name=source_name,
        header=0,
    )

    named_column = find_named_smiles_column(dataframe)

    if named_column is not None:
        return normalize_smiles_dataframe(
            dataframe=dataframe,
            smiles_column=named_column,
            source_name=source_name,
        )

    if len(dataframe.columns) != 1:
        raise ValueError(
            f"{source_name} must contain exactly one column "
            "when no SMILES column name is provided"
        )

    column = dataframe.columns[0]

    # If the column name itself is a valid SMILES, pandas probably
    # interpreted the first data row as a header.
    if is_attachment_smiles(column):
        dataframe = read_csv_bytes(
            raw=raw,
            source_name=source_name,
            header=None,
        )

        column = dataframe.columns[0]

    return normalize_smiles_dataframe(
        dataframe=dataframe,
        smiles_column=column,
        source_name=source_name,
    )


def dataframe_to_csv_bytes(
    dataframe: pd.DataFrame,
) -> bytes:
    """Serialize a DataFrame as UTF-8 encoded CSV bytes."""
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError(
            "Expected a pandas DataFrame"
        )

    return dataframe.to_csv(
        index=False,
    ).encode("utf-8")
