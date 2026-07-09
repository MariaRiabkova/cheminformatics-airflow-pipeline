from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator


DEFAULT_SMILES_COLUMN = "canonical_smiles"
DEFAULT_RADIUS = 2
DEFAULT_FINGERPRINT_SIZE = 2048
FINGERPRINT_TYPE = "ECFP4"


@dataclass(frozen=True)
class FingerprintResult:
    """Store fingerprint output and the numeric matrix used for clustering."""

    dataframe: pd.DataFrame
    matrix: np.ndarray


def _validate_smiles(smiles: object) -> str:
    """Validate and normalize one SMILES value."""
    if not isinstance(smiles, str):
        raise TypeError("SMILES must be a string")

    normalized_smiles = smiles.strip()

    if not normalized_smiles:
        raise ValueError("SMILES must not be blank")

    molecule = Chem.MolFromSmiles(normalized_smiles)

    if molecule is None:
        raise ValueError(f"Invalid SMILES: {normalized_smiles}")

    return Chem.MolToSmiles(
        molecule,
        canonical=True,
    )


def _fingerprint_to_array(
    fingerprint,
    fingerprint_size: int,
) -> np.ndarray:
    """Convert an RDKit bit vector to a NumPy array."""
    bit_string = fingerprint.ToBitString()

    array = np.fromiter(
        (int(bit) for bit in bit_string),
        dtype=np.uint8,
        count=fingerprint_size,
    )

    if array.shape != (fingerprint_size,):
        raise ValueError(
            "Unexpected fingerprint shape: "
            f"expected=({fingerprint_size},), "
            f"actual={array.shape}"
        )

    return array


def _fingerprint_on_bits(
    fingerprint_array: np.ndarray,
) -> str:
    """Serialize active fingerprint bit positions."""
    active_bits = np.flatnonzero(
        fingerprint_array
    )

    return " ".join(
        str(int(bit_index))
        for bit_index in active_bits
    )


def calculate_ecfp4_fingerprint(
    smiles: str,
    fingerprint_size: int = DEFAULT_FINGERPRINT_SIZE,
) -> tuple[str, str, np.ndarray]:
    """Calculate one ECFP4 fingerprint."""
    if fingerprint_size <= 0:
        raise ValueError(
            "Fingerprint size must be greater than zero"
        )

    canonical_smiles = _validate_smiles(
        smiles
    )

    molecule = Chem.MolFromSmiles(
        canonical_smiles
    )

    if molecule is None:
        raise ValueError(
            "Unexpected failure while parsing canonical SMILES: "
            f"{canonical_smiles}"
        )

    generator = (
        rdFingerprintGenerator.GetMorganGenerator(
            radius=DEFAULT_RADIUS,
            fpSize=fingerprint_size,
        )
    )

    fingerprint = generator.GetFingerprint(
        molecule
    )

    fingerprint_array = _fingerprint_to_array(
        fingerprint=fingerprint,
        fingerprint_size=fingerprint_size,
    )

    fingerprint_string = "".join(
        str(int(bit))
        for bit in fingerprint_array
    )

    on_bits = _fingerprint_on_bits(
        fingerprint_array
    )

    return (
        fingerprint_string,
        on_bits,
        fingerprint_array,
    )


def calculate_fingerprints_dataframe(
    molecules: pd.DataFrame,
    smiles_column: str = DEFAULT_SMILES_COLUMN,
    fingerprint_size: int = DEFAULT_FINGERPRINT_SIZE,
) -> FingerprintResult:
    """
    Calculate ECFP4 fingerprints for all molecules.

    Fingerprints are calculated once for every unique canonical SMILES
    and then mapped back to all original rows.
    """
    if not isinstance(molecules, pd.DataFrame):
        raise TypeError(
            "Molecules input must be a pandas DataFrame"
        )

    if smiles_column not in molecules.columns:
        raise ValueError(
            f"Molecules DataFrame must contain the "
            f"'{smiles_column}' column"
        )

    if molecules.empty:
        raise ValueError(
            "Molecules DataFrame must not be empty"
        )

    if fingerprint_size <= 0:
        raise ValueError(
            "Fingerprint size must be greater than zero"
        )

    result = molecules.copy()

    canonical_smiles_values: list[str] = []

    for row_position, smiles in enumerate(
        result[smiles_column].tolist()
    ):
        try:
            canonical_smiles = _validate_smiles(
                smiles
            )
        except Exception as exc:
            raise ValueError(
                "Failed to calculate fingerprint for "
                f"row_position={row_position}, "
                f"smiles={smiles!r}: {exc}"
            ) from exc

        canonical_smiles_values.append(
            canonical_smiles
        )

    unique_smiles = list(
        dict.fromkeys(
            canonical_smiles_values
        )
    )

    fingerprint_cache: dict[
        str,
        tuple[str, str, np.ndarray],
    ] = {}

    for canonical_smiles in unique_smiles:
        fingerprint_cache[canonical_smiles] = (
            calculate_ecfp4_fingerprint(
                smiles=canonical_smiles,
                fingerprint_size=fingerprint_size,
            )
        )

    fingerprint_strings: list[str] = []
    fingerprint_on_bits_values: list[str] = []
    fingerprint_arrays: list[np.ndarray] = []

    for canonical_smiles in canonical_smiles_values:
        (
            fingerprint_string,
            on_bits,
            fingerprint_array,
        ) = fingerprint_cache[canonical_smiles]

        fingerprint_strings.append(
            fingerprint_string
        )

        fingerprint_on_bits_values.append(
            on_bits
        )

        fingerprint_arrays.append(
            fingerprint_array
        )

    fingerprint_matrix = np.vstack(
        fingerprint_arrays
    ).astype(
        np.uint8,
        copy=False,
    )

    result[smiles_column] = (
        canonical_smiles_values
    )

    result["fingerprint_type"] = (
        FINGERPRINT_TYPE
    )

    result["fingerprint"] = (
        fingerprint_strings
    )

    result["fingerprint_on_bits"] = (
        fingerprint_on_bits_values
    )

    return FingerprintResult(
        dataframe=result,
        matrix=fingerprint_matrix,
    )
