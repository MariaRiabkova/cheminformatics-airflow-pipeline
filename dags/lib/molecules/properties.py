from __future__ import annotations

import pandas as pd
from rdkit import Chem
from rdkit.Chem import (
    Crippen,
    Descriptors,
    Lipinski,
    QED,
    rdMolDescriptors,
)


DEFAULT_SMILES_COLUMN = "generated_smiles"

PROPERTY_COLUMNS = [
    "canonical_smiles",
    "mol_formula",
    "mol_weight",
    "log_p",
    "tpsa",
    "hba",
    "hbd",
    "rotatable_bonds",
    "aromatic_rings",
    "heavy_atom_count",
    "hetero_atom_count",
    "ring_count",
    "fraction_csp3",
    "formal_charge",
    "qed",
    "lipinski_pass",
]


def check_lipinski_rule(
    mol_weight: float,
    log_p: float,
    hba: int,
    hbd: int,
) -> bool:
    """Check whether a molecule satisfies Lipinski's rule of five."""
    return (
        mol_weight <= 500
        and log_p <= 5
        and hba <= 10
        and hbd <= 5
    )


def calculate_molecular_properties(
    smiles: str,
) -> dict[str, str | float | int | bool]:
    """Calculate molecular properties for one SMILES value."""
    if not isinstance(smiles, str):
        raise TypeError(
            "SMILES must be a string"
        )

    normalized_smiles = smiles.strip()

    if not normalized_smiles:
        raise ValueError(
            "SMILES must not be blank"
        )

    molecule = Chem.MolFromSmiles(
        normalized_smiles
    )

    if molecule is None:
        raise ValueError(
            f"Invalid SMILES: {normalized_smiles}"
        )

    canonical_smiles = Chem.MolToSmiles(
        molecule,
        canonical=True,
    )

    mol_weight = round(
        Descriptors.MolWt(molecule),
        4,
    )

    log_p = round(
        Crippen.MolLogP(molecule),
        4,
    )

    tpsa = round(
        rdMolDescriptors.CalcTPSA(molecule),
        4,
    )

    hba = int(
        Lipinski.NumHAcceptors(molecule)
    )

    hbd = int(
        Lipinski.NumHDonors(molecule)
    )

    return {
        "canonical_smiles": canonical_smiles,
        "mol_formula": (
            rdMolDescriptors.CalcMolFormula(
                molecule
            )
        ),
        "mol_weight": mol_weight,
        "log_p": log_p,
        "tpsa": tpsa,
        "hba": hba,
        "hbd": hbd,
        "rotatable_bonds": int(
            Lipinski.NumRotatableBonds(
                molecule
            )
        ),
        "aromatic_rings": int(
            rdMolDescriptors.CalcNumAromaticRings(
                molecule
            )
        ),
        "heavy_atom_count": int(
            molecule.GetNumHeavyAtoms()
        ),
        "hetero_atom_count": int(
            rdMolDescriptors.CalcNumHeteroatoms(
                molecule
            )
        ),
        "ring_count": int(
            rdMolDescriptors.CalcNumRings(
                molecule
            )
        ),
        "fraction_csp3": round(
            rdMolDescriptors.CalcFractionCSP3(
                molecule
            ),
            4,
        ),
        "formal_charge": int(
            Chem.GetFormalCharge(
                molecule
            )
        ),
        "qed": round(
            QED.qed(molecule),
            4,
        ),
        "lipinski_pass": check_lipinski_rule(
            mol_weight=mol_weight,
            log_p=log_p,
            hba=hba,
            hbd=hbd,
        ),
    }


def calculate_properties_dataframe(
    molecules: pd.DataFrame,
    smiles_column: str = DEFAULT_SMILES_COLUMN,
) -> pd.DataFrame:
    """Add molecular properties to a molecule DataFrame."""
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

    calculated_rows = []

    for row_index, smiles in enumerate(
        molecules[smiles_column],
    ):
        try:
            properties = (
                calculate_molecular_properties(
                    smiles
                )
            )
        except Exception as exc:
            raise ValueError(
                "Failed to calculate molecular properties for "
                f"row_index={row_index}, "
                f"smiles={smiles!r}: {exc}"
            ) from exc

        calculated_rows.append(
            properties
        )

    properties_dataframe = pd.DataFrame(
        calculated_rows,
        index=molecules.index,
        columns=PROPERTY_COLUMNS,
    )

    return pd.concat(
        [
            molecules.copy(),
            properties_dataframe,
        ],
        axis=1,
    )
