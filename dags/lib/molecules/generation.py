from __future__ import annotations

from rdkit import Chem


def find_single_attachment_atom(
    molecule: Chem.Mol,
    source_name: str,
) -> Chem.Atom:
    """
    Find exactly one dummy attachment atom, such as [*:1].

    A dummy atom has atomic number 0.
    """
    attachment_atoms = [
        atom
        for atom in molecule.GetAtoms()
        if atom.GetAtomicNum() == 0
    ]

    if len(attachment_atoms) != 1:
        raise ValueError(
            f"{source_name} must contain exactly one attachment point; "
            f"found {len(attachment_atoms)}"
        )

    attachment_atom = attachment_atoms[0]

    if len(attachment_atom.GetNeighbors()) != 1:
        raise ValueError(
            f"{source_name} attachment point must have exactly one neighbour"
        )

    return attachment_atom


def generate_molecule(
    scaffold_smiles: str,
    r_group_smiles: str,
) -> str:
    """
    Join one scaffold and one R-group through their mapped attachment points.

    Example:
        c1ccccc1[*:1] + [*:1]C -> Cc1ccccc1
    """
    scaffold = Chem.MolFromSmiles(scaffold_smiles)
    r_group = Chem.MolFromSmiles(r_group_smiles)

    if scaffold is None:
        raise ValueError(
            f"Invalid scaffold SMILES: {scaffold_smiles}"
        )

    if r_group is None:
        raise ValueError(
            f"Invalid R-group SMILES: {r_group_smiles}"
        )

    scaffold_attachment = find_single_attachment_atom(
        scaffold,
        "scaffold",
    )
    r_group_attachment = find_single_attachment_atom(
        r_group,
        "R-group",
    )

    scaffold_map_number = scaffold_attachment.GetAtomMapNum()
    r_group_map_number = r_group_attachment.GetAtomMapNum()

    if scaffold_map_number != r_group_map_number:
        raise ValueError(
            "Attachment point mismatch: "
            f"scaffold has [*:{scaffold_map_number}], "
            f"R-group has [*:{r_group_map_number}]"
        )

    scaffold_attachment_index = scaffold_attachment.GetIdx()
    r_group_attachment_index = r_group_attachment.GetIdx()

    scaffold_neighbor_index = (
        scaffold_attachment.GetNeighbors()[0].GetIdx()
    )
    r_group_neighbor_index = (
        r_group_attachment.GetNeighbors()[0].GetIdx()
    )

    combined = Chem.CombineMols(scaffold, r_group)
    editable = Chem.RWMol(combined)

    r_group_offset = scaffold.GetNumAtoms()

    editable.AddBond(
        scaffold_neighbor_index,
        r_group_offset + r_group_neighbor_index,
        Chem.BondType.SINGLE,
    )

    dummy_atom_indexes = [
        scaffold_attachment_index,
        r_group_offset + r_group_attachment_index,
    ]

    # Remove atoms from the highest index to prevent index shifting.
    for atom_index in sorted(
        dummy_atom_indexes,
        reverse=True,
    ):
        editable.RemoveAtom(atom_index)

    result = editable.GetMol()
    Chem.SanitizeMol(result)

    return Chem.MolToSmiles(
        result,
        canonical=True,
    )
