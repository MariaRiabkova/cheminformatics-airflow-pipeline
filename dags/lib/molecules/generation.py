from __future__ import annotations

from rdkit import Chem


def find_single_attachment_atom(
    molecule: Chem.Mol,
    source_name: str,
) -> Chem.Atom:
    """
    Find exactly one dummy attachment atom in a molecule.

    A dummy atom has atomic number 0 and is represented in SMILES
    as either "*" or a mapped attachment point such as "[*:1]".
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


def validate_generated_molecule(
    molecule: Chem.Mol,
) -> str:
    """
    Validate a generated molecule before downstream processing.

    RDKit parseability alone is not sufficient validation because RDKit
    can accept dummy atoms and calculate descriptors for structures that
    are invalid for this pipeline.

    A valid generated molecule must:
    - pass RDKit sanitization;
    - contain no dummy atoms;
    - contain no unresolved atom-map numbers;
    - contain exactly one connected fragment;
    - produce a non-empty canonical SMILES;
    - be parseable again from the canonical SMILES.

    Returns:
        Canonical SMILES for the validated molecule.
    """
    if molecule is None:
        raise ValueError("Generated molecule is empty")

    try:
        Chem.SanitizeMol(molecule)
    except Exception as exc:
        raise ValueError(
            f"Generated molecule failed RDKit sanitization: {exc}"
        ) from exc

    dummy_atoms = [
        atom
        for atom in molecule.GetAtoms()
        if atom.GetAtomicNum() == 0
    ]

    if dummy_atoms:
        raise ValueError(
            "Generated molecule contains unresolved dummy atoms"
        )

    mapped_atoms = [
        atom
        for atom in molecule.GetAtoms()
        if atom.GetAtomMapNum() != 0
    ]

    if mapped_atoms:
        raise ValueError(
            "Generated molecule contains unresolved atom-map numbers"
        )

    fragments = Chem.GetMolFrags(
        molecule,
        asMols=False,
        sanitizeFrags=False,
    )

    if len(fragments) != 1:
        raise ValueError(
            f"Generated molecule contains {len(fragments)} "
            "disconnected fragments"
        )

    canonical_smiles = Chem.MolToSmiles(
        molecule,
        canonical=True,
    )

    if not canonical_smiles:
        raise ValueError(
            "Generated molecule has an empty canonical SMILES"
        )

    if "*" in canonical_smiles:
        raise ValueError(
            "Generated SMILES contains unresolved attachment points"
        )

    reparsed_molecule = Chem.MolFromSmiles(canonical_smiles)

    if reparsed_molecule is None:
        raise ValueError(
            "Generated canonical SMILES cannot be parsed by RDKit"
        )

    reparsed_fragments = Chem.GetMolFrags(
        reparsed_molecule,
        asMols=False,
        sanitizeFrags=False,
    )

    if len(reparsed_fragments) != 1:
        raise ValueError(
            "Generated canonical SMILES contains disconnected fragments"
        )

    return canonical_smiles


def generate_molecule(
    scaffold_smiles: str,
    r_group_smiles: str,
) -> str:
    """
    Join one scaffold and one R-group through a single attachment point.

    Supported input formats:
        CC* + *C
        CC[*:1] + [*:1]C

    Both inputs must contain exactly one dummy atom. Attachment points
    must either be unnumbered in both inputs or have the same atom-map
    number.

    Multiple attachment points are not supported in this pipeline
    iteration.

    Returns:
        Canonical SMILES for the validated generated molecule.
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
            f"scaffold has map number {scaffold_map_number}, "
            f"R-group has map number {r_group_map_number}"
        )

    scaffold_attachment_index = scaffold_attachment.GetIdx()
    r_group_attachment_index = r_group_attachment.GetIdx()

    scaffold_neighbor_index = (
        scaffold_attachment.GetNeighbors()[0].GetIdx()
    )
    r_group_neighbor_index = (
        r_group_attachment.GetNeighbors()[0].GetIdx()
    )

    combined = Chem.CombineMols(
        scaffold,
        r_group,
    )
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

    # Remove atoms from the highest index to avoid index shifting.
    for atom_index in sorted(
        dummy_atom_indexes,
        reverse=True,
    ):
        editable.RemoveAtom(atom_index)

    generated_molecule = editable.GetMol()

    return validate_generated_molecule(
        generated_molecule
    )
