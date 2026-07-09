import pytest
from rdkit import Chem

from dags.lib.molecules.generation import (
    generate_molecule,
    validate_generated_molecule,
)


def test_generate_methyl_benzene():
    result = generate_molecule(
        scaffold_smiles="c1ccccc1[*:1]",
        r_group_smiles="[*:1]C",
    )

    assert result == "Cc1ccccc1"


def test_generate_chlorobenzene():
    result = generate_molecule(
        scaffold_smiles="c1ccccc1[*:1]",
        r_group_smiles="[*:1]Cl",
    )

    assert result == "Clc1ccccc1"


def test_generate_molecule_with_unnumbered_attachment_points():
    result = generate_molecule(
        scaffold_smiles="c1ccccc1*",
        r_group_smiles="*C",
    )

    assert result == "Cc1ccccc1"


def test_attachment_point_mismatch():
    with pytest.raises(
        ValueError,
        match="Attachment point mismatch",
    ):
        generate_molecule(
            scaffold_smiles="c1ccccc1[*:1]",
            r_group_smiles="[*:2]C",
        )


def test_numbered_and_unnumbered_attachment_points_do_not_match():
    with pytest.raises(
        ValueError,
        match="Attachment point mismatch",
    ):
        generate_molecule(
            scaffold_smiles="c1ccccc1[*:1]",
            r_group_smiles="*C",
        )


def test_invalid_scaffold():
    with pytest.raises(
        ValueError,
        match="Invalid scaffold SMILES",
    ):
        generate_molecule(
            scaffold_smiles="not_a_smiles",
            r_group_smiles="[*:1]C",
        )


def test_invalid_r_group():
    with pytest.raises(
        ValueError,
        match="Invalid R-group SMILES",
    ):
        generate_molecule(
            scaffold_smiles="c1ccccc1[*:1]",
            r_group_smiles="not_a_smiles",
        )


def test_scaffold_without_attachment_point():
    with pytest.raises(
        ValueError,
        match="exactly one attachment point",
    ):
        generate_molecule(
            scaffold_smiles="c1ccccc1",
            r_group_smiles="[*:1]C",
        )


def test_r_group_without_attachment_point():
    with pytest.raises(
        ValueError,
        match="exactly one attachment point",
    ):
        generate_molecule(
            scaffold_smiles="c1ccccc1[*:1]",
            r_group_smiles="C",
        )


def test_multiple_attachment_points_are_not_supported():
    with pytest.raises(
        ValueError,
        match="exactly one attachment point",
    ):
        generate_molecule(
            scaffold_smiles="C([*:1])([*:2])",
            r_group_smiles="[*:1]C",
        )


def test_generated_molecule_contains_no_dummy_atoms():
    result = generate_molecule(
        scaffold_smiles="c1ccccc1[*:1]",
        r_group_smiles="[*:1]C",
    )

    molecule = Chem.MolFromSmiles(result)

    assert molecule is not None
    assert all(
        atom.GetAtomicNum() != 0
        for atom in molecule.GetAtoms()
    )


def test_generated_molecule_contains_no_atom_map_numbers():
    result = generate_molecule(
        scaffold_smiles="c1ccccc1[*:1]",
        r_group_smiles="[*:1]C",
    )

    molecule = Chem.MolFromSmiles(result)

    assert molecule is not None
    assert all(
        atom.GetAtomMapNum() == 0
        for atom in molecule.GetAtoms()
    )


def test_generated_molecule_is_a_single_fragment():
    result = generate_molecule(
        scaffold_smiles="c1ccccc1[*:1]",
        r_group_smiles="[*:1]C",
    )

    molecule = Chem.MolFromSmiles(result)

    assert molecule is not None
    assert len(Chem.GetMolFrags(molecule)) == 1


def test_generated_smiles_can_be_parsed_again():
    result = generate_molecule(
        scaffold_smiles="c1ccccc1[*:1]",
        r_group_smiles="[*:1]O",
    )

    reparsed_molecule = Chem.MolFromSmiles(result)

    assert reparsed_molecule is not None


def test_rejects_dummy_atom_at_end_of_molecule():
    molecule = Chem.MolFromSmiles("CC*")

    with pytest.raises(
        ValueError,
        match="unresolved dummy atoms",
    ):
        validate_generated_molecule(molecule)


def test_rejects_dummy_atom_inside_molecule():
    molecule = Chem.MolFromSmiles("C*C")

    with pytest.raises(
        ValueError,
        match="unresolved dummy atoms",
    ):
        validate_generated_molecule(molecule)


def test_rejects_disconnected_fragments():
    molecule = Chem.MolFromSmiles("CC.CC")

    with pytest.raises(
        ValueError,
        match="disconnected fragments",
    ):
        validate_generated_molecule(molecule)


def test_rejects_unresolved_atom_map_numbers():
    molecule = Chem.MolFromSmiles("[CH3:1]C")

    with pytest.raises(
        ValueError,
        match="unresolved atom-map numbers",
    ):
        validate_generated_molecule(molecule)


def test_rejects_empty_generated_molecule():
    with pytest.raises(
        ValueError,
        match="Generated molecule is empty",
    ):
        validate_generated_molecule(None)
