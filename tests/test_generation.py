import pytest

from dags.lib.molecules.generation import generate_molecule


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


def test_attachment_point_mismatch():
    with pytest.raises(ValueError, match="Attachment point mismatch"):
        generate_molecule(
            scaffold_smiles="c1ccccc1[*:1]",
            r_group_smiles="[*:2]C",
        )


def test_invalid_scaffold():
    with pytest.raises(ValueError, match="Invalid scaffold SMILES"):
        generate_molecule(
            scaffold_smiles="not_a_smiles",
            r_group_smiles="[*:1]C",
        )


def test_scaffold_without_attachment_point():
    with pytest.raises(ValueError, match="exactly one attachment point"):
        generate_molecule(
            scaffold_smiles="c1ccccc1",
            r_group_smiles="[*:1]C",
        )
