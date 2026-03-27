from rdkit import Chem
from openbabel import pybel

def xyz_to_canonical_smiles(xyz: str) -> str | None:
    '''
    Генерирует SMILES из координат xyz.
    '''
    try:
        ob_mol = pybel.readstring('xyz', xyz)
        mol_block = ob_mol.write('mol')

        rd_mol = Chem.MolFromMolBlock(mol_block, sanitize=True)
        if rd_mol is None:
            return None

        return Chem.MolToSmiles(
            rd_mol,
            canonical=True,
            isomericSmiles=False,
        )
    except Exception:
        return None