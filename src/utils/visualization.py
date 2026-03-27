import pandas as pd
import py3Dmol
from openbabel import pybel


def visual(df: pd.DataFrame, file_name: str) -> None:
    '''
    Визуализирует молекулы со-кристалла из xyz координат.
    '''
    row = df[df['source_file'] == file_name].iloc[0]

    for c in df.columns:
        if c.startswith('xyz_') and isinstance(row[c], str):
            ob_mol = pybel.readstring('xyz', row[c])
            mol_block = ob_mol.write('mol')

            view = py3Dmol.view(width=400, height=350)
            view.addModel(mol_block, 'mol')
            view.setStyle({'stick': {}, 'sphere': {'scale': 0.25}})
            view.zoomTo()
            view.show()