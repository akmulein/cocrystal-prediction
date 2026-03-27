from pathlib import Path
import numpy as np
import pandas as pd
import subprocess
import tempfile


def xyz_to_smiles_obabel(xyz_block: str) -> str | None:
    if xyz_block is None or (isinstance(xyz_block, float) and np.isnan(xyz_block)):
        return None
    if not isinstance(xyz_block, str):
        return None

    xyz_block = xyz_block.strip()
    if not xyz_block:
        return None

    try:
        with tempfile.TemporaryDirectory() as td:
            xyz_path = Path(td) / 'mol.xyz'
            xyz_path.write_text(xyz_block, encoding='utf-8')

            cmd = ['obabel', '-ixyz', str(xyz_path), '-osmi', '-c', '--errorlevel', '1']
            res = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )

            if res.returncode != 0:
                return None

            out = (res.stdout or '').strip()
            if not out:
                return None

            smiles = out.split()[0].strip()
            if smiles in ('', 'nan', 'NaN'):
                return None

            return smiles

    except FileNotFoundError:
        raise RuntimeError('Команда "obabel" не найдена в PATH.')
    except Exception:
        return None


def add_smiles_from_xyz(
    df: pd.DataFrame,
    xyz_cols: list[str],
    out_cols: list[str],
) -> pd.DataFrame:
    out = df.copy()

    for xyz_col, out_col in zip(xyz_cols, out_cols):
        out[out_col] = out[xyz_col].apply(xyz_to_smiles_obabel)

    return out