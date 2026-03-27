import re
from pathlib import Path
import pandas as pd

path = Path('CCDC_pymatgen_parquets')

METALS = {
    'Li', 'Na', 'K', 'Rb', 'Cs', 'Fr',
    'Be', 'Mg', 'Ca', 'Sr', 'Ba', 'Ra',
    'Al', 'Ga', 'In', 'Tl',
    'Sn', 'Pb', 'Bi',
    'Ge', 'As', 'Sb', 'Te',
    'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
    'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
    'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
    'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn',
    'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu',
    'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr'
}

_EL_TOKEN = re.compile(r'([A-Z][a-z]?)')
_FORM_RE = re.compile(r'([A-Z][a-z]?)(\d*)')

def explode_to_wide(df, col_prefix):
    max_len = df.apply(len).max()
    return pd.DataFrame(
        df.tolist(),
        index=df.index,
        columns=[f'{col_prefix}_{i}' for i in range(max_len)]
    )

def parse_formula(formula: str) -> dict:
    return {
        el: int(cnt) if cnt else 1
        for el, cnt in _FORM_RE.findall(formula)
    }

def non_h_size(formula: str) -> int:
    comp = parse_formula(formula)
    return sum(v for k, v in comp.items() if k != 'H')

def _idxs(df: pd.DataFrame):
    return sorted(
        int(c.split('_')[1])
        for c in df.columns
        if c.startswith('formula_')
    )

def contains_metal(formula):
    if formula is None or pd.isna(formula):
        return False
    s = re.sub(r'[\[\]\+\-\(\)]', '', str(formula))
    elements = {el for el, _ in _FORM_RE.findall(s)}
    return bool(elements & METALS)


def row_has_metal(row):
    for col in row.index:
        if col.startswith('formula_') and contains_metal(row[col]):
            return True
    return False


def sort_molecules_inside_row(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    idxs = _idxs(out)
    if not idxs:
        return out

    for r in out.index:
        mols = []

        for i in idxs:
            fcol, xcol = f'formula_{i}', f'xyz_{i}'
            fval, xval = out.at[r, fcol], out.at[r, xcol]

            if not isinstance(fval, str):
                continue

            f = fval.strip()
            if not f:
                continue

            mols.append({
                'formula': fval,
                'xyz': xval,
                'has_metal': contains_metal(f),
                'size': non_h_size(f),
            })

        mols.sort(
            key=lambda m: (
                not m['has_metal'],
                -m['size']
            )
        )

        for i in idxs:
            out.at[r, f'formula_{i}'] = pd.NA
            out.at[r, f'xyz_{i}'] = pd.NA

        for i, m in zip(idxs, mols):
            out.at[r, f'formula_{i}'] = m['formula']
            out.at[r, f'xyz_{i}'] = m['xyz']

    return out


dfs = []
for file in path.glob('molecules_*.parquet'):
    df = pd.read_parquet(file)
    df['source_file'] = file.name
    dfs.append(df)

df_all = pd.concat(dfs, ignore_index=True)

df_all = df_all.sort_values(
    ['source_file', 'mol_index']
)

grouped = (
    df_all
    .groupby('source_file')
    .agg({
        'formula': list,
        'xyz': list,
        'mol_index': list
    })
)

formula_wide = explode_to_wide(grouped['formula'], 'formula')
xyz_wide = explode_to_wide(grouped['xyz'], 'xyz')
molid_wide = explode_to_wide(grouped['mol_index'], 'mol_index')

df = pd.concat(
    [formula_wide, molid_wide, xyz_wide],
    axis=1
).reset_index()

formula_cols = [c for c in df.columns if c.startswith('formula_')]

df['n_molecules'] = df[formula_cols].notna().sum(axis=1)

df['source_file'] = df['source_file'].str.extract(
    r'molecules_CSD_CIF_(.+?)\.cif\.parquet',
    expand=False
)

rename_map = {}

for c in df.columns:
    if c.startswith('formula_'):
        i = int(c.split('_')[1])
        rename_map[c] = f'formula_{i+1}'
    elif c.startswith('xyz_'):
        i = int(c.split('_')[1])
        rename_map[c] = f'xyz_{i+1}'

df = df.rename(columns=rename_map)

df['has_metal'] = df.apply(row_has_metal, axis=1)

df = df.drop(
    columns=[c for c in df.columns if c.startswith('mol_index_')]
)

df = sort_molecules_inside_row(df)

df.to_pickle('csd_raw.pkl')
