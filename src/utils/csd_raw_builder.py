from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
from .composition import annotate_metal_metadata


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
    'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr',
}

_FORM_RE = re.compile(r'([A-Z][a-z]?)(\d*)')


def explode_to_wide(series: pd.Series, col_prefix: str) -> pd.DataFrame:
    if series.empty:
        return pd.DataFrame(index=series.index)

    max_len = int(series.apply(len).max())
    if max_len == 0:
        return pd.DataFrame(index=series.index)

    return pd.DataFrame(
        series.tolist(),
        index=series.index,
        columns=[f'{col_prefix}_{i}' for i in range(max_len)],
    )


def parse_formula(formula: str) -> dict[str, int]:
    return {
        element: int(count) if count else 1
        for element, count in _FORM_RE.findall(formula)
    }


def non_h_size(formula: str) -> int:
    composition = parse_formula(formula)
    return sum(value for key, value in composition.items() if key != 'H')


def contains_metal(formula: str | None) -> bool:
    if formula is None or pd.isna(formula):
        return False

    clean_formula = re.sub(r'[\[\]\+\-\(\)]', '', str(formula))
    elements = {element for element, _ in _FORM_RE.findall(clean_formula)}
    return bool(elements & METALS)


def row_has_metal(row: pd.Series) -> bool:
    for col in row.index:
        if col.startswith('formula_') and contains_metal(row[col]):
            return True
    return False


def _component_indices(df: pd.DataFrame) -> list[int]:
    return sorted(
        int(col.split('_')[1])
        for col in df.columns
        if col.startswith('formula_')
    )


def sort_molecules_inside_row(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    idxs = _component_indices(out)

    if not idxs:
        return out

    for row_idx in out.index:
        molecules = []

        for i in idxs:
            formula_col = f'formula_{i}'
            xyz_col = f'xyz_{i}'

            formula_value = out.at[row_idx, formula_col]
            xyz_value = out.at[row_idx, xyz_col]

            if not isinstance(formula_value, str):
                continue

            formula = formula_value.strip()
            if not formula:
                continue

            molecules.append(
                {
                    'formula': formula_value,
                    'xyz': xyz_value,
                    'has_metal': contains_metal(formula),
                    'size': non_h_size(formula),
                }
            )

        molecules.sort(
            key=lambda molecule: (
                not molecule['has_metal'],
                -molecule['size'],
            )
        )

        for i in idxs:
            out.at[row_idx, f'formula_{i}'] = pd.NA
            out.at[row_idx, f'xyz_{i}'] = pd.NA

        for i, molecule in zip(idxs, molecules):
            out.at[row_idx, f'formula_{i}'] = molecule['formula']
            out.at[row_idx, f'xyz_{i}'] = molecule['xyz']

    return out


def load_parsed_cif_rows(
    input_dir: str | Path,
    pattern: str = '*.parquet',
) -> pd.DataFrame:
    input_dir = Path(input_dir)
    parquet_files = sorted(input_dir.glob(pattern))

    if not parquet_files:
        raise FileNotFoundError(
            f'No parquet files matching "{pattern}" found in {input_dir}'
        )

    frames = []
    for file in parquet_files:
        df = pd.read_parquet(file)
        df['source_file'] = file.stem
        frames.append(df)

    return pd.concat(frames, ignore_index=True)


def build_csd_raw_dataframe(
    input_dir: str | Path,
    pattern: str = '*.parquet',
) -> pd.DataFrame:
    df_all = load_parsed_cif_rows(input_dir=input_dir, pattern=pattern)
    df_all = df_all.sort_values(['source_file', 'mol_index'])

    grouped = (
        df_all
        .groupby('source_file')
        .agg(
            {
                'formula': list,
                'xyz': list,
                'mol_index': list,
            }
        )
    )

    formula_wide = explode_to_wide(grouped['formula'], 'formula')
    xyz_wide = explode_to_wide(grouped['xyz'], 'xyz')
    molid_wide = explode_to_wide(grouped['mol_index'], 'mol_index')

    df = pd.concat([formula_wide, molid_wide, xyz_wide], axis=1).reset_index()

    formula_cols = [col for col in df.columns if col.startswith('formula_')]
    df['n_molecules'] = df[formula_cols].notna().sum(axis=1)
    rename_map = {}
    for col in df.columns:
        if col.startswith('formula_'):
            idx = int(col.split('_')[1])
            rename_map[col] = f'formula_{idx + 1}'
        elif col.startswith('xyz_'):
            idx = int(col.split('_')[1])
            rename_map[col] = f'xyz_{idx + 1}'

    df = df.rename(columns=rename_map)

    mol_index_cols = [col for col in df.columns if col.startswith('mol_index_')]
    df = df.drop(columns=mol_index_cols)
    df = sort_molecules_inside_row(df)
    df = annotate_metal_metadata(df)

    return df
