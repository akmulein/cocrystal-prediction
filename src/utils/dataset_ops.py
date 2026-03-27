import pandas as pd
from .composition import _row_flags, _row_metal_atom_counts

def _idxs(df: pd.DataFrame) -> list[int]:
    '''
    извлекает номера компонентов из колонок вида formula_i
    '''
    return sorted(
        int(c.split('_')[1])
        for c in df.columns
        if c.startswith('formula_')
    )

def upd_n_mol(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Обновляет счетчик молекул в строке.
    '''
    out = df.copy()
    formula_cols = [c for c in out.columns if c.startswith('formula_')]
    out['n_molecules'] = out[formula_cols].notna().sum(axis=1)
    return out

def left(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Сдвигает молекулы после очистки влево.
    '''
    out = df.copy()
    idxs = _idxs(out)
    if not idxs:
        return out

    for r in out.index:
        kept = []

        for i in idxs:
            f = out.at[r, f'formula_{i}']
            x = out.at[r, f'xyz_{i}']
            if isinstance(f, str):
                kept.append((f, x))

        for i in idxs:
            out.at[r, f'formula_{i}'] = pd.NA
            out.at[r, f'xyz_{i}'] = pd.NA

        for i, (f, x) in zip(idxs, kept):
            out.at[r, f'formula_{i}'] = f
            out.at[r, f'xyz_{i}'] = x

    return out

def summary(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Саммари по датасету.
    '''
    tmp = df.copy()

    counts = tmp.apply(_row_metal_atom_counts, axis=1)
    tmp['metal_atoms_all'] = counts.map(lambda x: x[0])
    tmp['df_atoms'] = counts.map(lambda x: x[1])
    tmp['d_atoms'] = counts.map(lambda x: x[2])
    tmp['f_atoms'] = counts.map(lambda x: x[3])

    tmp['has_metal'] = tmp['metal_atoms_all'] > 0
    tmp['has_df'] = tmp['df_atoms'] > 0

    tmp['single_metal'] = tmp['metal_atoms_all'] == 1
    tmp['single_df'] = (tmp['df_atoms'] == 1) & (tmp['metal_atoms_all'] == 1)

    flags = tmp.apply(_row_flags, axis=1, metal_col='metal_atoms_all', df_col='df_atoms')
    tmp = pd.concat([tmp, flags], axis=1)

    out = (
        tmp.groupby('n_molecules')
        .agg(
            total=('n_molecules', 'size'),
            with_metal=('has_metal', 'sum'),
            with_df=('has_df', 'sum'),
            single_metal=('single_metal', 'sum'),
            single_df=('single_df', 'sum'),
            single_metal_sulfonic_amine=('single_metal_sulfonic_amine', 'sum'),
            single_df_sulfonic_amine=('single_df_sulfonic_amine', 'sum'),
            amine_sulfonic_any=('amine_sulfonic_any', 'sum'),
        )
        .reset_index()
        .rename(columns={'n_molecules': 'n_components'})
        .sort_values('n_components')
        .reset_index(drop=True)
    )

    out = out[out['n_components'] > 0]
    out['without_metal'] = out['total'] - out['with_metal']

    return out[
        [
            'n_components',
            'total',
            'without_metal',
            'with_metal',
            'with_df',
            'single_metal',
            'single_df',
            'single_metal_sulfonic_amine',
            'single_df_sulfonic_amine',
            'amine_sulfonic_any'
        ]
    ]

def system_selection(df: pd.DataFrame) -> pd.DataFrame:
    tmp = df.copy()

    counts = tmp.apply(_row_metal_atom_counts, axis=1)
    tmp['_metal_atoms_all'] = counts.map(lambda x: x[0])
    tmp['_df_atoms'] = counts.map(lambda x: x[1])

    tmp['_single_metal'] = tmp['_metal_atoms_all'] == 1
    tmp['_single_df'] = (tmp['_df_atoms'] == 1) & (tmp['_metal_atoms_all'] == 1)

    flags = tmp.apply(_row_flags, axis=1, metal_col='_metal_atoms_all', df_col='_df_atoms')
    flags = flags.rename(columns=lambda c: f'_{c}')
    tmp = pd.concat([tmp, flags], axis=1)

    return tmp

def select_systems(df: pd.DataFrame, **conditions) -> pd.DataFrame:
    '''
    Формирует датасет из систем с заданными параметрами.
    '''
    tmp = system_selection(df)

    for key, val in conditions.items():
        if key == 'n_components':
            tmp = tmp[tmp['n_molecules'] == val]
        else:
            tmp = tmp[tmp[f'_{key}'] == val]

    return tmp[df.columns]