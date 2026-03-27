import re
from pathlib import Path
import pandas as pd
from .cif_parser import xyz_to_canonical_smiles
from .dataset_ops import _idxs, left, upd_n_mol

SIMPLE_ATOM = re.compile(r'\[([A-Z][a-z]?)\]')

def load_solvent_set(path: str | Path) -> set[str]:
    '''
    Загружает множество SMILES растворителей из текстового файла.
    '''
    path = Path(path)

    with path.open('r', encoding='utf-8', errors='ignore') as f:
        return {
            line.strip()
            for line in f
            if line.strip() and not line.lstrip().startswith('#')
        }

def normalize_simple_smiles(smi: str | None) -> str | None:
    '''
    Убирает [] из SMILES.
    '''
    if smi is None:
        return None

    if any(x in smi for x in ['+', '-', '.', '@']):
        return smi

    return SIMPLE_ATOM.sub(r'\1', smi)

def remove_solvents(
    df: pd.DataFrame,
    solvent_set: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    '''
    Удаляет растворители из solvent_set.
    '''
    out = df.copy()
    idxs = _idxs(out)

    if not idxs or not solvent_set:
        empty_summary = pd.DataFrame(columns=['formula', 'count'])
        return out, empty_summary

    solvent_norm = {
        s.strip()
        for s in solvent_set
        if isinstance(s, str) and s.strip()
    }

    removed_formulas: list[str] = []

    for r in out.index:
        for i in idxs:
            fcol = f'formula_{i}'
            xcol = f'xyz_{i}'

            fval = out.at[r, fcol]
            xval = out.at[r, xcol]

            if not isinstance(xval, str) or not xval.strip():
                continue

            smi_raw = xyz_to_canonical_smiles(xval)
            smi = normalize_simple_smiles(smi_raw)

            if smi and smi in solvent_norm:
                removed_formulas.append(str(fval))
                out.at[r, fcol] = pd.NA
                out.at[r, xcol] = pd.NA

    out = left(out)
    out = upd_n_mol(out)

    if removed_formulas:
        removed_summary = (
            pd.Series(removed_formulas)
            .value_counts()
            .rename_axis('formula')
            .reset_index(name='count')
        )
    else:
        removed_summary = pd.DataFrame(columns=['formula', 'count'])

    return out, removed_summary