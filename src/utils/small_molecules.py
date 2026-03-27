from __future__ import annotations
import re
from collections import Counter, defaultdict
import pandas as pd
from .dataset_ops import left, upd_n_mol

_FORM_RE = re.compile(r'([A-Z][a-z]?)(\d*)')

def n_atoms(formula: str) -> int:
    '''
    Возвращает общее число атомов в формуле.
    '''
    return sum(int(n) if n else 1 for _, n in _FORM_RE.findall(formula))

def collect_small_molecules(
    df: pd.DataFrame,
    *,
    max_atoms: int = 2,
) -> pd.DataFrame:
    '''
    Собирает сводку по формулам малых молекул.
    '''
    small_counter = Counter()

    for col in df.columns:
        if not col.startswith('formula_'):
            continue

        vals = df[col].dropna()
        for formula in vals:
            if isinstance(formula, str) and n_atoms(formula) <= max_atoms:
                small_counter[formula] += 1

    if not small_counter:
        return pd.DataFrame(columns=['formula', 'count'])

    return (
        pd.DataFrame(
            [{'formula': formula, 'count': count} for formula, count in small_counter.items()]
        )
        .sort_values('count', ascending=False)
        .reset_index(drop=True)
    )

def remove_selected_small(
    df: pd.DataFrame,
    *,
    small: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    '''
    Удаляет только выбранные малые молекулы из каждой строки.
    После удаления выполняет left() и upd_n_mol().
    '''
    out = df.copy()

    removed_small_counter = Counter()
    removed_small_files = defaultdict(set)

    def _rm_selected_small(row: pd.Series) -> pd.Series:
        source = row.get('source_file')

        for col in row.index:
            if not col.startswith('formula_'):
                continue

            formula = row[col]
            if not isinstance(formula, str):
                continue

            if formula in small:
                removed_small_counter[formula] += 1
                if isinstance(source, str):
                    removed_small_files[formula].add(source)

                row[col] = pd.NA
                row[col.replace('formula_', 'xyz_')] = pd.NA

        return row

    out = out.apply(_rm_selected_small, axis=1)
    out = left(out)
    out = upd_n_mol(out)

    if removed_small_counter:
        removed_summary = (
            pd.DataFrame(
                [
                    {
                        'formula': formula,
                        'count': removed_small_counter[formula],
                        'n_files': len(removed_small_files[formula]),
                        'files': sorted(removed_small_files[formula]),
                    }
                    for formula in removed_small_counter
                ]
            )
            .sort_values('count', ascending=False)
            .reset_index(drop=True)
        )
    else:
        removed_summary = pd.DataFrame(columns=['formula', 'count', 'n_files', 'files'])

    return out, removed_summary

def drop_rows_with_forbidden_small(
    df: pd.DataFrame,
    *,
    all_small: set[str],
    allowed_small: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    '''
    Удаляет строки, содержащие малые молекулы из all_small, которые не входят в allowed_small.
    '''
    removed_rows_files = defaultdict(set)

    def _drop_row_small(row: pd.Series) -> pd.Series | None:
        source = row.get('source_file')

        formulas = [
            row[c]
            for c in row.index
            if c.startswith('formula_') and isinstance(row[c], str)
        ]

        small_here = {formula for formula in formulas if formula in all_small}
        forbidden = small_here - allowed_small

        if forbidden:
            if isinstance(source, str):
                for formula in forbidden:
                    removed_rows_files[formula].add(source)
            return None

        return row

    out = (
        df.copy()
        .apply(_drop_row_small, axis=1)
        .dropna(how='all')
    )

    out = left(out)
    out = upd_n_mol(out)

    if removed_rows_files:
        removed_rows_df = (
            pd.DataFrame(
                [
                    {
                        'formula': formula,
                        'source_file': sorted(files),
                        'n_rows_removed': len(files),
                    }
                    for formula, files in removed_rows_files.items()
                ]
            )
            .sort_values('n_rows_removed', ascending=False)
            .reset_index(drop=True)
        )
    else:
        removed_rows_df = pd.DataFrame(columns=['formula', 'source_file', 'n_rows_removed'])

    return out, removed_rows_df