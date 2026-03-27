import hashlib
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from .xtb_features import build_row_vectors_from_blocks


def _empty_vec(n: int = 135) -> list[float]:
    return [float('nan')] * n

def smiles_key_raw(smiles: str) -> str:
    return hashlib.sha1(str(smiles).encode('utf-8')).hexdigest()[:16]


def load_xtb_blocks_indexed(xtb_dir: str | Path) -> dict[tuple[str, int], object]:
    '''
    Читает файлы вида {source_file}__{i}.pkl из xtb_dir
    '''
    xtb_dir = Path(xtb_dir)
    blocks = {}

    for fp in xtb_dir.glob('*.pkl'):
        stem = fp.stem
        if '__' not in stem:
            continue

        sf, idx_str = stem.rsplit('__', 1)
        if not idx_str.isdigit():
            continue

        mol_idx = int(idx_str)

        with fp.open('rb') as f:
            payload = pickle.load(f)

        source_file = payload.get('source_file', sf)
        mol_idx_payload = payload.get('mol_idx', mol_idx)
        if mol_idx_payload != mol_idx:
            mol_idx_payload = mol_idx

        blocks[(source_file, mol_idx_payload)] = payload.get('xtb')

    return blocks


def load_xtb_rows_from_dir(
    xtb_dir: str | Path,
    sort_numeric_stems: bool = False,
) -> pd.DataFrame:
    xtb_dir = Path(xtb_dir)
    files = list(xtb_dir.glob('*.pkl'))

    if sort_numeric_stems:
        files = sorted(files, key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem)

    rows = []
    for fp in files:
        with fp.open('rb') as f:
            rows.append(pickle.load(f))

    return pd.DataFrame(rows)


def build_i_xtb_columns(
    df: pd.DataFrame,
    xtb_dir: str | Path,
    base_order_135: list[str],
    feature_order_405: list[str],
    require_all_three: bool = False,
    column_prefix: str = 'xtb',
) -> pd.DataFrame:
    '''
    Для строк df подтягивает xtb-блоки по source_file и mol_idx
    '''
    blocks = load_xtb_blocks_indexed(xtb_dir)

    out_rows = []
    v1_list, v2_list, v3_list, vcat_list = [], [], [], []

    for _, r in df.iterrows():
        sf = r['source_file']

        b1 = blocks.get((sf, 1))
        b2 = blocks.get((sf, 2))
        b3 = blocks.get((sf, 3))

        if require_all_three and (b1 is None or b2 is None or b3 is None):
            continue

        v1, v2, v3, vcat = build_row_vectors_from_blocks(
            b1, b2, b3, base_order_135, feature_order_405
        )

        if b1 is None:
            v1 = _empty_vec(len(base_order_135))
        if b2 is None:
            v2 = _empty_vec(len(base_order_135))
        if b3 is None:
            v3 = _empty_vec(len(base_order_135))

        out_rows.append(r)
        v1_list.append(v1)
        v2_list.append(v2)
        v3_list.append(v3)
        vcat_list.append(vcat)

    out = pd.DataFrame(out_rows).reset_index(drop=True)
    out[f'{column_prefix}_1'] = v1_list
    out[f'{column_prefix}_2'] = v2_list
    out[f'{column_prefix}_3'] = v3_list
    out[f'{column_prefix}_concat'] = vcat_list

    return out


def impute_and_scale_concat(
    vcat_list: list[list[float]],
) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    X = np.asarray(vcat_list, dtype=float)
    X_filled = X.copy()
    X_filled[~np.isfinite(X_filled)] = np.nan

    col_medians = np.nanmedian(X_filled, axis=0)
    col_medians = np.where(np.isfinite(col_medians), col_medians, 0.0)

    inds = np.where(~np.isfinite(X_filled))
    X_filled[inds] = np.take(col_medians, inds[1])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_filled)
    return X, X_scaled, scaler


def build_dataset_from_dataframe(
    df: pd.DataFrame,
    get_blocks_fn,
    base_order_135: list[str],
    feature_order_405: list[str],
    require_all_three: bool = True,
    add_scaled_concat: bool = False,
    column_prefix: str = 'xtb',
) -> tuple[pd.DataFrame, StandardScaler | None]:
    '''
    Сборка датасета с xtb-векторами
    '''
    meta_rows = []
    v1_list, v2_list, v3_list, vcat_list = [], [], [], []

    for _, r in df.iterrows():
        b1, b2, b3 = get_blocks_fn(r)

        if require_all_three and (b1 is None or b2 is None or b3 is None):
            continue

        v1, v2, v3, vcat = build_row_vectors_from_blocks(
            b1, b2, b3, base_order_135, feature_order_405
        )

        meta_rows.append(r)
        v1_list.append(v1)
        v2_list.append(v2)
        v3_list.append(v3)
        vcat_list.append(vcat)

    out = pd.DataFrame(meta_rows).reset_index(drop=True)
    out[f'{column_prefix}_1'] = v1_list
    out[f'{column_prefix}_2'] = v2_list
    out[f'{column_prefix}_3'] = v3_list
    out[f'{column_prefix}_concat'] = vcat_list

    scaler = None
    if add_scaled_concat:
        _, X_scaled, scaler = impute_and_scale_concat(vcat_list)
        out[f'{column_prefix}_concat_scaled'] = [row.tolist() for row in X_scaled]

    return out, scaler