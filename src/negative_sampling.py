import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def canon3(a, b, c):
    return tuple(sorted((a, b, c)))


def canon_components(*components):
    return tuple(sorted(components))


def concat_features(row: pd.Series, cols: list[str]) -> np.ndarray:
    parts = [row[c] for c in cols]
    return np.asarray(sum(parts, []), dtype=np.float32)


def filter_finite(df: pd.DataFrame, X: np.ndarray) -> tuple[pd.DataFrame, np.ndarray]:
    mask = np.isfinite(X).all(axis=1)
    return df.loc[mask].reset_index(drop=True), X[mask]


def build_xyz_feature_map(
    df: pd.DataFrame,
    xyz_cols: tuple[str, ...],
    feature_cols: tuple[str, ...],
) -> dict[str, object]:
    xyz_to_feat = {}

    for _, row in df.iterrows():
        for xyz_col, feat_col in zip(xyz_cols, feature_cols):
            xyz = row.get(xyz_col)
            feat = row.get(feat_col)

            if not isinstance(xyz, str):
                continue

            if isinstance(feat, list):
                xyz_to_feat.setdefault(xyz, feat)
            elif isinstance(feat, str):
                xyz_to_feat.setdefault(xyz, feat)

    return xyz_to_feat


def attach_features_by_xyz(
    df: pd.DataFrame,
    xyz_to_feat: dict[str, list[float]],
    xyz_cols: tuple[str, ...],
    out_cols: tuple[str, ...],
) -> pd.DataFrame:
    out = df.copy()

    for xyz_col, out_col in zip(xyz_cols, out_cols):
        out[out_col] = out[xyz_col].map(xyz_to_feat)

    return out


def keep_rows_with_all_features(
    df: pd.DataFrame,
    feature_cols: tuple[str, ...],
) -> pd.DataFrame:
    mask = df[feature_cols[0]].notnull()
    for col in feature_cols[1:]:
        mask &= df[col].notnull()
    return df.loc[mask].reset_index(drop=True)


def drop_positive_triples(
    df_perm: pd.DataFrame,
    df_pos: pd.DataFrame,
    xyz_cols: tuple[str, str, str] = ('xyz_1_3d', 'xyz_2_3d', 'xyz_3_3d'),
) -> pd.DataFrame:
    pos_set = set(
        df_pos.apply(lambda r: canon3(r[xyz_cols[0]], r[xyz_cols[1]], r[xyz_cols[2]]), axis=1).to_list()
    )

    perm_can = df_perm.apply(
        lambda r: canon3(r[xyz_cols[0]], r[xyz_cols[1]], r[xyz_cols[2]]), axis=1
    )

    return df_perm.loc[~perm_can.isin(pos_set)].reset_index(drop=True)


def drop_positive_combinations(
    df_perm: pd.DataFrame,
    df_pos: pd.DataFrame,
    xyz_cols: tuple[str, ...],
) -> pd.DataFrame:
    pos_set = set(
        df_pos.apply(lambda r: canon_components(*(r[col] for col in xyz_cols)), axis=1).to_list()
    )

    perm_can = df_perm.apply(
        lambda r: canon_components(*(r[col] for col in xyz_cols)), axis=1
    )

    return df_perm.loc[~perm_can.isin(pos_set)].reset_index(drop=True)


def prepare_concat_column(
    df: pd.DataFrame,
    source_cols: list[str],
    out_col: str,
) -> pd.DataFrame:
    out = df.copy()
    out[out_col] = out.apply(lambda r: concat_features(r, source_cols), axis=1)
    return out


def fit_transform_by_positives(
    df_pos: pd.DataFrame,
    df_perm: pd.DataFrame,
    concat_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, StandardScaler]:
    X_pos = np.vstack(df_pos[concat_col].to_numpy()).astype(np.float32)
    X_perm = np.vstack(df_perm[concat_col].to_numpy()).astype(np.float32)

    df_pos, X_pos = filter_finite(df_pos, X_pos)
    df_perm, X_perm = filter_finite(df_perm, X_perm)

    scaler = StandardScaler()
    X_pos_scaled = scaler.fit_transform(X_pos).astype(np.float32)
    X_perm_scaled = scaler.transform(X_perm).astype(np.float32)

    return df_pos, df_perm, X_pos_scaled, X_perm_scaled, scaler


def max_cosine_to_positive(
    X_pos_scaled: np.ndarray,
    X_perm_scaled: np.ndarray,
    batch: int = 5000,
    eps: float = 1e-12,
) -> np.ndarray:
    X_pos_norm = X_pos_scaled / (np.linalg.norm(X_pos_scaled, axis=1, keepdims=True) + eps)
    X_perm_norm = X_perm_scaled / (np.linalg.norm(X_perm_scaled, axis=1, keepdims=True) + eps)

    X_pos_t = X_pos_norm.T
    max_sims = np.empty((X_perm_norm.shape[0],), dtype=np.float32)

    for start in range(0, X_perm_norm.shape[0], batch):
        end = min(start + batch, X_perm_norm.shape[0])
        sims = X_perm_norm[start:end] @ X_pos_t
        max_sims[start:end] = np.max(sims, axis=1)

    return max_sims


def select_negatives_by_threshold(
    df_perm: pd.DataFrame,
    max_sims: np.ndarray,
    thresh: float,
    score_col: str,
) -> pd.DataFrame:
    mask = max_sims < thresh
    out = df_perm.loc[mask].copy()
    out['cryst'] = 0
    out[score_col] = max_sims[mask]
    return out


# def select_random_subset(
#     df: pd.DataFrame,
#     n: int,
#     seed: int = 42,
# ) -> pd.DataFrame:
#     if len(df) <= n:
#         return df.reset_index(drop=True)
#     return df.sample(n=n, random_state=seed).reset_index(drop=True)


def ensure_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col not in out.columns:
            out[col] = pd.NA
    return out[cols]
