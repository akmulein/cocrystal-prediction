# from __future__ import annotations

import numpy as np
import pandas as pd

from src.negative_sampling import (
    build_xyz_feature_map,
    attach_features_by_xyz,
    keep_rows_with_all_features,
    drop_positive_combinations,
    prepare_concat_column,
    fit_transform_by_positives,
    max_cosine_to_positive,
    select_negatives_by_threshold,
    ensure_columns,
)


def is_valid_vector(x) -> bool:
    if x is None:
        return False
    try:
        arr = np.asarray(x, dtype=float)
    except Exception:
        return False
    return arr.size > 0 and np.all(np.isfinite(arr))


def make_final_dataset(
    df_pos: pd.DataFrame,
    df_perm: pd.DataFrame,
    xyz_cols: tuple[str, ...],
    feature_cols: tuple[str, ...],
    concat_col: str,
    final_cols: list[str],
    thresh: float = 0.7,
    batch: int = 5000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    xyz_to_xtb = build_xyz_feature_map(
        df_pos,
        xyz_cols=xyz_cols,
        feature_cols=feature_cols,
    )

    df_perm = attach_features_by_xyz(
        df_perm,
        xyz_to_xtb,
        xyz_cols=xyz_cols,
        out_cols=feature_cols,
    )

    df_perm = keep_rows_with_all_features(df_perm, feature_cols)
    df_pos = keep_rows_with_all_features(df_pos, feature_cols)

    df_perm = drop_positive_combinations(df_perm, df_pos, xyz_cols=xyz_cols)

    df_perm = prepare_concat_column(df_perm, list(feature_cols), concat_col)
    df_pos = prepare_concat_column(df_pos, list(feature_cols), concat_col)

    df_pos = df_pos[df_pos[concat_col].apply(is_valid_vector)].copy()
    df_perm = df_perm[df_perm[concat_col].apply(is_valid_vector)].copy()

    print('positives used:', len(df_pos))
    print('candidates used:', len(df_perm))

    df_pos, df_perm, X_pos_scaled, X_perm_scaled, _ = fit_transform_by_positives(
        df_pos,
        df_perm,
        concat_col=concat_col,
    )

    max_sims = max_cosine_to_positive(X_pos_scaled, X_perm_scaled, batch=batch)

    # if len(df_perm) != len(max_sims):
    #     raise ValueError(
    #         f'Length mismatch: len(df_perm)={len(df_perm)}, len(max_sims)={len(max_sims)}'
    #     )

    df_neg = select_negatives_by_threshold(
        df_perm,
        max_sims,
        thresh=thresh,
        score_col='max_cosine_to_pos_scaled',
    )

    print('negatives kept:', len(df_neg))
    print('threshold:', thresh)

    df_pos_out = df_pos.copy()
    df_pos_out['cryst'] = 1

    df_pos_out = ensure_columns(df_pos_out, final_cols)
    df_neg = ensure_columns(df_neg, final_cols)

    df_out = pd.concat([df_pos_out, df_neg], ignore_index=True)
    return df_out, df_neg
