from pathlib import Path
import numpy as np
import pandas as pd

from src.negative_sampling import (
    build_xyz_feature_map,
    attach_features_by_xyz,
    keep_rows_with_all_features,
    drop_positive_triples,
    prepare_concat_column,
    fit_transform_by_positives,
    max_cosine_to_positive,
    select_negatives_by_threshold,
    ensure_columns,
)
from src.final_dataset import is_valid_vector

root = Path(__file__).resolve().parents[1]

data = root / 'data'
processed = data / 'processed'
intermediate = data / 'intermediate'
final = data / 'final'

df_pos_path = processed / 'holdout_pos_rebuilt_xtb.pkl'
df_perm_path = intermediate / 'holdout_neg_candidates.pkl'
output_path = final / 'holdout_rebuilt_xtb.pkl'

thresh = 0.7
batch = 5000
n_neg = 1000
seed = 42


def build_formula_lookup(df, formula_col, value_cols):
    cols = [formula_col] + value_cols
    tmp = (
        df[cols]
        .dropna(subset=[formula_col])
        .drop_duplicates(subset=[formula_col], keep='first')
    )
    return tmp.set_index(formula_col)[value_cols].to_dict(orient='index')


def attach_rebuilt_by_formula(df, lookup, formula_col, out_cols):
    out = df.copy()
    mapped = out[formula_col].map(lookup)

    for col in out_cols:
        out[col] = mapped.apply(lambda x: x.get(col) if isinstance(x, dict) else None)

    return out


def main() -> None:
    df_pos = pd.read_pickle(df_pos_path)
    df_perm = pd.read_pickle(df_perm_path)

    xyz_to_xtb = build_xyz_feature_map(
        df_pos,
        xyz_cols=('xyz_1_3d', 'xyz_2_3d', 'xyz_3_3d'),
        feature_cols=('xtb_1_3d', 'xtb_2_3d', 'xtb_3_3d'),
    )

    df_perm = attach_features_by_xyz(
        df_perm,
        xyz_to_xtb,
        xyz_cols=('xyz_1_3d', 'xyz_2_3d', 'xyz_3_3d'),
        out_cols=('xtb_1_3d', 'xtb_2_3d', 'xtb_3_3d'),
    )

    df_perm = keep_rows_with_all_features(df_perm, ('xtb_1_3d', 'xtb_2_3d', 'xtb_3_3d'))
    df_pos = keep_rows_with_all_features(df_pos, ('xtb_1_3d', 'xtb_2_3d', 'xtb_3_3d'))

    df_perm = drop_positive_triples(df_perm, df_pos)

    df_perm = prepare_concat_column(
        df_perm,
        ['xtb_1_3d', 'xtb_2_3d', 'xtb_3_3d'],
        'xtb_concat_3d',
    )
    df_pos = prepare_concat_column(
        df_pos,
        ['xtb_1_3d', 'xtb_2_3d', 'xtb_3_3d'],
        'xtb_concat_3d',
    )

    df_pos = df_pos[df_pos['xtb_concat_3d'].apply(is_valid_vector)].copy()
    df_perm = df_perm[df_perm['xtb_concat_3d'].apply(is_valid_vector)].copy()

    print('positives used:', len(df_pos))
    print('candidates used:', len(df_perm))

    df_pos, df_perm, X_pos_scaled, X_perm_scaled, _ = fit_transform_by_positives(
        df_pos,
        df_perm,
        concat_col='xtb_concat_3d',
    )

    max_sims = max_cosine_to_positive(X_pos_scaled, X_perm_scaled, batch=batch)

    if len(df_perm) != len(max_sims):
        raise ValueError(
            f'Length mismatch: len(df_perm)={len(df_perm)}, len(max_sims)={len(max_sims)}'
        )

    df_neg = select_negatives_by_threshold(
        df_perm,
        max_sims,
        thresh=thresh,
        score_col='max_cosine_to_pos_scaled',
    )

    print('negatives after threshold:', len(df_neg))
    print('threshold:', thresh)

    if len(df_neg) < n_neg:
        raise ValueError(
            f'Not enough negatives after thresholding: got {len(df_neg)}, need {n_neg}'
        )

    df_neg = df_neg.sample(n=n_neg, random_state=seed).copy()

    lookup_1 = build_formula_lookup(
        df_pos,
        'formula_1',
        ['A', 'xyz_1_rebuilt', 'xtb_1_rebuilt'],
    )
    lookup_2 = build_formula_lookup(
        df_pos,
        'formula_2',
        ['B', 'xyz_2_rebuilt', 'xtb_2_rebuilt'],
    )
    lookup_3 = build_formula_lookup(
        df_pos,
        'formula_3',
        ['G', 'xyz_3_rebuilt', 'xtb_3_rebuilt'],
    )

    df_neg = attach_rebuilt_by_formula(
        df_neg, lookup_1, 'formula_1', ['A', 'xyz_1_rebuilt', 'xtb_1_rebuilt']
    )
    df_neg = attach_rebuilt_by_formula(
        df_neg, lookup_2, 'formula_2', ['B', 'xyz_2_rebuilt', 'xtb_2_rebuilt']
    )
    df_neg = attach_rebuilt_by_formula(
        df_neg, lookup_3, 'formula_3', ['G', 'xyz_3_rebuilt', 'xtb_3_rebuilt']
    )

    df_pos = prepare_concat_column(
        df_pos,
        ['xtb_1_rebuilt', 'xtb_2_rebuilt', 'xtb_3_rebuilt'],
        'xtb_concat_rebuilt',
    )
    df_neg = prepare_concat_column(
        df_neg,
        ['xtb_1_rebuilt', 'xtb_2_rebuilt', 'xtb_3_rebuilt'],
        'xtb_concat_rebuilt',
    )

    df_pos = df_pos[df_pos['xtb_concat_rebuilt'].apply(is_valid_vector)].copy()
    df_neg = df_neg[df_neg['xtb_concat_rebuilt'].apply(is_valid_vector)].copy()

    df_pos_out = df_pos.copy()
    df_pos_out['cryst'] = 1
    df_neg['cryst'] = 0

    final_cols = [
        'source_file',
        'A', 'B', 'G',
        'formula_1', 'xyz_1_3d', 'xyz_1_rebuilt',
        'formula_2', 'xyz_2_3d', 'xyz_2_rebuilt',
        'formula_3', 'xyz_3_3d', 'xyz_3_rebuilt',
        'xtb_1_3d', 'xtb_2_3d', 'xtb_3_3d',
        'xtb_concat_3d',
        'xtb_1_rebuilt', 'xtb_2_rebuilt', 'xtb_3_rebuilt',
        'xtb_concat_rebuilt',
        'cryst',
    ]

    df_pos_out = ensure_columns(df_pos_out, final_cols)
    df_neg = ensure_columns(df_neg, final_cols)

    df_out = pd.concat([df_pos_out, df_neg], ignore_index=True)

    final.mkdir(parents=True, exist_ok=True)
    df_out.to_pickle(output_path, protocol=4)

    print('final shape:', df_out.shape)
    print(f'saved to: {output_path}')


if __name__ == '__main__':
    main()