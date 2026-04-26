from pathlib import Path
import pandas as pd
import numpy as np
from src.utils.composition import annotate_metal_metadata

root = Path(__file__).resolve().parents[1]

data = root / 'data'
processed = data / 'processed'
intermediate = data / 'intermediate'

triple_input = processed / 'triple_main_pos_real_xtb.pkl'
holdout_input = processed / 'holdout_pos_rebuilt_xtb.pkl'

triple_out = intermediate / 'triple_neg_candidates.pkl'
holdout_out = intermediate / 'holdout_neg_candidates.pkl'

n = 300_000
seed = 42


def build_component_pools(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    p1 = (
        df[['formula_1', 'xyz_1_3d']]
        .dropna(subset=['formula_1', 'xyz_1_3d'])
        .drop_duplicates('formula_1')
        .reset_index(drop=True)
    )

    p2 = (
        df[['formula_2', 'xyz_2_3d']]
        .dropna(subset=['formula_2', 'xyz_2_3d'])
        .drop_duplicates('formula_2')
        .reset_index(drop=True)
    )

    p3 = (
        df[['formula_3', 'xyz_3_3d']]
        .dropna(subset=['formula_3', 'xyz_3_3d'])
        .drop_duplicates('formula_3')
        .reset_index(drop=True)
    )

    return p1, p2, p3


def sample_random_triples(
    p1: pd.DataFrame,
    p2: pd.DataFrame,
    p3: pd.DataFrame,
    n: int,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    i1 = rng.integers(0, len(p1), size=n)
    i2 = rng.integers(0, len(p2), size=n)
    i3 = rng.integers(0, len(p3), size=n)

    return pd.DataFrame({
        'formula_1': p1.loc[i1, 'formula_1'].to_numpy(),
        'xyz_1_3d': p1.loc[i1, 'xyz_1_3d'].to_numpy(),
        'formula_2': p2.loc[i2, 'formula_2'].to_numpy(),
        'xyz_2_3d': p2.loc[i2, 'xyz_2_3d'].to_numpy(),
        'formula_3': p3.loc[i3, 'formula_3'].to_numpy(),
        'xyz_3_3d': p3.loc[i3, 'xyz_3_3d'].to_numpy(),
    })

def make_candidates(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    p1, p2, p3 = build_component_pools(df)
    out = sample_random_triples(p1, p2, p3, n=n, seed=seed)
    return annotate_metal_metadata(out, formula_cols=['formula_1', 'formula_2', 'formula_3'])

def main() -> None:
    intermediate.mkdir(parents=True, exist_ok=True)

    triple_df = pd.read_pickle(triple_input)
    triple_candidates = make_candidates(triple_df, n=n, seed=seed)
    triple_candidates.to_pickle(triple_out, protocol=4)

    holdout_df = pd.read_pickle(holdout_input)
    holdout_candidates = make_candidates(holdout_df, n=n, seed=seed)
    holdout_candidates.to_pickle(holdout_out, protocol=4)

    print(f'triple input shape: {triple_df.shape}')
    print(f'{triple_candidates.shape[0]} triple candidates saved to: {triple_out}')

    print(f'holdout input shape: {holdout_df.shape}')
    print(f'{holdout_candidates.shape[0]} holdout candidates saved to: {holdout_out}')

if __name__ == '__main__':
    main()
