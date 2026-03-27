from pathlib import Path
import pandas as pd

from src.final_dataset import make_final_dataset

root = Path(__file__).resolve().parents[1]

data = root / 'data'
processed = data / 'processed'
intermediate = data / 'intermediate'
final = data / 'final'

df_pos_path = processed / 'triple_main_pos_real_xtb.pkl'
df_perm_path = intermediate / 'triple_neg_candidates.pkl'
output_path = final / 'triple_real_xtb.pkl'

thresh = 0.7
batch = 5000


def main() -> None:
    df_pos = pd.read_pickle(df_pos_path)
    df_perm = pd.read_pickle(df_perm_path)

    final_cols = [
        'source_file',
        'formula_1', 'xyz_1_3d',
        'formula_2', 'xyz_2_3d',
        'formula_3', 'xyz_3_3d',
        'xtb_1_3d', 'xtb_2_3d', 'xtb_3_3d',
        'xtb_concat_3d',
        'cryst',
    ]

    df_out, df_neg = make_final_dataset(
        df_pos=df_pos,
        df_perm=df_perm,
        xyz_cols=('xyz_1_3d', 'xyz_2_3d', 'xyz_3_3d'),
        feature_cols=('xtb_1_3d', 'xtb_2_3d', 'xtb_3_3d'),
        concat_col='xtb_concat_3d',
        final_cols=final_cols,
        thresh=thresh,
        batch=batch,
    )

    final.mkdir(parents=True, exist_ok=True)
    df_out.to_pickle(output_path, protocol=4)

    print('final shape:', df_out.shape)
    print('saved to:', output_path)


if __name__ == '__main__':
    main()