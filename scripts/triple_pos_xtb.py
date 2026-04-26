from pathlib import Path
import pandas as pd

from src.features.xtb_features import load_feature_orders
from src.features.xtb_dataset import load_xtb_rows_from_dir, build_dataset_from_dataframe
from src.utils.composition import annotate_metal_metadata

root = Path(__file__).resolve().parents[1]

data = root / 'data'
raw = data / 'raw'
processed = data / 'processed'
xtb_dir = data / 'xtb' / 'xtb_results'
orders_dir = data / 'xtb'

triple_path = raw / 'triple_pos_raw.pkl'
output_path = processed / 'triple_pos_real_xtb.pkl'


def main() -> None:
    base_order_135, feature_order_405 = load_feature_orders(orders_dir)

    df_triple = pd.read_pickle(triple_path)
    df_xtb = load_xtb_rows_from_dir(xtb_dir)
    raw_meta_cols = ['source_file', 'formula_1', 'formula_2', 'formula_3', 'xyz_1', 'xyz_2', 'xyz_3']
    raw_meta_cols += [c for c in ['has_metal', 'metal_name'] if c in df_triple.columns]

    df = df_xtb.merge(
        df_triple[raw_meta_cols].drop_duplicates('source_file'),
        on='source_file',
        how='left',
    )

    if 'has_metal' not in df.columns or 'metal_name' not in df.columns:
        df = annotate_metal_metadata(df, formula_cols=['formula_1', 'formula_2', 'formula_3'])

    if 'cryst' not in df.columns:
        df['cryst'] = 1

    df_out, _ = build_dataset_from_dataframe(
        df=df,
        get_blocks_fn=lambda r: (r.get('1_xtb'), r.get('2_xtb'), r.get('3_xtb')),
        base_order_135=base_order_135,
        feature_order_405=feature_order_405,
        require_all_three=True,
        add_scaled_concat=False,  
        column_prefix='xtb',
    )

    rename_map = {
        'xyz_1': 'xyz_1_3d',
        'xyz_2': 'xyz_2_3d',
        'xyz_3': 'xyz_3_3d',
        'xtb_1': 'xtb_1_3d',
        'xtb_2': 'xtb_2_3d',
        'xtb_3': 'xtb_3_3d',
        'xtb_concat': 'xtb_concat_3d',
    }

    df_out = df_out.rename(columns=rename_map)

    keep = [
        'source_file',
        'formula_1', 'formula_2', 'formula_3',
        'xyz_1_3d', 'xyz_2_3d', 'xyz_3_3d',
        'xtb_1_3d', 'xtb_2_3d', 'xtb_3_3d',
        'xtb_concat_3d',
        'has_metal', 'metal_name',
        'cryst',
    ]

    keep = [c for c in keep if c in df_out.columns]

    df_final = df_out[keep]

    processed.mkdir(parents=True, exist_ok=True)
    df_final.to_pickle(output_path, protocol=4)

    print(df_final.shape)
    print(f'saved to: {output_path}')


if __name__ == '__main__':
    main()
