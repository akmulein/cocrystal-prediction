from pathlib import Path
import pandas as pd

from src.features.xtb_features import load_feature_orders, build_row_vectors_from_blocks
from src.features.smiles_geometry import xtb_block_from_smiles

root = Path(__file__).resolve().parents[1]

data = root / "data"
raw = data / "raw"
processed = data / "processed"
final = data / "final"
xtb_dir = data / "xtb"

holdout_input = processed / "holdout_pos_real_xtb.pkl"
sulf_input = raw / "sulf_raw.pkl"

holdout_out = processed / "holdout_pos_rebuilt_xtb.pkl"
sulf_out = final / "sulf_rebuilt_xtb.pkl"


def add_rebuilt_xtb(df: pd.DataFrame, orders_dir: Path) -> pd.DataFrame:
    base_order_135, feature_order_405 = load_feature_orders(orders_dir)

    xtb_1_rebuilt, xtb_2_rebuilt, xtb_3_rebuilt, xtb_concat_rebuilt = [], [], [], []
    xyz_1_rebuilt, xyz_2_rebuilt, xyz_3_rebuilt = [], [], []

    for _, row in df.iterrows():
        b1 = xtb_block_from_smiles(row["A"], seed=43)
        b2 = xtb_block_from_smiles(row["B"], seed=44)
        b3 = xtb_block_from_smiles(row["G"], seed=45)

        v1, v2, v3, vcat = build_row_vectors_from_blocks(
            b1, b2, b3, base_order_135, feature_order_405
        )

        xtb_1_rebuilt.append(v1)
        xtb_2_rebuilt.append(v2)
        xtb_3_rebuilt.append(v3)
        xtb_concat_rebuilt.append(vcat)

        xyz_1_rebuilt.append(b1["xyz_rebuilt"])
        xyz_2_rebuilt.append(b2["xyz_rebuilt"])
        xyz_3_rebuilt.append(b3["xyz_rebuilt"])

    out = df.copy()

    out["xyz_1_rebuilt"] = xyz_1_rebuilt
    out["xyz_2_rebuilt"] = xyz_2_rebuilt
    out["xyz_3_rebuilt"] = xyz_3_rebuilt

    out["xtb_1_rebuilt"] = xtb_1_rebuilt
    out["xtb_2_rebuilt"] = xtb_2_rebuilt
    out["xtb_3_rebuilt"] = xtb_3_rebuilt
    out["xtb_concat_rebuilt"] = xtb_concat_rebuilt

    return out


def main() -> None:
    processed.mkdir(parents=True, exist_ok=True)
    final.mkdir(parents=True, exist_ok=True)

    holdout_df = pd.read_pickle(holdout_input)
    holdout_rebuilt = add_rebuilt_xtb(holdout_df, xtb_dir)
    holdout_rebuilt.to_pickle(holdout_out, protocol=4)

    sulf_df = pd.read_pickle(sulf_input)
    sulf_rebuilt = add_rebuilt_xtb(sulf_df, xtb_dir)
    sulf_rebuilt.to_pickle(sulf_out, protocol=4)

    print(f"holdout shape: {holdout_rebuilt.shape}")
    print(f"holdout saved to: {holdout_out}")

    print(f"sulf shape: {sulf_rebuilt.shape}")
    print(f"sulf saved to: {sulf_out}")


if __name__ == "__main__":
    main()