from __future__ import annotations

import argparse
from pathlib import Path

from src.features.xtb_features import build_feature_orders, save_feature_orders


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Generate and save canonical xTB feature order files.',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('data/xtb'),
        help='Directory for xtb_feature_order_135.pkl and xtb_feature_order_405.pkl.',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    base_order_135, feature_order_405 = build_feature_orders()
    save_feature_orders(base_order_135, feature_order_405, args.output_dir)

    print(f'feature order 135 size: {len(base_order_135)}')
    print(f'feature order 405 size: {len(feature_order_405)}')
    print(f'saved to: {args.output_dir}')


if __name__ == '__main__':
    main()
