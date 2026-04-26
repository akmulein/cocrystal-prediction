from __future__ import annotations

import argparse
from pathlib import Path

from src.utils.csd_raw_builder import build_csd_raw_dataframe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Aggregate per-CIF parquet files into one raw CSD dataframe.',
    )
    parser.add_argument(
        '--input-dir',
        type=Path,
        default=Path('data/cif_parsing/parsed_cif'),
        help='Directory with parsed parquet files.',
    )
    parser.add_argument(
        '--pattern',
        default='*.parquet',
        help='Glob pattern for parsed CIF parquet files.',
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('data/raw/csd_raw.pkl'),
        help='Target pickle path for the aggregated raw dataframe.',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    df = build_csd_raw_dataframe(
        input_dir=args.input_dir,
        pattern=args.pattern,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(args.output, protocol=4)

    print(df.shape)
    print(f'saved to: {args.output}')


if __name__ == '__main__':
    main()
