from __future__ import annotations

import argparse
import multiprocessing as mp
import time
from pathlib import Path

import pandas as pd


def source_name_from_cif_path(cif_path: Path) -> str:
    source_name = cif_path.stem
    if source_name.startswith('CSD_CIF_'):
        source_name = source_name[len('CSD_CIF_'):]
    return source_name


def worker_runner(cif_path: str, return_dict) -> None:
    from src.utils.cif_worker import process_one_file

    return_dict['result'] = process_one_file(cif_path)


def run_with_hard_timeout(cif_path: Path, timeout: float) -> dict[str, object]:
    manager = mp.Manager()
    return_dict = manager.dict()
    process = mp.Process(target=worker_runner, args=(str(cif_path), return_dict))
    process.start()
    process.join(timeout)

    if process.is_alive():
        process.terminate()
        process.join()
        return {'status': 'timeout', 'molecules': [], 'error_message': 'timeout'}

    if 'result' not in return_dict:
        return {'status': 'fail', 'molecules': [], 'error_message': 'no result'}

    return dict(return_dict['result'])


def load_set(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return set(path.read_text(encoding='utf-8').splitlines())


def append_to_file(path: Path, line: str) -> None:
    with path.open('a', encoding='utf-8') as handle:
        handle.write(line + '\n')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Parse CIF files into per-structure parquet files with a hard timeout.',
    )
    parser.add_argument(
        '--cif-dir',
        type=Path,
        default=Path('data/cif_parsing/cif'),
        help='Directory with source CIF files.',
    )
    parser.add_argument(
        '--pattern',
        default='CSD_CIF_*.cif',
        help='Glob pattern for CIF files.',
    )
    parser.add_argument(
        '--timeout',
        type=float,
        default=15.0,
        help='Hard timeout per CIF file in seconds.',
    )
    parser.add_argument(
        '--out-dir',
        type=Path,
        default=Path('data/cif_parsing/parsed_cif'),
        help='Directory for parquet outputs and progress logs.',
    )
    parser.add_argument(
        '--done-file',
        default='done.txt',
        help='Filename with successfully processed CIF names.',
    )
    parser.add_argument(
        '--skipped-file',
        default='skipped.txt',
        help='Filename with skipped/failed CIF names.',
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Reprocess files even if they are already listed in done/skipped logs.',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        from src.utils.cif_worker import process_one_file as _process_one_file  # noqa: F401
    except ModuleNotFoundError as exc:
        raise SystemExit(
            f'Missing dependency "{exc.name}". Use the environment from requirements/openbabel_env.txt.'
        ) from exc

    mp.set_start_method('spawn', force=True)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    done_path = out_dir / args.done_file
    skipped_path = out_dir / args.skipped_file

    done = set() if args.force else load_set(done_path)
    skipped = set() if args.force else load_set(skipped_path)

    cif_paths = sorted(args.cif_dir.glob(args.pattern))
    if not cif_paths:
        raise FileNotFoundError(
            f'No CIF files matching "{args.pattern}" found in {args.cif_dir}'
        )

    print(f'CIF files found: {len(cif_paths)}')
    print(f'Output dir: {out_dir}')
    print(f'Timeout per file: {args.timeout:.1f}s')
    print('-' * 40)

    processed_now = 0
    skipped_now = 0

    for cif_path in cif_paths:
        name = cif_path.name
        if not args.force and (name in done or name in skipped):
            continue

        print(f'\n=== Processing: {name} ===')

        start = time.time()
        result = run_with_hard_timeout(cif_path, args.timeout)
        elapsed = time.time() - start

        print(f"Status={result['status']}, time={elapsed:.2f}s")

        if result['status'] in ('ok_default', 'ok_fallback'):
            df = pd.DataFrame(result['molecules'])
            df.insert(0, 'filename', name)
            parquet_name = f'{source_name_from_cif_path(cif_path)}.parquet'
            df.to_parquet(out_dir / parquet_name, index=False)
            append_to_file(done_path, name)
            processed_now += 1
        else:
            append_to_file(skipped_path, name)
            skipped_now += 1

    print('\nParsing finished.')
    print(f'Processed in this run: {processed_now}')
    print(f'Skipped/failed in this run: {skipped_now}')
    print(f'done log: {done_path}')
    print(f'skipped log: {skipped_path}')


if __name__ == '__main__':
    main()
