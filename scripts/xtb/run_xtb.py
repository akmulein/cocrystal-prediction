import subprocess
import sys
from pathlib import Path

import pandas as pd


root = Path(__file__).resolve().parents[2]
dataset_path = root / 'data' / 'raw' / 'triple_pos_raw.pkl'
output_dir = root / 'data' / 'xtb' / 'xtb_results'
worker_path = Path(__file__).resolve().parent / 'xtb_worker.py'

df = pd.read_pickle(dataset_path)
output_dir.mkdir(parents=True, exist_ok=True)

all_sources = df['source_file'].unique().tolist()
total = len(all_sources)

done_files = set(
    f.stem
    for f in output_dir.glob('*.pkl')
)

done = len(done_files)

print(f'Total structures: {total}')
print(f'Already processed: {done}')
print('-' * 40)

for idx, sf in enumerate(all_sources, start=1):

    if sf in done_files:
        continue

    subprocess.run(
        [sys.executable, str(worker_path), sf],
        check=False
    )

    done += 1
    left = total - done

    print(
        f'[{done}/{total}] processed | remaining: {left} | last: {sf}',
        flush=True
    )
