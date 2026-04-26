import subprocess
import sys
from pathlib import Path

import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

N_WORKERS = 12  

root = Path(__file__).resolve().parents[2]
dataset_path = root / 'data' / 'raw' / 'triple_pos_raw.pkl'
output_dir = root / 'data' / 'xtb' / 'xtb_results'
worker_path = Path(__file__).resolve().parent / 'xtb_worker.py'

df = pd.read_pickle(dataset_path)
output_dir.mkdir(parents=True, exist_ok=True)

all_sources = df['source_file'].unique().tolist()
done_files = set(
    f.stem
    for f in output_dir.glob('*.pkl')
)
todo = [sf for sf in all_sources if sf not in done_files]

total = len(all_sources)
done = len(done_files)

print(f'Total structures: {total}')
print(f'Already processed: {done}')
print(f'To do: {len(todo)}')
print('-' * 40)

def run_one(sf):
    
    r = subprocess.run([sys.executable, str(worker_path), sf], check=False)
    return sf, r.returncode

with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
    futures = [ex.submit(run_one, sf) for sf in todo]
    for k, fut in enumerate(as_completed(futures), start=1):
        sf, code = fut.result()
        done += 1
        left = total - done
        status = 'ok' if code == 0 else f'err({code})'
        print(f'[{done}/{total}] {status} | remaining: {left} | last: {sf}', flush=True)
