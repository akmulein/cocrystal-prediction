import os
import subprocess
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

N_WORKERS = 12  

df = pd.read_pickle('df_triple.pkl')
os.makedirs('xtb_results', exist_ok=True)

all_sources = df['source_file'].unique().tolist()
done_files = set(
    f.replace('.pkl', '')
    for f in os.listdir('xtb_results')
    if f.endswith('.pkl')
)
todo = [sf for sf in all_sources if sf not in done_files]

total = len(all_sources)
done = len(done_files)

print(f'Total structures: {total}')
print(f'Already processed: {done}')
print(f'To do: {len(todo)}')
print('-' * 40)

def run_one(sf):
    
    r = subprocess.run(['python', 'xtb_worker.py', sf], check=False)
    return sf, r.returncode

with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
    futures = [ex.submit(run_one, sf) for sf in todo]
    for k, fut in enumerate(as_completed(futures), start=1):
        sf, code = fut.result()
        done += 1
        left = total - done
        status = 'ok' if code == 0 else f'err({code})'
        print(f'[{done}/{total}] {status} | remaining: {left} | last: {sf}', flush=True)
