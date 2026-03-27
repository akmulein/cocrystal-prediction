
import os
import subprocess
import pandas as pd

df = pd.read_pickle('df_triple.pkl')
os.makedirs('xtb_results', exist_ok=True)

all_sources = df['source_file'].unique().tolist()
total = len(all_sources)

done_files = set(
    f.replace('.pkl', '')
    for f in os.listdir('xtb_results')
    if f.endswith('.pkl')
)

done = len(done_files)

print(f'Total structures: {total}')
print(f'Already processed: {done}')
print('-' * 40)

for idx, sf in enumerate(all_sources, start=1):

    if sf in done_files:
        continue

    subprocess.run(
        ['python', 'xtb_worker.py', sf],
        check=False
    )

    done += 1
    left = total - done

    print(
        f'[{done}/{total}] processed | remaining: {left} | last: {sf}',
        flush=True
    )
