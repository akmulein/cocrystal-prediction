from pathlib import Path
import pandas as pd

from src.features.smiles_from_xyz import add_smiles_from_xyz

root = Path(__file__).resolve().parents[1]

data = root / 'data'
processed = data / 'processed'

input_path = processed / 'triple_pos_real_xtb.pkl'
triple_out = processed / 'triple_main_pos_real_xtb.pkl'
holdout_out = processed / 'holdout_pos_real_xtb.pkl'

holdout_ids = {
    'KAVDUH', 'VESROD', 'AHAGAT', 'SEMJOL', 'QEKBUE', 'MUWLOG',
    'IRIDAP', 'NEWNIP', 'OREGOK', 'UZITAA', 'XEWWOM', 'XUXZUN',
    'BAVZON', 'DETCDT', 'LUNFEF', 'BERWOO', 'ZATSOC', 'OPEVIR',
    'XOGDOP01', 'CUGPOL', 'ATATUM', 'BURJOR', 'MUHHUS', 'IXOCIK',
    'JUJDID', 'DOSPOR', 'NARGAP', 'KELWAZ', 'CUZBIM', 'JAVPUS',
    'BUWNEO', 'KAJXIC', 'MUYVOS', 'AGAGUN', 'MIGHOC', 'LOWQUJ',
    'LIFXOO', 'CISCEM', 'MIDVAX', 'KAQBIP', 'DOTXUF10', 'LIXWOG',
    'JIRWIR', 'EGAVEN', 'MELPUS', 'CEFWOA', 'KEVSAG', 'MOYKAM',
    'NEMSIH', 'BEYZIQ', 'LUQRAS',
}

df = pd.read_pickle(input_path)

holdout_df = df[df['source_file'].isin(holdout_ids)].copy()
triple_df = df[~df['source_file'].isin(holdout_ids)].copy()

holdout_df = add_smiles_from_xyz(
    holdout_df,
    xyz_cols=['xyz_1_3d', 'xyz_2_3d', 'xyz_3_3d'],
    out_cols=['A', 'B', 'G'],
)

processed.mkdir(parents=True, exist_ok=True)

holdout_df.to_pickle(holdout_out)
triple_df.to_pickle(triple_out)

print(f'shape before: {df.shape}')
print(f'triple shape: {triple_df.shape}')
print(f'holdout shape: {holdout_df.shape}')

print(f'holdout saved to: {holdout_out}')
print(f'triple saved to: {triple_out}')