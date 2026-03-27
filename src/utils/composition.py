import re
from collections import Counter
import pandas as pd

_EL_CNT = re.compile(r'([A-Z][a-z]?)(\d*)')

D_METALS = {
    'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
    'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
    'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
    'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn',
}

F_METALS = {
    'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu',
    'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr',
}

DF_METALS = D_METALS | F_METALS

METALS_ALL = DF_METALS | {
    'Li', 'Na', 'K', 'Rb', 'Cs', 'Fr',
    'Be', 'Mg', 'Ca', 'Sr', 'Ba', 'Ra',
    'Al', 'Ga', 'In', 'Tl',
    'Sn', 'Pb', 'Bi',
    'Ge', 'As', 'Sb', 'Te',
}

def parse_formula_counts(formula: str) -> Counter:
    '''
    Возвращает число атомов каждого элемента из формулы.
    '''
    cnt = Counter()
    for el, n in _EL_CNT.findall(formula):
        cnt[el] += int(n) if n else 1
    return cnt

def _row_metal_atom_counts(row: pd.Series) -> tuple[int, int, int, int]:
    '''
    Считает число атомов металлов в одной строке датасета.
    '''
    metal_all = df_atoms = d_atoms = f_atoms = 0

    for c in row.index:
        if not c.startswith('formula_') or not isinstance(row[c], str):
            continue

        cnt = parse_formula_counts(row[c])
        for el, n in cnt.items():
            if el in METALS_ALL:
                metal_all += n
            if el in D_METALS:
                d_atoms += n
                df_atoms += n
            elif el in F_METALS:
                f_atoms += n
                df_atoms += n

    return metal_all, df_atoms, d_atoms, f_atoms

def _row_flags(row: pd.Series, metal_col: str, df_col: str) -> pd.Series:
    '''
    Ставит флаги для строки датасета на основе состава компонентов.
    Определяет наличие:
    - системы с одним металлом, амином и сульфокислотой,
    - системы с одним d/f-металлом, амином и сульфокислотой,
    - любой комбинации амин + сульфокислота.
    '''
    sulfonic = amine = 0
    total_counts = Counter()

    for c in row.index:
        if not c.startswith('formula_') or not isinstance(row[c], str):
            continue

        cnt = parse_formula_counts(row[c])
        total_counts += cnt

        if cnt.get('S', 0) >= 1 and cnt.get('O', 0) >= 3:
            sulfonic += 1

        if cnt.get('N', 0) >= 1:
            amine += 1

    return pd.Series(
        {
            'single_metal_sulfonic_amine':
                (row[metal_col] == 1) and (sulfonic == 1) and (amine == 1),
            'single_df_sulfonic_amine':
                (row[df_col] == 1)
                and (row[metal_col] == 1)
                and (sulfonic == 1)
                and (amine == 1),
            'amine_sulfonic_any':
                (sulfonic == 1) and (amine == 1),
        }
    )

def extract_metals_from_formula(formula: str) -> set[str]:
    '''
    Извлекает множество металлов из формулы.
    '''
    clean = re.sub(r'[\[\]\+\-\(\)]', '', formula)
    elements = {el for el, _ in _EL_CNT.findall(clean)}
    return elements & METALS_ALL

def metal_summary(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Строит сводку по металлам, встречающимся в системах датасета.
    '''
    metals = []

    for _, row in df.filter(regex=r'^formula_').iterrows():
        row_metals = set()

        for formula in row.dropna():
            if isinstance(formula, str):
                row_metals.update(extract_metals_from_formula(formula))

        metals.extend(row_metals)

    if not metals:
        return pd.DataFrame(columns=['metal', 'n_combinations'])

    return (
        pd.Series(metals, name='metal')
        .value_counts()
        .rename_axis('metal')
        .reset_index(name='n_combinations')
    )