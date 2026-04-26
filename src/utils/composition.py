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


def extract_metals_from_smiles(smiles: str) -> set[str]:
    '''
    Извлекает множество металлов из SMILES.
    Используется как fallback для датасетов без formula_* колонок.
    '''
    if not isinstance(smiles, str) or not smiles.strip():
        return set()

    try:
        from rdkit import Chem
    except ImportError:
        return set()

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return set()

    return {atom.GetSymbol() for atom in mol.GetAtoms() if atom.GetSymbol() in METALS_ALL}


def extract_metals_from_row(
    row: pd.Series,
    formula_cols: list[str] | None = None,
    smiles_cols: list[str] | None = None,
) -> list[str]:
    '''
    Возвращает отсортированный список всех металлов, найденных в строке датасета.
    '''
    metals = set()

    formula_cols = formula_cols or []
    smiles_cols = smiles_cols or []

    for col in formula_cols:
        value = row.get(col)
        if isinstance(value, str):
            metals.update(extract_metals_from_formula(value))

    for col in smiles_cols:
        value = row.get(col)
        if isinstance(value, str):
            metals.update(extract_metals_from_smiles(value))

    return sorted(metals)


def format_metal_name(metals: list[str]) -> object:
    '''
    Превращает список металлов в строку для хранения в колонке metal_name.
    Для систем с несколькими металлами имена соединяются через ";".
    '''
    if not metals:
        return pd.NA
    return ';'.join(metals)


def annotate_metal_metadata(
    df: pd.DataFrame,
    formula_cols: list[str] | None = None,
    smiles_cols: list[str] | None = None,
) -> pd.DataFrame:
    '''
    Добавляет в датасет колонки has_metal и metal_name.
    '''
    out = df.copy()

    if formula_cols is None:
        formula_cols = [c for c in out.columns if c.startswith('formula_')]
    if smiles_cols is None:
        smiles_cols = []

    if not formula_cols and not smiles_cols:
        return out

    metal_lists = out.apply(
        extract_metals_from_row,
        axis=1,
        formula_cols=formula_cols,
        smiles_cols=smiles_cols,
    )

    out['has_metal'] = metal_lists.map(bool)
    out['metal_name'] = metal_lists.map(format_metal_name)
    return out

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
