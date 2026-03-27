import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import sys
import os
import pickle
import numpy as np
import pandas as pd

from xtb.interface import Calculator, Param
from xtb.utils import get_method
from rdkit.Chem import GetPeriodicTable

from ase import Atoms
from ase.optimize import BFGS
from xtb.ase.calculator import XTB

pt = GetPeriodicTable()

ANGSTROM_TO_BOHR = 1.0 / 0.52917721092

def parse_xyz(xyz_block):
    lines = xyz_block.strip().splitlines()
    nat = int(lines[0])
    symbols, pos = [], []
    for line in lines[2:2 + nat]:
        s, x, y, z = line.split()
        symbols.append(s)
        pos.append([float(x), float(y), float(z)])

    numbers = np.array([pt.GetAtomicNumber(s) for s in symbols], dtype=int)     # перевод символов в атомные номера
    positions_ang = np.array(pos, dtype=float)      # координаты в ангстремах
    return numbers, positions_ang, symbols

def _safe_get(res, name):
    fn = getattr(res, name, None)
    if fn is None or not callable(fn):
        return None
    try:
        return fn()
    except Exception:
        return None

# Радиус инерции  
def _rg_angstrom(positions_ang):
    center = positions_ang.mean(axis=0)
    r = positions_ang - center
    return float(np.sqrt((np.sum(r * r, axis=1)).mean()))   # отклонение от центра
   
def _homo_lumo(eps, occ, thr=1e-6): 
    # eps - энергии орбиталей
    # occ - заселенности
    # thr - порог заселенности

    filled = np.where(occ > thr)[0]
    empty = np.where(occ <= thr)[0]
    homo_i = int(filled.max()) if len(filled) else None
    lumo_i = int(empty.min()) if len(empty) else None
    homo = float(eps[homo_i]) if homo_i is not None else None
    lumo = float(eps[lumo_i]) if lumo_i is not None else None
    gap = (lumo - homo) if (homo is not None and lumo is not None) else None
    return homo_i, lumo_i, homo, lumo, gap

def _bond_order_summaries(bo_mat, topk=200, thr=0.05):
    n = bo_mat.shape[0]
    iu = np.triu_indices(n, k=1)
    vals = bo_mat[iu].astype(float)

    out = {
        'bo_sum': float(vals.sum()),
        'bo_mean': float(vals.mean()) if vals.size else None,
        'bo_max': float(vals.max()) if vals.size else None,
        'bo_q25': float(np.quantile(vals, 0.25)) if vals.size else None,
        'bo_q50': float(np.quantile(vals, 0.50)) if vals.size else None,
        'bo_q75': float(np.quantile(vals, 0.75)) if vals.size else None,
    }

    mask = vals >= thr
    if np.any(mask):
        idx = np.where(mask)[0]
        sel_vals = vals[mask]
        order = np.argsort(sel_vals)[::-1][:topk]
        top_idx = idx[order]
        top_vals = sel_vals[order]
        out['bo_top_pairs'] = list(
            zip(iu[0][top_idx].tolist(), iu[1][top_idx].tolist(), top_vals.tolist())
        )
    else:
        out['bo_top_pairs'] = []

    return out

def xtb_singlepoint_features(numbers, positions_ang, symbols, method='GFN2-xTB', save_orbitals=True):
  
    positions_bohr = positions_ang * ANGSTROM_TO_BOHR

    calc = Calculator(Param(get_method(method)), numbers=numbers, positions=positions_bohr)
    res = calc.singlepoint()

    out = {
        'method': method,
        'natoms': int(len(numbers)),
        'elements': symbols,

        'radius_of_gyration_ang': _rg_angstrom(positions_ang),

        'positions_units_handling': 'input_xyz_angstrom -> converted_to_bohr_for_xtb.interface',
        'units': {
            'radius_of_gyration_ang': 'angstrom',
            'energy': 'hartree',
            'orbital_energies': 'hartree',
            'gradient_norm': 'hartree/bohr',
            'dipole': 'e*bohr',
            'charges': 'e',
            'bond_orders': 'dimensionless'
        }
    }

    out['energy'] = float(_safe_get(res, 'get_energy'))

    q = _safe_get(res, 'get_charges')
    q = None if q is None else np.asarray(q, dtype=float)
    out['charges'] = q.tolist() if q is not None else None
    out['charges_mean'] = float(q.mean()) if q is not None else None
    out['charges_std'] = float(q.std(ddof=0)) if q is not None else None
    out['charges_min'] = float(q.min()) if q is not None else None
    out['charges_max'] = float(q.max()) if q is not None else None
    out['charges_q25'] = float(np.quantile(q, 0.25)) if q is not None else None
    out['charges_q50'] = float(np.quantile(q, 0.50)) if q is not None else None
    out['charges_q75'] = float(np.quantile(q, 0.75)) if q is not None else None

    mu = _safe_get(res, 'get_dipole')
    mu = None if mu is None else np.asarray(mu, dtype=float)
    out['dipole'] = mu.tolist() if mu is not None else None
    out['dipole_norm'] = float(np.linalg.norm(mu)) if mu is not None else None

    grad = _safe_get(res, 'get_gradient')
    grad = None if grad is None else np.asarray(grad, dtype=float)
    out['gradient_norm'] = float(np.linalg.norm(grad)) if grad is not None else None

    bo = _safe_get(res, 'get_bond_orders')
    if bo is None:
        out['bond_orders_summary'] = None
    else:
        bo = np.asarray(bo, dtype=float)
        out['bond_orders_summary'] = _bond_order_summaries(bo, topk=200, thr=0.05)

    eps = _safe_get(res, 'get_orbital_eigenvalues')
    occ = _safe_get(res, 'get_orbital_occupations')
    if eps is not None and occ is not None:
        eps = np.asarray(eps, dtype=float)
        occ = np.asarray(occ, dtype=float)

        homo_i, lumo_i, homo, lumo, gap = _homo_lumo(eps, occ)
        out['homo_index'] = homo_i
        out['lumo_index'] = lumo_i
        out['homo_energy'] = homo
        out['lumo_energy'] = lumo
        out['homo_lumo_gap'] = gap

        if save_orbitals:
            out['orbital_eigenvalues'] = eps.tolist()
            out['orbital_occupations'] = occ.tolist()
        else:
            out['orbital_eigenvalues'] = None
            out['orbital_occupations'] = None
    else:
        out['homo_index'] = out['lumo_index'] = None
        out['homo_energy'] = out['lumo_energy'] = out['homo_lumo_gap'] = None
        out['orbital_eigenvalues'] = None
        out['orbital_occupations'] = None

    return out

def optimize_geometry_ase(numbers, positions_ang, method='GFN2-xTB',
                          fmax=0.05, steps=200):
    '''
    Оптимизация геометрии ASE + xtb
    Возвращает оптимизированные координаты в Å и метаданные оптимизации
    '''
    
    symbols = [pt.GetElementSymbol(int(z)) for z in numbers]
    atoms = Atoms(symbols=symbols, positions=positions_ang)

    atoms.calc = XTB(method=method)

    opt = BFGS(atoms, logfile=None)  
    converged = opt.run(fmax=fmax, steps=steps)

    forces = atoms.get_forces()
    fmax_last = float(np.linalg.norm(forces, axis=1).max()) if forces is not None else None

    meta = {
        'converged': bool(converged),
        'nsteps': int(opt.nsteps),
        'fmax_last': fmax_last,
        'ase_fmax_threshold': float(fmax),
        'ase_steps_max': int(steps),
        'units': {'forces': 'eV/angstrom'}
    }
    return atoms.get_positions(), meta  # Å

source_file = sys.argv[1]

df = pd.read_pickle('df_triple.pkl')
row = df[df['source_file'] == source_file].iloc[0]

os.makedirs('xtb_results', exist_ok=True)

out = {'source_file': source_file, 'cryst': 1}

for i in [1, 2, 3]:
    key = f'{i}_xtb'
    try:
        numbers, positions_ang, symbols = parse_xyz(row[f'xyz_{i}'])

        raw = xtb_singlepoint_features(numbers, positions_ang, symbols, method='GFN2-xTB', save_orbitals=True)

        positions_opt_ang, opt_meta = optimize_geometry_ase(
            numbers, positions_ang, method='GFN2-xTB',
            fmax=0.05, steps=200
        )

        if positions_opt_ang is None:
            opt = None
        else:
            opt = xtb_singlepoint_features(numbers, positions_opt_ang, symbols, method='GFN2-xTB', save_orbitals=True)
            
            opt['radius_of_gyration_ang'] = _rg_angstrom(positions_opt_ang)

        out[key] = {
            'raw': raw,
            'opt': opt,
            'opt_meta': opt_meta
        }

    except Exception as e:
        out[key] = None
        out[f'{i}_xtb_error'] = repr(e)

with open(f'xtb_results/{source_file}.pkl', 'wb') as f:
    pickle.dump(out, f)
