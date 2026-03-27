import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import GetPeriodicTable

from ase import Atoms
from ase.optimize import BFGS
from xtb.ase.calculator import XTB
from xtb.interface import Calculator, Param
from xtb.utils import get_method


pt = GetPeriodicTable()
angstrom_to_bohr = 1.0 / 0.52917721092


def optimize_geometry_ase(
    numbers,
    positions_ang,
    method='GFN2-xTB',
    fmax=0.05,
    steps=200,
):
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
        'units': {'forces': 'eV/angstrom'},
        'optimization_backend': 'ase_xtb',
    }
    return atoms.get_positions(), meta


def smiles_to_heavyatom_3d(smiles, seed=42, max_iters=500):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f'RDKit cannot parse SMILES: {smiles}')

    params = AllChem.ETKDGv3()
    params.randomSeed = int(seed)
    params.useRandomCoords = True

    code = AllChem.EmbedMolecule(mol, params)
    if code != 0:
        raise ValueError(f'RDKit 3D embedding failed: {smiles}')

    ff_status = 'etkdg_only'
    try:
        if AllChem.MMFFHasAllMoleculeParams(mol):
            AllChem.MMFFOptimizeMolecule(mol, maxIters=int(max_iters))
            ff_status = 'mmff'
        else:
            try:
                AllChem.UFFOptimizeMolecule(mol, maxIters=int(max_iters))
                ff_status = 'uff'
            except Exception:
                ff_status = 'etkdg_only'
    except Exception:
        ff_status = 'etkdg_only'

    conf = mol.GetConformer()
    symbols = [a.GetSymbol() for a in mol.GetAtoms()]
    numbers = np.array([pt.GetAtomicNumber(s) for s in symbols], dtype=int)

    positions_ang = np.array(
        [
            [
                conf.GetAtomPosition(i).x,
                conf.GetAtomPosition(i).y,
                conf.GetAtomPosition(i).z,
            ]
            for i in range(mol.GetNumAtoms())
        ],
        dtype=float,
    )

    return numbers, positions_ang, symbols, ff_status


def geometry_from_smiles(smiles, method='GFN2-xTB', seed=42):
    numbers, positions_ang, symbols, ff_status = smiles_to_heavyatom_3d(smiles, seed=seed)

    try:
        positions_opt_ang, opt_meta = optimize_geometry_ase(
            numbers=numbers,
            positions_ang=positions_ang,
            method=method,
            fmax=0.05,
            steps=200,
        )
        optimization_status = 'xtb_success'
    except Exception:
        positions_opt_ang = positions_ang.copy()
        opt_meta = {
            'converged': False,
            'nsteps': None,
            'fmax_last': None,
            'ase_fmax_threshold': 0.05,
            'ase_steps_max': 200,
            'units': {'forces': 'eV/angstrom'},
            'optimization_backend': 'rdkit_fallback',
        }
        optimization_status = 'xtb_failed_rdkit_geometry_used'

    return {
        'symbols': symbols,
        'numbers': numbers,
        'positions_ang': positions_opt_ang,
        'opt_meta': opt_meta,
        'rdkit_ff_status': ff_status,
        'geometry_status': optimization_status,
    }


def geometry_to_xyz_string(symbols, positions_ang, comment=''):
    lines = [str(len(symbols)), comment]
    for sym, xyz in zip(symbols, positions_ang):
        lines.append(f'{sym} {xyz[0]:.6f} {xyz[1]:.6f} {xyz[2]:.6f}')
    return '\n'.join(lines)


def _safe_get(res, name):
    fn = getattr(res, name, None)
    if fn is None or not callable(fn):
        return None
    try:
        return fn()
    except Exception:
        return None


def _rg_angstrom(positions_ang):
    center = positions_ang.mean(axis=0)
    r = positions_ang - center
    return float(np.sqrt((np.sum(r * r, axis=1)).mean()))


def _homo_lumo(eps, occ, thr=1e-6):
    filled = np.where(occ > thr)[0]
    empty = np.where(occ <= thr)[0]

    homo_i = int(filled.max()) if len(filled) else None
    lumo_i = int(empty.min()) if len(empty) else None
    homo = float(eps[homo_i]) if homo_i is not None else None
    lumo = float(eps[lumo_i]) if lumo_i is not None else None
    gap = (lumo - homo) if (homo is not None and lumo is not None) else None

    return homo_i, lumo_i, homo, lumo, gap


def _bond_order_summaries(bo_mat, topk=200, thr=0.05):
    iu = np.triu_indices(bo_mat.shape[0], k=1)
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
    positions_bohr = positions_ang * angstrom_to_bohr

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
            'bond_orders': 'dimensionless',
        },
    }

    energy = _safe_get(res, 'get_energy')
    out['energy'] = float(energy) if energy is not None else None

    q = _safe_get(res, 'get_charges')
    q = None if q is None else np.asarray(q, dtype=float)
    out['charges'] = q.tolist() if q is not None else None

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
        out['homo_index'] = None
        out['lumo_index'] = None
        out['homo_energy'] = None
        out['lumo_energy'] = None
        out['homo_lumo_gap'] = None
        out['orbital_eigenvalues'] = None
        out['orbital_occupations'] = None

    return out


def xtb_block_from_smiles(smiles, method='GFN2-xTB', seed=42):
    geom = geometry_from_smiles(smiles, method=method, seed=seed)

    try:
        opt = xtb_singlepoint_features(
            geom['numbers'],
            geom['positions_ang'],
            geom['symbols'],
            method=method,
            save_orbitals=True,
        )
        opt['radius_of_gyration_ang'] = _rg_angstrom(geom['positions_ang'])
    except Exception:
        opt = None

    return {
        'opt': opt,
        'opt_meta': geom['opt_meta'],
        'xyz_rebuilt': geometry_to_xyz_string(geom['symbols'], geom['positions_ang']),
        'rdkit_ff_status': geom['rdkit_ff_status'],
        'geometry_status': geom['geometry_status'],
    }