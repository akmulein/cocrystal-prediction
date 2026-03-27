import numpy as np
import pandas as pd

def build_feature_orders(
    eps_hist_bins: int = 32,
    frontier_k: int = 3,
    thr: tuple[float, ...] = (0.05, 0.1, 0.2),
) -> tuple[list[str], list[str]]:
    '''
    Формирует фиксированный порядок xtb-фичей:
    - base_order_135: порядок для одной молекулы
    - feature_order_405: порядок для трёх конкатенированных молекул
    '''
    base = []

    base += ['present', 'has_opt']
    base += [
        'energy', 'natoms', 'energy_per_atom', 'rg', 'dipole_norm',
        'grad_norm', 'homo', 'lumo', 'gap'
    ]
    base += ['opt_converged', 'opt_nsteps', 'opt_fmax_last', 'opt_fmax_thr']

    base += ['q_n', 'q_mean', 'q_std', 'q_min', 'q_max', 'q_q10', 'q_q50', 'q_q90', 'q_l1', 'q_l2']
    for t in thr:
        base += [f'q_frac_gt_{t}', f'q_frac_lt_-{t}']

    base += ['bo_sum', 'bo_mean', 'bo_max', 'bo_q25', 'bo_q50', 'bo_q75']
    base += ['bo_n_pairs', 'bo_sum_thr', 'bo_mean_thr', 'bo_max_thr', 'bo_sum_norm_natoms']

    for di in range(-frontier_k, 1):
        base += [f'frontier_homo{di:+d}']
    for di in range(0, frontier_k + 1):
        base += [f'frontier_lumo{di:+d}']
    base += ['frontier_n']

    base += [
        'eps_occ_n', 'eps_occ_mean', 'eps_occ_std', 'eps_occ_min', 'eps_occ_max',
        'eps_occ_q10', 'eps_occ_q50', 'eps_occ_q90', 'eps_occ_l1', 'eps_occ_l2'
    ]
    base += [
        'eps_vir_n', 'eps_vir_mean', 'eps_vir_std', 'eps_vir_min', 'eps_vir_max',
        'eps_vir_q10', 'eps_vir_q50', 'eps_vir_q90', 'eps_vir_l1', 'eps_vir_l2'
    ]

    base += [f'eps_occ_hist_{i:03d}' for i in range(1, eps_hist_bins + 1)]
    base += [f'eps_vir_hist_{i:03d}' for i in range(1, eps_hist_bins + 1)]

    order_405 = [f'm{mol}_{b}' for mol in (1, 2, 3) for b in base]
    return base, order_405

def save_feature_orders(
    base_order_135: list[str],
    feature_order_405: list[str],
    out_dir,
) -> None:
    out_dir = pd.io.common.stringify_path(out_dir)
    pd.to_pickle(base_order_135, f'{out_dir}/xtb_feature_order_135.pkl')
    pd.to_pickle(feature_order_405, f'{out_dir}/xtb_feature_order_405.pkl')

def load_feature_orders(
    in_dir,
) -> tuple[list[str], list[str]]:
    in_dir = pd.io.common.stringify_path(in_dir)
    base_order_135 = pd.read_pickle(f'{in_dir}/xtb_feature_order_135.pkl')
    feature_order_405 = pd.read_pickle(f'{in_dir}/xtb_feature_order_405.pkl')
    return list(base_order_135), list(feature_order_405)

def nan() -> float:
    return float('nan')

def safe_float(x) -> float:
    try:
        if x is None:
            return nan()
        return float(x)
    except Exception:
        return nan()

def as_1d_float_list(x) -> list[float] | None:
    if x is None:
        return None
    try:
        arr = np.asarray(x, dtype=float).reshape(-1)
        arr = arr[np.isfinite(arr)]
        return arr.tolist()
    except Exception:
        return None

def summarize_1d_compact(arr, prefix: str) -> dict[str, float]:
    out = {}

    if arr is None:
        out[f'{prefix}_n'] = 0.0
        for k in ['mean', 'std', 'min', 'max', 'q10', 'q50', 'q90', 'l1', 'l2']:
            out[f'{prefix}_{k}'] = nan()
        return out

    a = np.asarray(arr, dtype=float).reshape(-1)
    a = a[np.isfinite(a)]
    out[f'{prefix}_n'] = float(a.size)

    if a.size == 0:
        for k in ['mean', 'std', 'min', 'max', 'q10', 'q50', 'q90', 'l1', 'l2']:
            out[f'{prefix}_{k}'] = nan()
        return out

    out[f'{prefix}_mean'] = float(a.mean())
    out[f'{prefix}_std'] = float(a.std(ddof=0))
    out[f'{prefix}_min'] = float(a.min())
    out[f'{prefix}_max'] = float(a.max())
    out[f'{prefix}_q10'] = float(np.quantile(a, 0.10))
    out[f'{prefix}_q50'] = float(np.quantile(a, 0.50))
    out[f'{prefix}_q90'] = float(np.quantile(a, 0.90))
    out[f'{prefix}_l1'] = float(np.sum(np.abs(a)))
    out[f'{prefix}_l2'] = float(np.sqrt(np.sum(a * a)))
    return out

def fractions_thresholds(
    arr,
    prefix: str,
    thr: tuple[float, ...] = (0.05, 0.1, 0.2),
) -> dict[str, float]:
    out = {}

    if arr is None:
        for t in thr:
            out[f'{prefix}_frac_gt_{t}'] = nan()
            out[f'{prefix}_frac_lt_-{t}'] = nan()
        return out

    a = np.asarray(arr, dtype=float).reshape(-1)
    a = a[np.isfinite(a)]

    if a.size == 0:
        for t in thr:
            out[f'{prefix}_frac_gt_{t}'] = nan()
            out[f'{prefix}_frac_lt_-{t}'] = nan()
        return out

    for t in thr:
        out[f'{prefix}_frac_gt_{t}'] = float(np.mean(a > t))
        out[f'{prefix}_frac_lt_-{t}'] = float(np.mean(a < -t))
    return out

def frontier_levels(
    eps,
    occ,
    k: int = 3,
    prefix: str = 'frontier',
) -> dict[str, float]:
    out = {}

    if eps is None or occ is None:
        for di in range(-k, 1):
            out[f'{prefix}_homo{di:+d}'] = nan()
        for di in range(0, k + 1):
            out[f'{prefix}_lumo{di:+d}'] = nan()
        out[f'{prefix}_n'] = 0.0
        return out

    e = np.asarray(eps, dtype=float).reshape(-1)
    o = np.asarray(occ, dtype=float).reshape(-1)
    m = min(e.size, o.size)
    e, o = e[:m], o[:m]

    mask = np.isfinite(e) & np.isfinite(o)
    e, o = e[mask], o[mask]
    out[f'{prefix}_n'] = float(e.size)

    if e.size == 0:
        for di in range(-k, 1):
            out[f'{prefix}_homo{di:+d}'] = nan()
        for di in range(0, k + 1):
            out[f'{prefix}_lumo{di:+d}'] = nan()
        return out

    occ_idx = np.where(o > 1e-6)[0]
    vir_idx = np.where(o <= 1e-6)[0]
    homo_i = int(occ_idx.max()) if occ_idx.size else None
    lumo_i = int(vir_idx.min()) if vir_idx.size else None

    for di in range(-k, 1):
        if homo_i is None:
            out[f'{prefix}_homo{di:+d}'] = nan()
        else:
            j = homo_i + di
            out[f'{prefix}_homo{di:+d}'] = float(e[j]) if 0 <= j < e.size else nan()

    for di in range(0, k + 1):
        if lumo_i is None:
            out[f'{prefix}_lumo{di:+d}'] = nan()
        else:
            j = lumo_i + di
            out[f'{prefix}_lumo{di:+d}'] = float(e[j]) if 0 <= j < e.size else nan()

    return out

def hist_density(
    arr,
    bins: int = 32,
    rng: tuple[float, float] = (-1.5, 0.5),
    prefix: str = 'hist',
) -> dict[str, float]:
    out = {}
    keys = [f'{prefix}_{i:03d}' for i in range(1, bins + 1)]

    if arr is None:
        for k in keys:
            out[k] = nan()
        return out

    a = np.asarray(arr, dtype=float).reshape(-1)
    a = a[np.isfinite(a)]

    if a.size == 0:
        for k in keys:
            out[k] = nan()
        return out

    hist, _ = np.histogram(a, bins=bins, range=rng, density=True)
    for i, k in enumerate(keys):
        out[k] = float(hist[i])
    return out

def dict_to_vector(d: dict[str, float], feature_order: list[str]) -> list[float]:
    return [float(d.get(k, np.nan)) for k in feature_order]

def xtb_opt_only_features(
    block,
    prefix: str,
    eps_hist_bins: int = 32,
    eps_occ_range: tuple[float, float] = (-1.5, 0.5),
    eps_vir_range: tuple[float, float] = (-0.5, 1.5),
    frontier_k: int = 3,
) -> dict[str, float]:
    out = {}
    out[f'{prefix}_present'] = 0.0 if block is None else 1.0

    opt = None if block is None else block.get('opt')
    opt_meta = {} if block is None else (block.get('opt_meta') or {})

    if opt is None:
        out[f'{prefix}_has_opt'] = 0.0
        for k in ['energy', 'energy_per_atom', 'natoms', 'rg', 'dipole_norm', 'grad_norm', 'homo', 'lumo', 'gap']:
            out[f'{prefix}_{k}'] = nan()
    else:
        out[f'{prefix}_has_opt'] = 1.0
        nat = safe_float(opt.get('natoms'))
        energy = safe_float(opt.get('energy'))

        out[f'{prefix}_energy'] = energy
        out[f'{prefix}_natoms'] = nat
        out[f'{prefix}_energy_per_atom'] = energy / nat if (np.isfinite(energy) and np.isfinite(nat) and nat != 0) else nan()
        out[f'{prefix}_rg'] = safe_float(opt.get('radius_of_gyration_ang'))
        out[f'{prefix}_dipole_norm'] = safe_float(opt.get('dipole_norm'))
        out[f'{prefix}_grad_norm'] = safe_float(opt.get('gradient_norm'))
        out[f'{prefix}_homo'] = safe_float(opt.get('homo_energy'))
        out[f'{prefix}_lumo'] = safe_float(opt.get('lumo_energy'))
        out[f'{prefix}_gap'] = safe_float(opt.get('homo_lumo_gap'))

    out[f'{prefix}_opt_converged'] = float(bool(opt_meta.get('converged'))) if block is not None else 0.0
    out[f'{prefix}_opt_nsteps'] = safe_float(opt_meta.get('nsteps'))
    out[f'{prefix}_opt_fmax_last'] = safe_float(opt_meta.get('fmax_last'))
    out[f'{prefix}_opt_fmax_thr'] = safe_float(opt_meta.get('ase_fmax_threshold'))

    q = as_1d_float_list(opt.get('charges')) if opt else None
    out.update(summarize_1d_compact(q, f'{prefix}_q'))
    out.update(fractions_thresholds(q, f'{prefix}_q', thr=(0.05, 0.1, 0.2)))

    bo = opt.get('bond_orders_summary') if opt else None
    if bo is None:
        out[f'{prefix}_bo_sum'] = nan()
        out[f'{prefix}_bo_mean'] = nan()
        out[f'{prefix}_bo_max'] = nan()
        out[f'{prefix}_bo_q25'] = nan()
        out[f'{prefix}_bo_q50'] = nan()
        out[f'{prefix}_bo_q75'] = nan()
        out[f'{prefix}_bo_n_pairs'] = 0.0
        out[f'{prefix}_bo_sum_thr'] = nan()
        out[f'{prefix}_bo_mean_thr'] = nan()
        out[f'{prefix}_bo_max_thr'] = nan()
        out[f'{prefix}_bo_sum_norm_natoms'] = nan()
    else:
        out[f'{prefix}_bo_sum'] = safe_float(bo.get('bo_sum'))
        out[f'{prefix}_bo_mean'] = safe_float(bo.get('bo_mean'))
        out[f'{prefix}_bo_max'] = safe_float(bo.get('bo_max'))
        out[f'{prefix}_bo_q25'] = safe_float(bo.get('bo_q25'))
        out[f'{prefix}_bo_q50'] = safe_float(bo.get('bo_q50'))
        out[f'{prefix}_bo_q75'] = safe_float(bo.get('bo_q75'))

        pairs = bo.get('bo_top_pairs', []) or []
        vals = np.array([p[2] for p in pairs], dtype=float) if len(pairs) else np.array([], dtype=float)
        vals = vals[np.isfinite(vals)]

        out[f'{prefix}_bo_n_pairs'] = float(vals.size)
        out[f'{prefix}_bo_sum_thr'] = float(vals.sum()) if vals.size else 0.0
        out[f'{prefix}_bo_mean_thr'] = float(vals.mean()) if vals.size else nan()
        out[f'{prefix}_bo_max_thr'] = float(vals.max()) if vals.size else nan()

        nat = out.get(f'{prefix}_natoms', nan())
        bo_sum = out.get(f'{prefix}_bo_sum', nan())
        out[f'{prefix}_bo_sum_norm_natoms'] = bo_sum / nat if (np.isfinite(bo_sum) and np.isfinite(nat) and nat != 0) else nan()

    eps = as_1d_float_list(opt.get('orbital_eigenvalues')) if opt else None
    occ = as_1d_float_list(opt.get('orbital_occupations')) if opt else None

    out.update(frontier_levels(eps, occ, k=frontier_k, prefix=f'{prefix}_frontier'))

    if eps is not None and occ is not None:
        eps_arr = np.asarray(eps, dtype=float)
        occ_arr = np.asarray(occ, dtype=float)
        m = min(eps_arr.size, occ_arr.size)
        eps_arr, occ_arr = eps_arr[:m], occ_arr[:m]
        mask = np.isfinite(eps_arr) & np.isfinite(occ_arr)
        eps_arr, occ_arr = eps_arr[mask], occ_arr[mask]
        occ_mask = occ_arr > 1e-6
        eps_occ = eps_arr[occ_mask]
        eps_vir = eps_arr[~occ_mask]
    else:
        eps_occ, eps_vir = None, None

    out.update(summarize_1d_compact(eps_occ, f'{prefix}_eps_occ'))
    out.update(summarize_1d_compact(eps_vir, f'{prefix}_eps_vir'))
    out.update(hist_density(eps_occ, bins=eps_hist_bins, rng=eps_occ_range, prefix=f'{prefix}_eps_occ_hist'))
    out.update(hist_density(eps_vir, bins=eps_hist_bins, rng=eps_vir_range, prefix=f'{prefix}_eps_vir_hist'))

    return out

def build_row_vectors_from_blocks(
    b1,
    b2,
    b3,
    base_order_135: list[str],
    feature_order_405: list[str],
) -> tuple[list[float], list[float], list[float], list[float]]:
    feat = {}
    feat.update(xtb_opt_only_features(b1, prefix='m1'))
    feat.update(xtb_opt_only_features(b2, prefix='m2'))
    feat.update(xtb_opt_only_features(b3, prefix='m3'))

    v1 = dict_to_vector(feat, [f'm1_{b}' for b in base_order_135])
    v2 = dict_to_vector(feat, [f'm2_{b}' for b in base_order_135])
    v3 = dict_to_vector(feat, [f'm3_{b}' for b in base_order_135])
    vcat = dict_to_vector(feat, feature_order_405)
    return v1, v2, v3, vcat