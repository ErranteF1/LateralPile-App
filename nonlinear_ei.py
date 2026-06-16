"""
nonlinear_ei.py -- Calibrated nonlinear EI(M) (M-phi) capability for the lateral
                   pile solver.  Built on top of fem_solver (no monkey-patching:
                   it drives the public EI_sections API, one section per element).

Outcome of the calibration study (folder 06): a global EI(M) law forms an
artificial plastic hinge too shallow; a SECTION-DEPENDENT, BILINEAR (elastic ->
plastic plateau) M-phi backbone reproduces LPile's capped-moment / plastic-hinge
behaviour. This module provides that backbone as a reusable, optional solver mode.

Backbone (per element): secant EI_eff = M_backbone(phi)/phi, with
    M_backbone(phi) = EI_g*phi                         for phi <= phi_y   (elastic)
                    = M_y + alpha*EI_g*(phi - phi_y)   for phi >  phi_y   (plastic + hardening)
    phi_y = M_y / EI_g ;  alpha = post-yield hardening ratio (small, e.g. 0.03-0.05)
M_y is SECTION-DEPENDENT (e.g. high for a large shaft, lower for a slender socket).

Two entry points:
    solve_mphi(...)              -- load-controlled (given H) nonlinear-EI analysis
    solve_mphi_target_yhead(...) -- displacement-controlled (bisect H to a target y_head)

Both return the standard fem_solver.extract_results() dict (kip-in units).
"""
import numpy as np
import fem_solver as F


# ---------------------------------------------------------------------------
def _mesh(pile_params):
    L = float(pile_params['L_in']); n_el = int(pile_params['n_elements'])
    z_nodes = np.linspace(0.0, L, n_el + 1)
    zmid = 0.5 * (z_nodes[:-1] + z_nodes[1:])
    return z_nodes, zmid, n_el


def _gross_EI_per_element(pile_params, zmid):
    """Gross (uncracked) EI per element from the base pile definition
       (EI_sections list if present, else scalar EI_kip_in2)."""
    secs = pile_params.get('EI_sections')
    if secs:
        EIg = np.empty(len(zmid))
        for i, zm in enumerate(zmid):
            val = float(secs[-1]['EI_kip_in2'])
            for s in secs:
                if s['z_top_in'] <= zm < s['z_bot_in']:
                    val = float(s['EI_kip_in2']); break
            EIg[i] = val
        return EIg
    return np.full(len(zmid), float(pile_params['EI_kip_in2']))


def _My_per_element(My_sections, zmid, default_My):
    """Section-dependent yield moment per element (by midpoint depth)."""
    My = np.full(len(zmid), float(default_My))
    if My_sections:
        for i, zm in enumerate(zmid):
            for s in My_sections:
                if s['z_top_in'] <= zm < s['z_bot_in']:
                    My[i] = float(s['My_kip_in']); break
    return My


def _EI_sections_from_array(z_nodes, EI_arr):
    """One EI_section per element (so build_EI_array assigns it exactly by midpoint)."""
    return [dict(z_top_in=float(z_nodes[e]), z_bot_in=float(z_nodes[e + 1]),
                 EI_kip_in2=float(EI_arr[e])) for e in range(len(EI_arr))]


def _bilinear_secant(EI_g, M_elem, My, alpha):
    """Secant EI from the bilinear elastic-plastic M-phi backbone, given the
       current element moment M_elem (uses phi = |M|/EI_eff_current implicitly via
       the fixed-point iteration; here we use phi = |M|/EI_g as the backbone abscissa
       is recovered from the converged moment -- consistent at convergence)."""
    M = np.abs(M_elem)
    phi_y = My / EI_g
    phi = M / EI_g                       # curvature consistent with gross-EI moment proxy
    Mbb = np.where(phi <= phi_y, EI_g * phi, My + alpha * EI_g * (phi - phi_y))
    with np.errstate(divide='ignore', invalid='ignore'):
        EIt = np.where(phi > 0, Mbb / phi, EI_g)
    return np.clip(EIt, 0.1 * EI_g, EI_g)


# ---------------------------------------------------------------------------
def solve_mphi(pile_params, soil_layers, H_kip, M_kip_in=0.0, P_axial=0.0,
               My_sections=None, default_My=1.0e30, alpha=0.05,
               relax=0.4, max_outer=40, tol=1.5e-3,
               max_iter=150, solver_tol=1e-6, verbose=False):
    """
    Load-controlled nonlinear-EI (bilinear M-phi) analysis.

    Parameters
    ----------
    pile_params, soil_layers, H_kip, M_kip_in, P_axial : as for fem_solver.solve_nonlinear
    My_sections : list of dicts {z_top_in, z_bot_in, My_kip_in}  (section yield moments)
    default_My  : yield moment where no section is given (default huge -> stays elastic)
    alpha       : post-yield hardening ratio
    relax       : under-relaxation on the EI update (0<relax<=1)
    max_outer   : max outer EI(M) iterations ; tol : convergence on max|dEI|/EI_g

    Returns
    -------
    res : fem_solver.extract_results() dict, with extra keys
          'EI_eff_arr' (per-element converged EI) and 'mphi_outer_iters'.
    """
    z_nodes, zmid, n_el = _mesh(pile_params)
    EI_g = _gross_EI_per_element(pile_params, zmid)
    My   = _My_per_element(My_sections, zmid, default_My)

    pp = dict(pile_params)                    # shallow copy; we override EI_sections
    EI_eff = EI_g.copy()
    res = None
    for it in range(max_outer):
        pp['EI_sections'] = _EI_sections_from_array(z_nodes, EI_eff)
        sol = F.solve_nonlinear(pp, soil_layers, H_kip, M_kip_in, P_axial=P_axial,
                                max_iter=max_iter, tol=solver_tol)
        res = F.extract_results(sol, pp, soil_layers, H_kip, M_kip_in, P_axial=P_axial)
        Mn = res['M_kip_in']
        M_elem = np.maximum(np.abs(Mn[:-1]), np.abs(Mn[1:]))
        EI_target = _bilinear_secant(EI_g, M_elem, My, alpha)
        EI_new = relax * EI_target + (1.0 - relax) * EI_eff
        rel = float(np.max(np.abs(EI_new - EI_eff) / EI_g))
        EI_eff = EI_new
        if verbose:
            print("  M-phi outer %2d  M_max=%.0f  max dEI/EIg=%.2e" % (it, res['M_max_kip_in'], rel))
        if rel < tol:
            break
    res['EI_eff_arr'] = EI_eff
    res['mphi_outer_iters'] = it + 1
    return res


def solve_mphi_target_yhead(pile_params, soil_layers, y_target_in, M_kip_in=0.0,
                            P_axial=0.0, H_lo=10.0, H_hi=1000.0, n_bisect=40, **kw):
    """Displacement-controlled wrapper: bisect H so y_head == y_target_in, with the
       bilinear M-phi backbone active. Returns (H_found, res)."""
    lo, hi = H_lo, H_hi
    res = None
    for _ in range(n_bisect):
        mid = 0.5 * (lo + hi)
        res = solve_mphi(pile_params, soil_layers, mid, M_kip_in, P_axial, **kw)
        if res['y_head_in'] < y_target_in:
            lo = mid
        else:
            hi = mid
        if abs(res['y_head_in'] - y_target_in) < 1e-5:
            break
    return 0.5 * (lo + hi), res
