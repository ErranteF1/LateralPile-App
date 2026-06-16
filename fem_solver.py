"""
fem_solver.py  —  Hermitian Beam-on-Winkler-Foundation FEM Solver
==================================================================
Internal unit system: kip, in throughout.

Theory: RSPile Laterally Loaded Piles — Theory Manual (Rocscience, 2022)
        Hetenyi (1946)  governing ODE for beam-column on Winkler foundation

Axial load P (beam-column geometric stiffness) — §2 RSPile
-----------------------------------------------------------
Governing equation (Eq. 1):
    EpIp·d⁴y/dz⁴  +  Px·d²y/dz²  +  Epy·y  =  W

The Px·d²y/dz² term is the classical beam-column P-delta (geometric) effect.
In the FEM weak form, it contributes:

    δW_geo = −∫₀ᴸ P·(dy/dz)·(dδy/dz) dz

giving the element geometric stiffness matrix:

    K_g = P·∫₀ᴸ (dN/dz)(dN/dz)ᵀ dz
        = P/(30·Le) · [[36,  3Le, −36,  3Le],
                        [ 3Le, 4Le², −3Le, −Le²],
                        [−36, −3Le,  36, −3Le],
                        [ 3Le, −Le², −3Le, 4Le²]]

Assembly: K_total = K_beam + K_soil − K_g

    P > 0  compressive  →  −K_g reduces effective stiffness  →  larger y, M
    P < 0  tensile      →  −K_g increases effective stiffness →  smaller y, M
    P = 0  no axial     →  recovers original solver exactly

Key features
------------
- Euler-Bernoulli beam, 2 DOFs/node: lateral deflection y [in], rotation θ [rad]
- Hermitian cubic shape functions
- 2-point Gauss quadrature for Winkler soil stiffness and shear recovery
- Secant stiffness iteration (p-y method engine)
- Layered soil profile: arbitrary number of layers, each with its own p-y model
- Fixed or free pile head; free/pinned/fixed tip
- Returns all internal forces and displacements at every node

Sign conventions
----------------
    z  → positive downward (0 = head, L = tip)
    y  → positive in direction of applied lateral force H
    θ  = dy/dz
    M  = EI · d²y/dz²   (positive = tension on +z face)
    V  = EI · d³y/dz³   (V(0) = H_head by construction)
    p  → acts opposite to y (resists deflection); dV/dz = −p

All kip-in unless noted.
"""

import numpy as np
import warnings
from py_models import get_soil_response, compute_pult, _XI_GP, _W_GP

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 0 — GAUSS POINT AND SHAPE FUNCTION UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def hermitian_shape_N(zeta, Le):
    """
    Hermitian cubic shape functions at normalised coordinate zeta = s/Le ∈ [0,1].
    DOF order: [y_i, θ_i, y_j, θ_j].
    Returns ndarray N[4].
    """
    z2 = zeta ** 2
    z3 = zeta ** 3
    return np.array([
        1.0 - 3.0 * z2 + 2.0 * z3,          # N1 — deflection at node i
        Le  * zeta * (1.0 - zeta) ** 2,      # N2 — rotation    at node i
        3.0 * z2 - 2.0 * z3,                 # N3 — deflection at node j
        Le  * z2   * (zeta - 1.0)            # N4 — rotation    at node j
    ])


def _gauss_points_on_element(Le):
    """
    2-point Gauss data on element [0, Le].
    Returns (s_gp [in],  zeta_gp [-])  each shape (2,).
    Mapping: ξ ∈ [−1,1] → s = (Le/2)·(ξ+1)
    """
    s_gp    = (Le / 2.0) * (_XI_GP + 1.0)
    zeta_gp = s_gp / Le
    return s_gp, zeta_gp


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — ELEMENT STIFFNESS MATRICES
# ══════════════════════════════════════════════════════════════════════════════

def beam_element_stiffness(EI, Le):
    """
    4×4 Euler-Bernoulli beam stiffness matrix.
    DOF order: [y_i, θ_i, y_j, θ_j].  Units: kip/in, kip, kip·in, ...
    """
    c = EI / Le ** 3
    L = Le
    return c * np.array([
        [ 12.0,    6.0*L,   -12.0,    6.0*L],
        [  6.0*L,  4.0*L**2, -6.0*L,  2.0*L**2],
        [-12.0,   -6.0*L,    12.0,   -6.0*L],
        [  6.0*L,  2.0*L**2, -6.0*L,  4.0*L**2]
    ])


def soil_element_stiffness_gauss(k_gp1, k_gp2, Le):
    """
    4×4 Gauss-integrated Winkler soil stiffness matrix.

        K_soil,e = (Le/2) · Σ_g  w_g · N(ζ_g)^T ⊗ N(ζ_g) · k_sec(ζ_g)

    Parameters
    ----------
    k_gp1 : float  [kip/in²]  secant stiffness at Gauss point 1
    k_gp2 : float  [kip/in²]  secant stiffness at Gauss point 2
    Le    : float  [in]        element length

    Returns
    -------
    K_soil : ndarray (4,4)  [kip/in]
    """
    Jac    = Le / 2.0
    K_soil = np.zeros((4, 4))
    s_gp, zeta_gp = _gauss_points_on_element(Le)
    for g, k_s in enumerate([k_gp1, k_gp2]):
        N = hermitian_shape_N(zeta_gp[g], Le)
        K_soil += Jac * _W_GP[g] * k_s * np.outer(N, N)
    return K_soil


def geometric_stiffness_element(P_axial, Le):
    """
    4×4 beam-column geometric stiffness matrix (consistent form).

    Derived from the weak form of the Px·d²y/dz² term:

        K_g = P·∫₀ᴸᵉ (dN/dz)ᵀ(dN/dz) dz
            = P/(30·Le) · [[36,   3Le,  -36,   3Le ],
                            [ 3Le, 4Le², -3Le, -Le² ],
                            [-36, -3Le,   36,  -3Le ],
                            [ 3Le, -Le², -3Le,  4Le²]]

    Assembled as:  K_total = K_beam + K_soil − K_g

        P > 0  compressive  →  reduces stiffness  →  larger deflection and moment
        P < 0  tensile      →  increases stiffness →  smaller deflection and moment
        P = 0  →  zero matrix, recovers pure lateral solver exactly

    Parameters
    ----------
    P_axial : float  [kip]   axial load  (positive = compressive)
    Le      : float  [in]    element length

    Returns
    -------
    K_g : ndarray (4,4)  [kip/in]
    """
    if P_axial == 0.0:
        return np.zeros((4, 4))

    L  = Le
    L2 = L * L
    c  = P_axial / (30.0 * L)

    return c * np.array([
        [ 36.0,    3.0*L,   -36.0,    3.0*L],
        [  3.0*L,  4.0*L2,   -3.0*L, -1.0*L2],
        [-36.0,   -3.0*L,    36.0,   -3.0*L],
        [  3.0*L, -1.0*L2,   -3.0*L,  4.0*L2]
    ])


def estimate_critical_load(EI, L_in, tip_cond='free'):
    """
    Classical Euler critical load for a pile treated as a column.

    Boundary conditions (pile head = free to rotate unless stated otherwise):

        'free'   → cantilever, K_eff = 2.0  → P_cr = π²·EI / (2·L)²  =  π²·EI / (4·L²)
        'pinned' → pin-pin,    K_eff = 1.0  → P_cr = π²·EI / L²
        'fixed'  → fix-fix,    K_eff = 0.5  → P_cr = 4·π²·EI / L²

    Note: This is a conservative free-field estimate.  The actual buckling load
    with Winkler soil support is always higher.

    Parameters
    ----------
    EI      : float  [kip·in²]
    L_in    : float  [in]
    tip_cond: str    'free' | 'pinned' | 'fixed'

    Returns
    -------
    P_cr : float  [kip]
    """
    K_eff_map = {'free': 2.0, 'pinned': 1.0, 'fixed': 0.5}
    K_eff = K_eff_map.get(tip_cond, 2.0)
    return (np.pi ** 2 * EI) / (K_eff * L_in) ** 2


def build_EI_array(pile_params, z_nodes):
    """
    Build a per-element EI array from pile_params.

    If pile_params contains 'EI_sections' (a list of structural section dicts),
    each FEM element is assigned the EI of the section that contains its midpoint.
    This supports any number of pile structural sections by depth — the general
    variable-EI implementation.

    If 'EI_sections' is absent or empty, the scalar 'EI_kip_in2' is broadcast
    uniformly to all elements (backward-compatible with single-section piles).

    Parameters
    ----------
    pile_params : dict
        Option A — variable sections:
          'EI_sections'  : list of dicts, each with keys:
                             'z_top_in'    float  top depth of section [in]
                             'z_bot_in'    float  bottom depth of section [in]
                             'EI_kip_in2'  float  bending stiffness [kip·in²]
                           Sections must cover the full pile length; the last
                           section acts as a catch-all for any deeper elements.
        Option B — uniform stiffness (legacy / single-section):
          'EI_kip_in2'   : float   uniform bending stiffness [kip·in²]

    z_nodes : ndarray (n_nodes,)  [in]
        Node depths from head (0) to tip (L).  n_elements = n_nodes − 1.

    Returns
    -------
    EI_arr : ndarray (n_elements,)  [kip·in²]
        Per-element bending stiffness; element e spans z_nodes[e]..z_nodes[e+1].
    """
    n_el = len(z_nodes) - 1

    # ── Option B: uniform (no sections defined) — backward-compatible path ──
    if 'EI_sections' not in pile_params or not pile_params['EI_sections']:
        return np.full(n_el, float(pile_params['EI_kip_in2']))

    # ── Option A: variable sections — assign EI by element midpoint depth ───
    sections = pile_params['EI_sections']   # list, sorted by z_top_in ascending

    EI_arr = np.empty(n_el)
    for e in range(n_el):
        # Element midpoint depth governs section assignment
        z_mid = 0.5 * (z_nodes[e] + z_nodes[e + 1])

        # Walk sections; the last section is the fallback for any deeper elements
        EI_e = float(sections[-1]['EI_kip_in2'])
        for sec in sections:
            if sec['z_top_in'] <= z_mid < sec['z_bot_in']:
                EI_e = float(sec['EI_kip_in2'])
                break

        EI_arr[e] = EI_e

    return EI_arr


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — LAYERED SOIL INTERFACE  (+ Georgiadis 1983 correction)
# ══════════════════════════════════════════════════════════════════════════════

# Models eligible for the Georgiadis equivalent-depth correction.
# The correction is defined for sand-to-sand interfaces; clay and rock models
# use different strength frameworks and are excluded.
_GEORGIADIS_ELIGIBLE = frozenset(['reese_sand'])


def _find_layer(z, layers):
    """
    Return the layer dict whose [z_top_in, z_bot_in) bracket contains depth z.
    Falls back to the last layer for z beyond the defined profile.

    layers: list of dicts with keys 'z_top_in', 'z_bot_in', 'model', 'params'
    """
    for layer in layers:
        if layer['z_top_in'] <= z < layer['z_bot_in']:
            return layer
    return layers[-1]   # extend last layer to pile tip


# ── Georgiadis (1983) — Equivalent-depth bisection ───────────────────────────

def _bisect_z_eq(p_target, lower_params, lower_model,
                 z_init, n_iter=200, z_hi_growth=2.0, tol=1.0e-4):
    """
    Find z_eq > 0 such that:
        compute_pult(z_eq, lower_params, lower_model) ≈ p_target
    using bisection on the monotonically increasing p_ult(z) relationship.

    Parameters
    ----------
    p_target     : float  [kip/in]  target p_ult (= p_ult of upper layer at interface)
    lower_params : dict             soil params of the lower layer
    lower_model  : str              p-y model key of the lower layer
    z_init       : float  [in]      starting estimate for z_hi (interface depth)
    n_iter       : int              maximum bisection iterations
    z_hi_growth  : float            bracket-expansion multiplier if p_hi < p_target
    tol          : float  [in]      convergence tolerance on z_eq interval width

    Returns
    -------
    z_eq : float  [in]   equivalent depth in the lower layer;
                         returned as z_init if bracketing fails (safety fallback)
    """
    z_lo = 1.0e-3                                 # just above zero (p_ult = 0 at z = 0)
    z_hi = max(z_init * 1.5, z_init + 120.0)     # initial upper bracket

    # Expand z_hi until p_ult(z_hi) ≥ p_target (at most 60 doublings)
    for _ in range(60):
        if compute_pult(z_hi, lower_params, lower_model) >= p_target:
            break
        z_hi *= z_hi_growth
    else:
        # Could not bracket — return z_init unchanged (conservative fallback)
        warnings.warn(
            f"_bisect_z_eq: could not bracket p_target={p_target:.4f} kip/in "
            f"for model '{lower_model}'. Georgiadis offset set to zero at this interface.",
            RuntimeWarning, stacklevel=3)
        return z_init

    # Standard bisection
    for _ in range(n_iter):
        z_mid = 0.5 * (z_lo + z_hi)
        p_mid = compute_pult(z_mid, lower_params, lower_model)
        if p_mid < p_target:
            z_lo = z_mid
        else:
            z_hi = z_mid
        if (z_hi - z_lo) < tol:
            break

    return 0.5 * (z_lo + z_hi)


def compute_georgiadis_offsets(layers):
    """
    Georgiadis (1983) equivalent-depth layering correction.

    Theory
    ------
    In a layered sand profile, applying each layer's p-y formula starting from the
    real depth z creates a step discontinuity in p_ult at layer interfaces wherever
    the effective unit weight γ changes.  This artificially softens (or stiffens) the
    layer just below the interface.

    Georgiadis (1983) resolves this by computing an *equivalent depth* z_eq for the
    lower layer such that p_ult is continuous across the interface:

        Find z_eq  such that  p_ult(lower_layer, z_eq) = p_ult(upper_layer, z_interface)

    For any real depth z inside the lower layer:

        z_eff = z + Δz      where  Δz = z_eq − z_interface

    This depth offset Δz is stored as '_z_geo_offset' in each layer dict.  Layers
    with no correction (non-eligible model or first layer) receive Δz = 0.

    The correction chains for profiles with more than two layers:
        - Layer 0 → Δz₀ = 0 (reference)
        - Layer 1 → Δz₁ = z_eq₁ − z₁_top
          where z_eq₁ solves: p_ult(L1, z_eq₁) = p_ult(L0, z₁_top + Δz₀)
        - Layer 2 → Δz₂ = z_eq₂ − z₂_top
          where z_eq₂ solves: p_ult(L2, z_eq₂) = p_ult(L1, z₂_top + Δz₁)
        - ... and so on

    Applicability
    -------------
    Sand-to-sand interfaces only ('reese_sand' → 'reese_sand').
    Sand-to-rock and sand-to-clay transitions are excluded; Δz = 0 at those interfaces.

    Parameters
    ----------
    layers : list of layer dicts
        Each dict must have 'z_top_in', 'z_bot_in', 'model', 'params'.

    Returns
    -------
    layers_corrected : list (deep copy)
        Each dict gains '_z_geo_offset' [in].  Layers without correction have 0.0.
    """
    import copy
    layers_corrected = copy.deepcopy(layers)

    # Layer 0: always the reference (no offset)
    layers_corrected[0]['_z_geo_offset'] = 0.0

    for i in range(1, len(layers_corrected)):
        upper = layers_corrected[i - 1]
        lower = layers_corrected[i]
        z_interface = float(lower['z_top_in'])

        # Only correct eligible (sand-to-sand) interfaces
        if (upper['model'] not in _GEORGIADIS_ELIGIBLE or
                lower['model'] not in _GEORGIADIS_ELIGIBLE):
            lower['_z_geo_offset'] = 0.0
            continue

        # Effective depth at the interface in the upper layer (chains correctly)
        z_upper_eff = z_interface + upper.get('_z_geo_offset', 0.0)

        # Target p_ult: upper layer evaluated at its effective depth at the interface
        p_target = compute_pult(z_upper_eff, upper['params'], upper['model'])

        if p_target <= 0.0:
            # No resistance at the interface (z ≈ 0 or degenerate) — skip
            lower['_z_geo_offset'] = 0.0
            continue

        # Find z_eq in the lower layer matching p_target
        z_eq = _bisect_z_eq(p_target, lower['params'], lower['model'],
                             z_init=z_interface)

        lower['_z_geo_offset'] = z_eq - z_interface

    return layers_corrected


def georgiadis_debug_table(layers, z_array_in):
    """
    Build a debug table showing the Georgiadis correction at selected depths.

    Returns a list of dicts, one per depth, with keys:
        'z_ft'          actual depth [ft]
        'z_in'          actual depth [in]
        'z_offset_in'   Georgiadis depth offset Δz [in]
        'z_eff_in'      effective depth used in p-y evaluation [in]
        'z_eff_ft'      effective depth [ft]
        'layer_idx'     0-based layer index
        'layer_model'   p-y model name string

    Parameters
    ----------
    layers      : list of layer dicts (must have '_z_geo_offset' already set)
    z_array_in  : array-like of real depths [in]
    """
    rows = []
    for z in z_array_in:
        lyr = _find_layer(z, layers)
        idx = next((k for k, l in enumerate(layers) if l is lyr), len(layers) - 1)
        z_off = lyr.get('_z_geo_offset', 0.0)
        z_eff = z + z_off
        rows.append({
            'z_ft':        z / 12.0,
            'z_in':        float(z),
            'z_offset_in': z_off,
            'z_eff_in':    z_eff,
            'z_eff_ft':    z_eff / 12.0,
            'layer_idx':   idx,
            'layer_model': lyr['model'],
        })
    return rows


# ── Core layered-soil dispatch (Georgiadis-aware) ────────────────────────────

def soil_response_layered(y, z, layers):
    """
    Evaluate p-y response at depth z using the correct soil layer.

    If the layer carries a '_z_geo_offset' key (set by compute_georgiadis_offsets),
    the Georgiadis equivalent depth is substituted:
        z_eff = z + layer['_z_geo_offset']
    Without the key (offset absent or zero) z_eff = z, recovering original behaviour.

    Returns (p [kip/in], k_sec [kip/in²], k_tan [kip/in²]).
    """
    lyr   = _find_layer(z, layers)
    z_eff = z + lyr.get('_z_geo_offset', 0.0)
    return get_soil_response(y, z_eff, lyr['params'], lyr['model'])


def pult_layered(z, layers):
    """
    Return p_ult [kip/in] at depth z for post-processing.
    Uses the Georgiadis-corrected effective depth if '_z_geo_offset' is set.
    """
    lyr   = _find_layer(z, layers)
    z_eff = z + lyr.get('_z_geo_offset', 0.0)
    return compute_pult(z_eff, lyr['params'], lyr['model'])


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — SECANT STIFFNESS UPDATE (PER-ELEMENT, 2 GAUSS POINTS)
# ══════════════════════════════════════════════════════════════════════════════

def _initial_k_sec_gp(n_elements, z_nodes, layers):
    """
    k_sec at y = 0 for all Gauss points → shape (n_elements, 2).
    Calls soil_response_layered(y=0, z_g) for each Gauss point.
    """
    k_gp = np.zeros((n_elements, 2))
    for e in range(n_elements):
        Le_e = z_nodes[e + 1] - z_nodes[e]
        s_gp, _ = _gauss_points_on_element(Le_e)
        for g in range(2):
            z_g = z_nodes[e] + s_gp[g]
            _, k_s, _ = soil_response_layered(0.0, z_g, layers)
            k_gp[e, g] = k_s
    return k_gp


def _update_k_sec_gp(u, n_elements, z_nodes, layers):
    """
    k_sec at the converging deflection field u → shape (n_elements, 2).
    Interpolates y at each Gauss point via Hermitian cubic field.
    """
    k_gp = np.zeros((n_elements, 2))
    for e in range(n_elements):
        Le_e   = z_nodes[e + 1] - z_nodes[e]
        u_e    = np.array([u[2*e], u[2*e+1], u[2*(e+1)], u[2*(e+1)+1]])
        s_gp, zeta_gp = _gauss_points_on_element(Le_e)
        for g in range(2):
            N   = hermitian_shape_N(zeta_gp[g], Le_e)
            y_g = np.dot(N, u_e)
            z_g = z_nodes[e] + s_gp[g]
            _, k_s, _ = soil_response_layered(y_g, z_g, layers)
            k_gp[e, g] = k_s
    return k_gp


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — GLOBAL ASSEMBLY
# ══════════════════════════════════════════════════════════════════════════════

def assemble_global_stiffness(n_elements, n_dof, EI_array, k_sec_gp, Le_array,
                              P_axial=0.0):
    """
    Assemble K = K_beam + K_soil − K_geo using Gauss-integrated soil matrices.

    Parameters
    ----------
    n_elements  : int
    n_dof       : int
    EI_array    : float or (n_elements,) [kip·in²]
    k_sec_gp    : ndarray (n_elements, 2)  [kip/in²]  secant at 2 GPs per element
    Le_array    : float or (n_elements,) [in]
    P_axial     : float  [kip]  axial compressive load (positive = compressive)
                  When 0.0 (default) the geometric term vanishes and the
                  original purely lateral solver is recovered exactly.
    """
    EI_v = np.broadcast_to(np.atleast_1d(EI_array), (n_elements,))
    Le_v = np.broadcast_to(np.atleast_1d(Le_array), (n_elements,))
    k_gp = np.asarray(k_sec_gp)

    K = np.zeros((n_dof, n_dof))
    for e in range(n_elements):
        K_e = (beam_element_stiffness(EI_v[e], Le_v[e])
               + soil_element_stiffness_gauss(k_gp[e, 0], k_gp[e, 1], Le_v[e])
               - geometric_stiffness_element(P_axial, Le_v[e]))
        dof = [2*e, 2*e+1, 2*(e+1), 2*(e+1)+1]
        for r in range(4):
            for c in range(4):
                K[dof[r], dof[c]] += K_e[r, c]
    return K


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — BOUNDARY CONDITIONS AND LOAD VECTOR
# ══════════════════════════════════════════════════════════════════════════════

def build_fixed_dofs(n_elements, head_fixed, tip_condition):
    """
    Return list of globally constrained DOF indices.

    head_fixed     : bool    True = fixed head (θ(0) = 0 → DOF 1 fixed)
    tip_condition  : str     'free' | 'pinned' | 'fixed'
    """
    fixed = []
    if head_fixed:
        fixed.append(1)    # θ at node 0

    n = n_elements         # last node index
    if tip_condition == 'pinned':
        fixed.append(2 * n)
    elif tip_condition == 'fixed':
        fixed.extend([2 * n, 2 * n + 1])
    return fixed


def build_load_vector(n_dof, H_kip, M_kip_in=0.0):
    """F[0] = H (shear at head) [kip],  F[1] = M (moment at head) [kip·in]."""
    F = np.zeros(n_dof)
    F[0] = H_kip
    F[1] = M_kip_in
    return F


def apply_boundary_conditions(K_global, F, fixed_dofs):
    """Zeroing-and-diagonal method; preserves symmetry."""
    Km = K_global.copy()
    Fm = F.copy()
    for d in fixed_dofs:
        Km[d, :] = 0.0
        Km[:, d] = 0.0
        Km[d, d] = 1.0
        Fm[d] = 0.0
    return Km, Fm


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — NONLINEAR SOLVER (SECANT STIFFNESS ITERATION)
# ══════════════════════════════════════════════════════════════════════════════

def solve_nonlinear(pile_params, soil_layers, H_kip, M_kip_in=0.0,
                    P_axial=0.0,
                    max_iter=150, tol=1.0e-6, relax=1.0, verbose=False):
    """
    Secant stiffness p-y iteration, with optional beam-column P-delta effect.

    Algorithm
    ---------
    1. k_sec_gp ← initial_k_sec(y=0)
    2. For it = 1 .. max_iter:
         K   ← assemble(EI, k_sec_gp, Le, P_axial)   ← includes −K_geo
         u   ← solve(K_bc, F_bc)
         u   ← relax·u + (1−relax)·u_prev
         Δu  ← max|y_new − y_prev|  (lateral DOFs only)
         if Δu < tol: converged
         k_sec_gp ← update_k_sec(u)

    Parameters
    ----------
    pile_params : dict
        'L_in'       float   pile length [in]
        'EI_kip_in2' float   bending stiffness [kip·in²]
        'n_elements' int     FEM mesh density
        'head_fixed' bool    True = fixed head (θ=0)
        'tip_cond'   str     'free' | 'pinned' | 'fixed'

    soil_layers : list of dicts  (see _find_layer docstring)
    H_kip       : float   lateral shear at head [kip]
    M_kip_in    : float   moment at head [kip·in]
    P_axial     : float   axial compressive load at pile head [kip]
                  Positive = compressive (reduces lateral stiffness via P-delta).
                  Default 0.0 → recovers original purely lateral solver exactly.

    Returns
    -------
    dict with keys:
        'converged', 'n_iter', 'conv_hist',
        'u', 'z_nodes', 'Le', 'k_sec_gp', 'P_axial'
    """
    L        = float(pile_params['L_in'])
    n_el     = int(pile_params['n_elements'])
    head_fix = bool(pile_params.get('head_fixed', False))
    tip_cond = str(pile_params.get('tip_cond', 'free'))
    P        = float(P_axial)

    # ── Build mesh first (needed for per-element EI assignment) ─────────────
    n_nodes  = n_el + 1
    n_dof    = 2 * n_nodes
    Le       = L / n_el
    z_nodes  = np.linspace(0.0, L, n_nodes)

    # ── Per-element bending stiffness array ──────────────────────────────────
    # build_EI_array handles both single-section (scalar broadcast) and
    # multi-section (depth-varying) pile definitions.
    EI_arr   = build_EI_array(pile_params, z_nodes)   # shape (n_el,)
    EI_min   = float(np.min(EI_arr))                  # most flexible section

    # ── Warn if P is dangerously close to the free-field Euler critical load ──
    P_cr = estimate_critical_load(EI_min, L, tip_cond)
    if abs(P) > 0.0 and abs(P) >= 0.80 * P_cr:
        warnings.warn(
            f"Axial load P={P:.1f} kip is ≥80% of the free-field Euler critical "
            f"load P_cr≈{P_cr:.1f} kip. Results may be unreliable. "
            f"Consider reducing P or using a longer/stiffer pile.",
            UserWarning, stacklevel=2)

    fixed    = build_fixed_dofs(n_el, head_fix, tip_cond)
    F        = build_load_vector(n_dof, H_kip, M_kip_in)

    k_sec_gp  = _initial_k_sec_gp(n_el, z_nodes, soil_layers)
    u_prev    = np.zeros(n_dof)
    conv_hist = []
    converged = False

    if verbose:
        print(f"  Solver: H={H_kip:.3f} kip  M={M_kip_in:.3f} kip·in  "
              f"P={P:.3f} kip  n_el={n_el}  tol={tol:.1e}")

    for it in range(1, max_iter + 1):
        K      = assemble_global_stiffness(n_el, n_dof, EI_arr, k_sec_gp, Le,
                                           P_axial=P)
        Km, Fm = apply_boundary_conditions(K, F, fixed)
        u_raw  = np.linalg.solve(Km, Fm)
        u_new  = relax * u_raw + (1.0 - relax) * u_prev

        delta = float(np.max(np.abs(u_new[0::2] - u_prev[0::2])))
        conv_hist.append(delta)

        if delta < tol:
            converged = True
            break

        k_sec_gp = _update_k_sec_gp(u_new, n_el, z_nodes, soil_layers)
        u_prev   = u_new.copy()
    else:
        warnings.warn(
            f"Secant iteration did not converge in {max_iter} iterations "
            f"(max|Δy|={conv_hist[-1]:.3e} in > tol={tol:.1e} in).",
            RuntimeWarning, stacklevel=2)

    return {
        'converged':  converged,
        'n_iter':     it,
        'conv_hist':  conv_hist,
        'u':          u_new,
        'z_nodes':    z_nodes,
        'Le':         Le,
        'k_sec_gp':   k_sec_gp,
        'P_axial':    P,
        'EI_arr':     EI_arr,    # per-element EI [kip·in²]; shape (n_el,)
    }


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — INTERNAL FORCE RECOVERY
# ══════════════════════════════════════════════════════════════════════════════

def recover_internal_forces(sol, pile_params, soil_layers, H_kip):
    """
    Recover M(z) and V(z) at all nodes from converged displacement field.

    Moment: M = EI·d²y/dz² via Hermitian B2 operator (averaged at shared nodes).
    Shear:  V[e+1] = V[e] − (Le/2)·Σ_g p(y(ζ_g), z_g)
            (integrates actual p-y curve response at Gauss points)

    Returns  M_nodes [kip·in],  V_nodes [kip]
    """
    u       = sol['u']
    z_nodes = sol['z_nodes']
    Le      = sol['Le']
    n_el    = len(z_nodes) - 1
    n_nodes = n_el + 1

    # ── Per-element EI (from solver output; rebuild if not present) ──────────
    # sol['EI_arr'] is written by solve_nonlinear for all new runs.
    # The fallback via build_EI_array handles any legacy sol dict that predates
    # the variable-EI implementation (e.g., loaded from a cached session).
    EI_arr = sol.get('EI_arr', None)
    if EI_arr is None:
        EI_arr = build_EI_array(pile_params, z_nodes)

    # ── Moment ──────────────────────────────────────────────────────────────
    # B2 operators for a uniform-length element (Le is scalar, uniform mesh).
    # Each element's moment is M = EI_arr[e] · B2 · u_e (Hermitian B² operator).
    B2_L = np.array([-6.0 / Le**2, -4.0 / Le,  6.0 / Le**2, -2.0 / Le])
    B2_R = np.array([ 6.0 / Le**2,  2.0 / Le, -6.0 / Le**2,  4.0 / Le])

    M_L = np.zeros(n_el)
    M_R = np.zeros(n_el)
    for e in range(n_el):
        u_e    = np.array([u[2*e], u[2*e+1], u[2*(e+1)], u[2*(e+1)+1]])
        M_L[e] = EI_arr[e] * np.dot(B2_L, u_e)
        M_R[e] = EI_arr[e] * np.dot(B2_R, u_e)

    M_nodes    = np.zeros(n_nodes)
    M_nodes[0] = M_L[0]
    for e in range(1, n_el):
        M_nodes[e] = 0.5 * (M_R[e - 1] + M_L[e])
    M_nodes[-1] = M_R[-1]

    # ── Shear (integrate actual p(y,z) at Gauss points) ──────────────────
    Jac     = Le / 2.0
    V_nodes = np.zeros(n_nodes)
    V_nodes[0] = H_kip

    for e in range(n_el):
        z_i   = z_nodes[e]
        u_e   = np.array([u[2*e], u[2*e+1], u[2*(e+1)], u[2*(e+1)+1]])
        s_gp, zeta_gp = _gauss_points_on_element(Le)
        p_sum = 0.0
        for g in range(2):
            N   = hermitian_shape_N(zeta_gp[g], Le)
            y_g = np.dot(N, u_e)
            z_g = z_i + s_gp[g]
            p_g, _, _ = soil_response_layered(y_g, z_g, soil_layers)
            p_sum += _W_GP[g] * p_g
        V_nodes[e + 1] = V_nodes[e] - Jac * p_sum

    return M_nodes, V_nodes


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — POST-PROCESSING
# ══════════════════════════════════════════════════════════════════════════════

def extract_results(sol, pile_params, soil_layers, H_kip, M_kip_in,
                    P_axial=0.0, case_name=''):
    """
    Full post-processing: returns all profiles and key values in kip-in.
    Caller converts to display units.

    Parameters
    ----------
    sol         : dict   returned by solve_nonlinear()
    pile_params : dict
    soil_layers : list
    H_kip       : float  [kip]
    M_kip_in    : float  [kip·in]
    P_axial     : float  [kip]   axial load stored verbatim in result dict
    case_name   : str

    Returns
    -------
    dict with all arrays in kip-in and z in inches.
    """
    u       = sol['u']
    z_nodes = sol['z_nodes']   # [in]
    n_nodes = len(z_nodes)

    y_nodes = u[0::2]          # lateral deflection [in]
    t_nodes = u[1::2]          # rotation           [rad]

    M_nodes, V_nodes = recover_internal_forces(sol, pile_params, soil_layers, H_kip)

    # Nodal p and p_ult
    p_nodes    = np.zeros(n_nodes)
    pult_nodes = np.zeros(n_nodes)
    for n in range(n_nodes):
        p_nodes[n], _, _ = soil_response_layered(y_nodes[n], z_nodes[n], soil_layers)
        pult_nodes[n]    = pult_layered(z_nodes[n], soil_layers)

    # Mobilisation ratio p/p_ult
    # Use explicit masked division to avoid RuntimeWarning from np.where evaluating
    # p/pult at all nodes before applying the mask (sand surface node has pult=0).
    ppult = np.zeros_like(p_nodes)
    _mask = pult_nodes > 0.0
    ppult[_mask] = p_nodes[_mask] / pult_nodes[_mask]

    # Key scalar values
    imax   = np.argmax(np.abs(M_nodes))
    M_max  = float(np.max(np.abs(M_nodes)))
    z_Mmax = float(z_nodes[imax])

    # Free-field Euler critical load (for stability reporting).
    # Use the minimum EI across all elements (most flexible section) — conservative.
    EI_arr   = sol.get('EI_arr', build_EI_array(pile_params, z_nodes))
    EI_min   = float(np.min(EI_arr))
    tip_cond = str(pile_params.get('tip_cond', 'free'))
    P_cr     = estimate_critical_load(EI_min, float(pile_params['L_in']), tip_cond)

    return {
        'name':        case_name,
        'converged':   sol['converged'],
        'n_iter':      sol['n_iter'],
        'conv_hist':   sol['conv_hist'],
        # profiles (kip-in)
        'z_in':        z_nodes,
        'y_in':        y_nodes,
        'theta_rad':   t_nodes,
        'M_kip_in':    M_nodes,
        'V_kip':       V_nodes,
        'p_kip_in':    p_nodes,
        'pult_kip_in': pult_nodes,
        'p_over_pult': ppult,
        # scalars
        'y_head_in':       float(y_nodes[0]),
        'y_tip_in':        float(y_nodes[-1]),
        'M_head_kip_in':   float(M_nodes[0]),
        'M_max_kip_in':    M_max,
        'z_Mmax_in':       z_Mmax,
        'V_head_kip':      float(V_nodes[0]),
        'V_tip_kip':       float(V_nodes[-1]),
        'H_applied_kip':   H_kip,
        'M_applied_kip_in': M_kip_in,
        'P_axial_kip':     float(P_axial),
        'P_cr_kip':        float(P_cr),
    }


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — TOP-LEVEL ANALYSIS RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def run_analysis(pile_params, soil_layers, load_cases,
                 max_iter=150, tol=1.0e-6, relax=1.0):
    """
    Run the full nonlinear p-y analysis for a list of load cases.

    Parameters
    ----------
    pile_params  : dict  (see solve_nonlinear)
    soil_layers  : list of layer dicts  (kip-in params)
    load_cases   : list of dicts
        Required keys : 'H_kip'
        Optional keys : 'M_kip_in'  (default 0.0)
                        'P_kip'     (default 0.0, compressive axial load)
                        'name'      (default '')
    max_iter     : int
    tol          : float  [in]   convergence criterion on max|Δy|
    relax        : float  [0,1]  under-relaxation factor (1 = no relaxation)

    Returns
    -------
    list of result dicts (one per load case), each from extract_results()
    """
    results = []
    for lc in load_cases:
        H   = lc['H_kip']
        M   = lc.get('M_kip_in', 0.0)
        P   = lc.get('P_kip',    0.0)
        nm  = lc.get('name',     '')

        sol = solve_nonlinear(
            pile_params, soil_layers,
            H, M, P_axial=P,
            max_iter=max_iter, tol=tol, relax=relax)
        res = extract_results(
            sol, pile_params, soil_layers,
            H, M, P_axial=P, case_name=nm)
        results.append(res)
    return results
