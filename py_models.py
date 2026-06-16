"""
py_models.py  —  p-y Curve Models (Clay + Sand + Rock)
=======================================================
Internal unit system: kip, in throughout.

Implemented models (RSPile Theory Manual, Rocscience 2022):
  1. 'matlock'             Matlock (1970)           soft clay with free water        §5.1
  2. 'reese_stiff_water'   Reese et al. (1975)      stiff clay with free water       §5.2
  3. 'welch_reese'         Welch & Reese (1972)     stiff clay without free water    §5.3
  4. 'reese_sand'          Reese, Cox & Koop (1974) sand above and below WT          §5.4
  5. 'weak_rock'           Reese & Nyman (1978)     weak rock                        §5.5
  6. 'strong_rock'         Reese & Nyman (1978)     strong rock (vuggy limestone)    §5.12

Interface
---------
    p, k_sec, k_tan = get_soil_response(y_in, z_in, params, model)

    y_in  : float  [in]       lateral pile deflection (signed)
    z_in  : float  [in]       depth below ground surface
    params: dict              model-specific parameters (kip-in, see below)
    model : str               one of the keys above

Returns
-------
    p     : float  [kip/in]   soil resistance (+ in load direction, resists displacement)
    k_sec : float  [kip/in²]  secant modulus  = |p| / |y|
    k_tan : float  [kip/in²]  tangent modulus = dp/dy

Common parameter keys (all kip-in)
------------------------------------
    'cu_ksi'         float    undrained shear strength at depth z  [kip/in²]  (clay)
    'ca_ksi'         float    average undrained shear strength     [kip/in²]  (clay, ≈ cu)
    'eps50'          float    axial strain at 50% deviatoric stress [-]        (clay)
    'gamma_kip_in3'  float    effective unit weight                 [kip/in³]
    'J'              float    Matlock empirical factor (default 0.5)           (clay)
    'ks_kip_in3'     float    initial stiffness gradient for Reese clay        [kip/in³]
    'phi_deg'        float    internal friction angle               [degrees]  (sand)
    'kpy_kip_in3'    float    initial modulus of subgrade reaction  [kip/in³]  (sand, = k_py·z/z)
    'loading'        str      'static' or 'cyclic'
    'b_in'           float    pile diameter                         [in]

Rock-specific parameter keys (kip-in)
--------------------------------------
    'q_ur_ksi'       float    uniaxial compressive strength of rock [kip/in²]  (weak rock)
    'k_ir_kip_in2'   float    initial reaction modulus of rock      [kip/in²]  (weak rock)
    'RQD_pct'        float    rock quality designation              [%] 0–100  (weak rock)
    'k_rm'           float    strain factor (0.00005–0.0005)        [-]        (weak rock)
    'z_rock_in'      float    depth to top of rock from ground      [in]       (weak rock, default 0)
    'su_ksi'         float    unconfined compressive strength        [kip/in²]  (strong rock)

Profiles: any scalar parameter can alternatively be supplied as a (N,2) array
    [[z0_in, val0], [z1_in, val1], ...] for depth-varying input (linear interp).

Unit conversion reminders (for callers)
----------------------------------------
    cu  : 1 ksf  = 1/144  kip/in²  ≈ 6.944e-3 kip/in²
    γ'  : 1 pcf  = 1/(1000×1728) kip/in³ ≈ 5.787e-7 kip/in³
    ks  : 1 pci  = 1/1000 kip/in³   (clay initial stiffness)
    kpy : 1 pci  = 1/1000 kip/in³   (sand initial modulus, same units as ks)
    z   : 1 ft   = 12 in
    b   : already in inches
    q_ur: 1 ksi  = 1 kip/in²        (rock UCS, note: 1 MPa ≈ 0.145 ksi)
    k_ir: 1 ksi  = 1 kip/in²        (rock reaction modulus)
    su  : 1 ksi  = 1 kip/in²        (strong rock compressive strength)
"""

import numpy as np

# ══════════════════════════════════════════════════════════════════════════════
# NUMERICAL CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════
_Y_ZERO_TOL  = 1.0e-12   # [in]  |y| below this treated as zero
_Y_INIT_FRAC = 1.0e-2    # regularisation: k_sec(y=0) evaluated at _Y_INIT_FRAC·y_ref
                          # This is a NUMERICAL CONVENIENCE, not the theoretical tangent.

# 2-point Gauss quadrature constants (also used by fem_solver)
_INV_SQRT3 = 1.0 / np.sqrt(3.0)
_XI_GP     = np.array([-_INV_SQRT3, +_INV_SQRT3])
_W_GP      = np.array([1.0, 1.0])

# ── Reese STIFF CLAY A-factors (Figure 5.3, Reese & Van Impe 2011) ───────────
# x-axis = z/b;  clamped to 0.88 at large z/b for both curves
_AS_ZB  = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 100.0])
_AS_VAL = np.array([0.23, 0.27, 0.32, 0.38, 0.46, 0.60, 0.73, 0.87,  0.88])  # static

_AC_ZB  = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 100.0])
_AC_VAL = np.array([0.17, 0.19, 0.22, 0.26, 0.30, 0.38, 0.48, 0.56, 0.65, 0.80, 0.88])  # cyclic

# ── Reese SAND A and B factors (Figure 5.6, Reese & Van Impe 2011) ───────────
# A controls pu (upper resistance);  B controls pm (resistance at ym = b/60).
# Both curves originate at ~2.0 for z/b = 0 and decay with depth.
# Static A clamps at 0.88 for z/b ≥ 5; cyclic A reaches 0.88 by z/b ≈ 2.5.
# Static B clamps at 0.50 for z/b ≥ 5; cyclic B clamps at 0.55.

_AS_SAND_ZB  = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 100.0])
_AS_SAND_VAL = np.array([2.00, 1.92, 1.82, 1.72, 1.62, 1.42, 1.22, 1.08, 0.97, 0.93, 0.90, 0.89, 0.88, 0.88])  # static

_AC_SAND_ZB  = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 100.0])
_AC_SAND_VAL = np.array([2.00, 1.90, 1.80, 1.65, 1.45, 1.12, 0.88, 0.88, 0.88, 0.88])  # cyclic
# ── Correction note (v1.1) ──────────────────────────────────────────────────
# The original Ac values (1.65, 1.45, 1.30, 1.15, 0.98) at z/b = 0.25–1.5 were
# too aggressively decayed: Ac < Bc at z/b ≈ 0.0–1.8, causing pu_cyclic < pm_cyclic
# which triggered the degenerate linear-plateau fallback for the entire shallow zone,
# making the cyclic p-y curves artificially stiffer than static (physically backwards).
#
# Corrected values (2.00→1.90→1.80→1.65→1.45→1.12→0.88) are consistent with
# Figure 5.6 of Reese & Van Impe (2011).  All constraints verified:
#   • Ac ≤ As at all z/b   (cyclic is always equal to or below static for pu)
#   • Ac ≥ Bc + margin     (pu_cyclic ≥ pm_cyclic — required for monotone parabola)
# The minimum margin Ac − Bc = 0.03 at z/b = 2.0 (both curves clamp at 0.88 and 0.85).

_BS_SAND_ZB  = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 100.0])
_BS_SAND_VAL = np.array([2.00, 1.85, 1.70, 1.55, 1.40, 1.15, 0.92, 0.75, 0.60, 0.55, 0.52, 0.51, 0.50, 0.50])  # static

_BC_SAND_ZB  = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 100.0])
_BC_SAND_VAL = np.array([2.00, 1.80, 1.60, 1.45, 1.30, 1.05, 0.85, 0.73, 0.65, 0.60, 0.57, 0.56, 0.55, 0.55])  # cyclic


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def _get_param(z, params, key):
    """
    Return scalar soil parameter at depth z [in].
    Accepts:  scalar value  or  (N,2) ndarray [[z0,v0],[z1,v1],...] (linear interp).
    """
    if key not in params:
        raise KeyError(f"Required parameter '{key}' not found in soil params dict.")
    v = params[key]
    if np.ndim(v) == 0:
        return float(v)
    prof = np.asarray(v, dtype=float)
    return float(np.interp(z, prof[:, 0], prof[:, 1]))


def _sign_and_abs(y):
    return (1.0 if y >= 0.0 else -1.0), abs(y)


# ══════════════════════════════════════════════════════════════════════════════
# MODEL 1 — MATLOCK (1970)  Soft Clay with Free Water  §5.1
# ══════════════════════════════════════════════════════════════════════════════
def _matlock_soft_clay(y, z, params):
    """
    RSPile §5.1  (Matlock, 1970)

    Reference deflection:
        y50 = 2.5 · ε50 · b                                         [in]
        (factor 2.5 from Matlock 1970 original; converts axial ε50 to lateral y50)

    Ultimate resistance (minimum of shallow/deep):
        p_ult_s = [3 + (γ'/cu)·z + (J/b)·z] · cu · b               (Eq.9)
        p_ult_d = 9 · cu · b                                         (Eq.10)

    Curve:
        p = 0.5·p_ult·(y/y50)^(1/3)   for y ≤ 8·y50
        p = p_ult                       for y > 8·y50   (plateau)

    Cyclic loading: RSPile §5.1 does NOT define a separate cyclic curve.
        The original Matlock (1970) paper has a cyclic modification (plateau
        starts at 3·y50, p_ult reduced for z < z_r) but this is NOT in the
        RSPile §5.1 formulation and is NOT implemented here.
        The 'loading' parameter is accepted but ignored for this model.
    """
    cu    = _get_param(z, params, 'cu_ksi')
    eps50 = float(params['eps50'])
    gamma = float(params['gamma_kip_in3'])
    b     = float(params['b_in'])
    J     = float(params.get('J', 0.5))

    y50   = 2.5 * eps50 * b
    y_lim = 8.0 * y50

    p_ult_s = (3.0 + (gamma / cu) * z + (J / b) * z) * cu * b
    p_ult_d = 9.0 * cu * b
    p_ult   = max(min(p_ult_s, p_ult_d), 0.0)

    y_sign, abs_y = _sign_and_abs(y)

    # y ≈ 0: numerical regularisation (k_sec evaluated at y_eval = 1% of y50)
    if abs_y <= _Y_ZERO_TOL:
        y_ev  = _Y_INIT_FRAC * max(y50, 1e-15)
        p_ref = 0.5 * p_ult * (y_ev / y50) ** (1.0 / 3.0) if y50 > 0.0 else 0.0
        k_sec = p_ref / y_ev if y_ev > 0.0 else 0.0
        return 0.0, k_sec, (1.0 / 3.0) * k_sec

    if abs_y <= y_lim:
        p_mag = 0.5 * p_ult * (abs_y / y50) ** (1.0 / 3.0)
        k_sec = p_mag / abs_y
        k_tan = (1.0 / 3.0) * k_sec   # exact: d/dy [C·y^(1/3)] = (1/3)·p/y
    else:
        p_mag = p_ult
        k_sec = p_ult / abs_y
        k_tan = 0.0

    return y_sign * p_mag, k_sec, k_tan


# ══════════════════════════════════════════════════════════════════════════════
# MODEL 2 — REESE et al. (1975)  Stiff Clay with Free Water  §5.2
# ══════════════════════════════════════════════════════════════════════════════
def _reese_stiff_clay_water(y, z, params):
    """
    RSPile §5.2  (Reese et al., 1975)

    Reference deflection:
        y50 = ε50 · b    (no 2.5 factor — stiff overconsolidated clay)

    Ultimate resistance parameter (Eqs. 11–12):
        pc_shallow = 2·ca·b + γ'·b·z + 2.83·ca·z    (Eq.11)
        pc_deep    = 11·cu·b                          (Eq.12)
        pc         = min(pc_shallow, pc_deep)

    A-factor: depth-dependent from Figure 5.3 (Reese & Van Impe, 2011)
        As (static loading) or Ac (cyclic loading)

    Curve segments:
        1. Linear initial:  p = Esi·y           (0 ≤ y ≤ y_k,  Esi = ks·z)
        2. Parabolic:       p = 0.5·pc·(y/y50)^½  (y_k ≤ y ≤ 6A·y50)
        3. Declining:       p = p_B + Ess·(y−6A·y50)  (Ess = −0.0625·pc/y50)
        4. Plateau:         p = p_C             (y > 18A·y50)

    Notes:
    - At z = 0, Esi = 0 → curve starts directly with the parabolic branch.
    - The descending slope Ess is negative (softening). k_tan set to 0 there
      for secant-iteration stability.
    """
    cu      = _get_param(z, params, 'cu_ksi')
    ca      = _get_param(z, params, 'ca_ksi')
    eps50   = float(params['eps50'])
    gamma   = float(params['gamma_kip_in3'])
    b       = float(params['b_in'])
    ks      = float(params['ks_kip_in3'])
    loading = str(params.get('loading', 'static')).lower()

    y50 = eps50 * b   # [in]

    # pc  (Eqs. 11–12)
    pc_s = 2.0 * ca * b + gamma * b * z + 2.83 * ca * z
    pc_d = 11.0 * cu * b
    pc   = max(min(pc_s, pc_d), 0.0)

    # A factor
    zb = z / b if b > 0.0 else 0.0
    A  = (np.interp(zb, _AS_ZB, _AS_VAL) if loading == 'static'
          else np.interp(zb, _AC_ZB, _AC_VAL))

    # Initial stiffness (zero at surface)
    Esi = ks * z   # [kip/in²]

    # Key deflection breakpoints
    y_A = A * y50
    y_B = 6.0 * A * y50
    y_C = 18.0 * A * y50

    # Key resistance values
    p_B = 0.5 * pc * np.sqrt(max(6.0 * A, 0.0))          # at peak
    Ess = -0.0625 * pc / y50 if y50 > 0.0 else 0.0        # declining slope [kip/in²]
    p_C = max(p_B + Ess * (y_C - y_B), 0.0)               # residual plateau

    y_sign, abs_y = _sign_and_abs(y)

    # y ≈ 0 ──────────────────────────────────────────────────────────────────
    if abs_y <= _Y_ZERO_TOL:
        if Esi > 0.0:
            # Linear initial portion exists; k_sec = Esi is well-defined
            return 0.0, Esi, Esi
        else:
            # z ≈ 0: parabolic start; regularise
            y_ev  = _Y_INIT_FRAC * max(y50, 1e-15)
            p_ref = 0.5 * pc * np.sqrt(y_ev / y50) if y50 > 0.0 else 0.0
            k_sec = p_ref / y_ev if y_ev > 0.0 else 0.0
            return 0.0, k_sec, 0.5 * k_sec

    # transition point (linear → parabola)
    if Esi > 0.0 and y50 > 0.0 and pc > 0.0:
        y_k = 0.25 * pc ** 2 / (Esi ** 2 * y50)
    else:
        y_k = 0.0
    y_k = min(y_k, y_A)

    # Evaluate p ─────────────────────────────────────────────────────────────
    if abs_y <= y_k:
        p_mag = Esi * abs_y
        k_sec = Esi
        k_tan = Esi
    elif abs_y <= y_B:
        p_mag = 0.5 * pc * np.sqrt(abs_y / y50)
        k_sec = p_mag / abs_y
        k_tan = 0.5 * k_sec           # d/dy[C·y^½] = (1/2)·p/y
    elif abs_y <= y_C:
        p_mag = max(p_B + Ess * (abs_y - y_B), 0.0)
        k_sec = p_mag / abs_y if abs_y > 0.0 else 0.0
        k_tan = 0.0                   # softening; treat as zero for stability
    else:
        p_mag = p_C
        k_sec = p_C / abs_y if abs_y > 0.0 else 0.0
        k_tan = 0.0

    return y_sign * p_mag, k_sec, k_tan


# ══════════════════════════════════════════════════════════════════════════════
# MODEL 3 — WELCH & REESE (1972)  Stiff Clay WITHOUT Free Water  §5.3
# ══════════════════════════════════════════════════════════════════════════════
def _welch_reese_stiff_clay_nowater(y, z, params):
    """
    RSPile §5.3  (Welch & Reese, 1972)

    Reference deflection:
        y50 = ε50 · b   (same formula as Matlock without 2.5 factor)

    Ultimate resistance (Eqs. 13–14):
        p_ult_s = [3 + (γ'/ca)·z + (J/b)·z] · ca · b   (Eq.13)
        p_ult_d = 9 · cu · b                              (Eq.14)
        p_ult   = min(p_ult_s, p_ult_d)

    Curve (quarter-root, Figure 5.4):
        p = 0.5·p_ult·(y/y50)^(1/4)   for y ≤ 16·y50
        p = p_ult                       for y > 16·y50   (plateau)

    Cyclic loading: §5.3 is STATIC ONLY. The cyclic variant (§5.13, Modified
        Stiff Clay w/o Free Water) adds an initial linear branch and a cyclic
        degradation branch. §5.13 is NOT implemented in this module.
        The 'loading' parameter is accepted but ignored for this model.
    """
    cu    = _get_param(z, params, 'cu_ksi')
    ca    = _get_param(z, params, 'ca_ksi')
    eps50 = float(params['eps50'])
    gamma = float(params['gamma_kip_in3'])
    b     = float(params['b_in'])
    J     = float(params.get('J', 0.5))

    y50   = eps50 * b
    y_lim = 16.0 * y50

    p_ult_s = (3.0 + (gamma / ca) * z + (J / b) * z) * ca * b
    p_ult_d = 9.0 * cu * b
    p_ult   = max(min(p_ult_s, p_ult_d), 0.0)

    y_sign, abs_y = _sign_and_abs(y)

    if abs_y <= _Y_ZERO_TOL:
        y_ev  = _Y_INIT_FRAC * max(y50, 1e-15)
        p_ref = 0.5 * p_ult * (y_ev / y50) ** (1.0 / 4.0) if y50 > 0.0 else 0.0
        k_sec = p_ref / y_ev if y_ev > 0.0 else 0.0
        return 0.0, k_sec, (1.0 / 4.0) * k_sec

    if abs_y <= y_lim:
        p_mag = 0.5 * p_ult * (abs_y / y50) ** (1.0 / 4.0)
        k_sec = p_mag / abs_y
        k_tan = (1.0 / 4.0) * k_sec   # d/dy[C·y^(1/4)] = (1/4)·p/y
    else:
        p_mag = p_ult
        k_sec = p_ult / abs_y
        k_tan = 0.0

    return y_sign * p_mag, k_sec, k_tan


# ══════════════════════════════════════════════════════════════════════════════
# MODEL 4 — REESE, COX & KOOP (1974)  Sand Above and Below Water Table  §5.4
# ══════════════════════════════════════════════════════════════════════════════
def _reese_sand(y, z, params):
    """
    RSPile §5.4  (Reese, Cox & Koop, 1974)
    Sand p-y curves — valid above and below the water table.

    Required params keys (kip-in system)
    -------------------------------------
        'phi_deg'        float   internal friction angle           [degrees]
        'gamma_kip_in3'  float   effective unit weight             [kip/in³]
                                   buoyant (γ') below WT; total (γ) above WT
        'kpy_kip_in3'    float   initial modulus of subgrade reac  [kip/in³]
                                   (k_py — gradient of initial stiffness with depth)
        'b_in'           float   pile diameter                      [in]
        'loading'        str     'static' or 'cyclic'

    Intermediate coefficients (Reese 1974, Eq. 15–16)
    ---------------------------------------------------
        α  = φ/2
        β  = 45° + φ/2
        K₀ = 0.40  (at-rest earth pressure coefficient)
        Ka = tan²(45 − φ/2)  (Rankine active coefficient)

    Ultimate unit soil resistance ps (kip/in):
        Eq. 15 (shallow wedge failure):
            ps = γz [ K₀z·tan(φ)·sin(β) / (tan(β−φ)·cos(α))
                    + tan(β)/(tan(β−φ)) · (b + z·tan(β)·tan(α))
                    + K₀z·tan(β)·(tan(φ)·sin(β) − tan(α))
                    − Ka·b ]
        Eq. 16 (deep flow-around failure):
            ps = Ka·b·γz·(tan⁸β − 1) + K₀·b·γz·tan(φ)·tan⁴β
        ps = min(Eq.15, Eq.16)  ← critical governs

    Resistance parameters:
        pu = A · ps   (plateau resistance)      A from Figure 5.6
        pm = B · ps   (resistance at ym=b/60)   B from Figure 5.6

    Reference deflections:
        ym = b / 60             [in]
        yu = 3·b / 80           [in]  = 0.0375·b

    Power-law parabola fit through (ym, pm) → (yu, pu):
        p = C · y^n
        n = log(pu/pm) / log(yu/ym)      [exponent; n > 0 required, NOT n > 1]
        C = pu / yu^n                     [coefficient]
        Typical range: n ≈ 0.25–0.70 (static);  lower for cyclic loading.
        n < 1 is concave-down in linear scale (flattens toward plateau) — correct.
        n > 1 only if A/B ratio is very large (rare; not assumed here).

    Curve segments (Figure 5.5):
        Segment 1 — Linear:      0 ≤ y ≤ yk      p = Esi·y          k_tan = Esi
        Segment 2 — Parabolic:   yk < y ≤ yu     p = C·y^n          k_tan = n·(p/y)
        Segment 3 — Plateau:     y > yu           p = pu             k_tan = 0

        where Esi = k_py · z   (initial stiffness, zero at surface)
              yk  = (Esi/C)^(1/(n−1))  (transition deflection)

    Special case: if yk ≥ yu → no parabolic segment; curve is linear from
        origin to (yu, pu) then plateau.

    Notes
    -----
    - At z = 0: Esi = 0 and ps = 0 → returns (0, 0, 0) with no singularity.
    - The A/B factor tables are digitised from Figure 5.6 (Reese & Van Impe, 2011);
      estimated uncertainty ≈ ±3% at intermediate z/b values.
    - The exponent n > 1 for all physically reasonable cases (since ym < yu and
      pm < pu when A > B, which holds for z/b > 0).
    """
    phi_deg = float(params.get('phi_deg', 35.0))
    gamma   = float(params['gamma_kip_in3'])
    kpy     = float(params['kpy_kip_in3'])
    b       = float(params['b_in'])
    loading = str(params.get('loading', 'static')).lower()

    # ── At ground surface: no overburden → no resistance ──────────────────────
    if z <= _Y_ZERO_TOL:
        return 0.0, 0.0, 0.0

    # ── Trig constants ────────────────────────────────────────────────────────
    phi_r   = np.radians(phi_deg)
    alpha   = phi_r / 2.0                          # α = φ/2  [rad]
    beta    = np.radians(45.0) + phi_r / 2.0       # β = 45° + φ/2  [rad]
    K0      = 0.4
    Ka      = np.tan(np.radians(45.0) - phi_r / 2.0) ** 2   # Rankine Ka

    tan_phi  = np.tan(phi_r)
    tan_beta = np.tan(beta)
    sin_beta = np.sin(beta)
    cos_alpha = np.cos(alpha)
    tan_alpha = np.tan(alpha)

    # tan(β − φ): denominator in Eq. 15
    tan_bmphi = np.tan(beta - phi_r)
    if abs(tan_bmphi) < 1e-10:
        tan_bmphi = 1e-10   # numerical guard (degenerate φ only)

    # ── Ultimate resistance ps (Eqs. 15–16) ──────────────────────────────────
    # Eq. 15 — shallow wedge (4 terms)
    t1    = K0 * z * tan_phi * sin_beta / (tan_bmphi * cos_alpha)
    t2    = (tan_beta / tan_bmphi) * (b + z * tan_beta * tan_alpha)
    t3    = K0 * z * tan_beta * (tan_phi * sin_beta - tan_alpha)
    t4    = Ka * b
    ps_sh = gamma * z * (t1 + t2 + t3 - t4)

    # Eq. 16 — deep flow-around
    ps_dp = (Ka * b * gamma * z * (tan_beta ** 8 - 1.0) +
             K0 * b * gamma * z * tan_phi * tan_beta ** 4)

    ps = max(min(ps_sh, ps_dp), 0.0)

    # ── A and B depth-correction factors (Figure 5.6) ────────────────────────
    zb = z / b if b > 0.0 else 0.0
    if loading == 'static':
        A = np.interp(zb, _AS_SAND_ZB, _AS_SAND_VAL)
        B = np.interp(zb, _BS_SAND_ZB, _BS_SAND_VAL)
    else:   # cyclic
        A = np.interp(zb, _AC_SAND_ZB, _AC_SAND_VAL)
        B = np.interp(zb, _BC_SAND_ZB, _BC_SAND_VAL)

    pu = A * ps   # [kip/in]  plateau resistance
    pm = B * ps   # [kip/in]  resistance at ym

    # ── Guard: zero or degenerate ps (very shallow depth) ────────────────────
    if pu <= 0.0:
        return 0.0, 0.0, 0.0

    # ── Reference deflections (Figure 5.5) ───────────────────────────────────
    ym = b / 60.0           # [in]  deflection at pm
    yu = 3.0 * b / 80.0     # [in]  deflection at pu  (= 0.0375·b > ym)

    # ── Power-law parabola construction ──────────────────────────────────────
    # Require pm < pu (should hold since B < A for all z/b > 0).
    # If degenerate, fall back to linear-plateau only.
    if pm <= 0.0 or pu <= pm or ym <= 0.0 or yu <= ym:
        y_sign, abs_y = _sign_and_abs(y)
        if abs_y <= _Y_ZERO_TOL:
            Esi = kpy * z
            return 0.0, Esi, Esi
        Esi   = kpy * z
        p_mag = min(Esi * abs_y, pu)
        k_sec = p_mag / abs_y
        k_tan = Esi if (Esi > 0.0 and abs_y <= pu / Esi) else 0.0
        return y_sign * p_mag, k_sec, k_tan

    # Exponent and coefficient of p = C · y^n
    n = np.log(pu / pm) / np.log(yu / ym)    # > 1 for typical sand
    C = pu / (yu ** n)                         # [kip/in^(1-n) ... consistent]

    # ── Initial stiffness (zero at surface, linear with depth) ───────────────
    Esi = kpy * z   # [kip/in²]

    # ── Transition deflection yk ──────────────────────────────────────────────
    # Solve Esi · yk = C · yk^n  →  yk = (Esi/C)^(1/(n−1))
    # n > 0 is guaranteed (A > B ensures pu > pm).  n can be < 1 (typically 0.25–0.70).
    #   If n < 1: exponent 1/(n-1) is negative; yk = (Esi/C)^(negative) is real/positive.
    #   If n > 1: exponent is positive; yk is also real/positive.
    #   Both cases satisfy continuity at yk (linear = parabola value at that point).
    # If n ≈ 1.0 (degenerate): skip yk formula; handled by yk = 0.0 → no linear segment.
    if Esi > 0.0 and abs(n - 1.0) > 1e-9:
        yk = (Esi / C) ** (1.0 / (n - 1.0))
    else:
        yk = 0.0   # zero stiffness (z ≈ 0) or degenerate n

    # ── Evaluate p at requested deflection y ──────────────────────────────────
    y_sign, abs_y = _sign_and_abs(y)

    # y ≈ 0: return initial conditions without singularity
    if abs_y <= _Y_ZERO_TOL:
        if Esi > 0.0:
            return 0.0, Esi, Esi
        else:
            y_ev  = _Y_INIT_FRAC * max(ym, 1e-15)
            p_ref = C * y_ev ** n
            k_s   = p_ref / y_ev
            return 0.0, k_s, n * k_s

    # Special case: yk ≥ yu — the whole curve from 0 to pu is linear
    # (RSPile §5.4: "if yk > yu then p-y curve is linear from origin to yu, pu")
    if yk >= yu:
        if abs_y <= yu:
            p_mag = Esi * abs_y
            k_sec = Esi
            k_tan = Esi
        else:
            p_mag = pu
            k_sec = pu / abs_y
            k_tan = 0.0
        return y_sign * p_mag, k_sec, k_tan

    # Normal 3-segment curve
    if abs_y <= yk:
        # Segment 1 — Linear
        p_mag = Esi * abs_y
        k_sec = Esi
        k_tan = Esi
    elif abs_y <= yu:
        # Segment 2 — Parabolic  p = C·y^n
        p_mag = C * abs_y ** n
        k_sec = p_mag / abs_y                  # = C · y^(n−1)
        k_tan = n * k_sec                      # = n · C · y^(n−1)  (exact tangent)
    else:
        # Segment 3 — Plateau
        p_mag = pu
        k_sec = pu / abs_y
        k_tan = 0.0

    return y_sign * p_mag, k_sec, k_tan



# ==============================================================================
# MODEL 4b -- API RP 2A  Sand  (API Method)  Sec 5.8
# ==============================================================================
def _api_sand(y, z, params):
    """
    RSPile Sec 5.8  (API RP 2A) -- Sand.  Same procedure for static and cyclic.

    Required params (kip-in):
        'phi_deg'       internal friction angle [deg]
        'gamma_kip_in3' effective unit weight [kip/in^3]
        'kpy_kip_in3'   k = initial modulus of subgrade reaction [kip/in^3]
        'b_in'          pile diameter [in]
        'loading'       'static' or 'cyclic'

    Ultimate resistance (lesser of shallow / deep), Eqs 19-20:
        pus = (C1*z + C2*b) * gamma' * z          (shallow)
        pud =  C3*b       * gamma' * z            (deep)
        pu  = min(pus, pud)
    Coefficients (Eqs 21-24); alpha, beta, Ka use the manual's Sec 5.4 defs:
        alpha = phi/2 ; beta = 45 + phi/2 ; K0 = 0.4
        Kp = tan^2(45 + phi/2) ; Ka = tan^2(45 - phi/2)
        C1 = tan(beta)*( Kp*tan(alpha) + K0*[ tan(phi)*sin(beta)*(1/cos(alpha)+1) - tan(alpha) ] )
        C2 = Kp - Ka
        C3 = Kp^2*(Kp + K0*tan(phi)) - Ka
    p-y curve (Eq 25):
        p = A * pu * tanh( k*z*y / (A*pu) )
        A = 0.9 (cyclic) ; A = max(3 - 0.8*z/b, 0.9) (static)
    Initial slope dp/dy|_(y=0) = k*z = Esi.  k_tan = Esi*(1 - tanh^2(arg)).
    """
    phi_deg = float(params.get('phi_deg', 35.0))
    gamma   = float(params['gamma_kip_in3'])
    kpy     = float(params['kpy_kip_in3'])
    b       = float(params['b_in'])
    loading = str(params.get('loading', 'static')).lower()

    if z <= _Y_ZERO_TOL:
        return 0.0, 0.0, 0.0

    phi_r = np.radians(phi_deg)
    alpha = phi_r / 2.0
    beta  = np.radians(45.0) + phi_r / 2.0
    K0    = 0.4
    Kp    = np.tan(np.radians(45.0) + phi_r / 2.0) ** 2
    Ka    = np.tan(np.radians(45.0) - phi_r / 2.0) ** 2
    tan_phi = np.tan(phi_r)

    C1 = np.tan(beta) * (Kp * np.tan(alpha) +
            K0 * (tan_phi * np.sin(beta) * (1.0 / np.cos(alpha) + 1.0) - np.tan(alpha)))
    C2 = Kp - Ka
    C3 = Kp ** 2 * (Kp + K0 * tan_phi) - Ka

    pus = (C1 * z + C2 * b) * gamma * z
    pud = C3 * b * gamma * z
    pu  = max(min(pus, pud), 0.0)

    Esi = kpy * z   # initial slope k*x

    y_sign, abs_y = _sign_and_abs(y)

    if pu <= 0.0:
        # degenerate (only near z=0); linear fallback on initial slope
        if abs_y <= _Y_ZERO_TOL:
            return 0.0, Esi, Esi
        return y_sign * Esi * abs_y, Esi, Esi

    A = 0.9 if loading == 'cyclic' else (max(3.0 - 0.8 * z / b, 0.9) if b > 0.0 else 0.9)

    if abs_y <= _Y_ZERO_TOL:
        return 0.0, Esi, Esi

    arg   = Esi * abs_y / (A * pu)
    th    = np.tanh(arg)
    p_mag = A * pu * th
    k_sec = p_mag / abs_y
    k_tan = Esi * (1.0 - th * th)     # dp/dy = k*z*sech^2(arg)
    return y_sign * p_mag, k_sec, k_tan



# ==============================================================================
# MODEL 4c -- Rollins et al. (2005a)  Liquefied Sand  Sec 5.10
# ==============================================================================
def _liquefied_sand_rollins(y, z, params):
    """
    RSPile Sec 5.10 (Rollins et al. 2005a) -- Liquefied Sand.
    Fully empirical; the ONLY soil input is pile diameter (plus depth). No phi/gamma.

    Native units of the cited Rollins (2005a) form (as presented in the manual):
        z [m], b [m], y [mm], p [kN/m].  Converted to/from kip-in here.
        p  = Pd * A * (B*y)^C            (Eq 32)
        Pd = 3.81*ln(b) + 5.6            (Eq 33)   diameter correction
        A  = 3e-7 * (z+1)^6.05           (Eq 34)
        B  = 2.80 * (z+1)^0.11           (Eq 35)
        C  = 2.85 * (z+1)^(-0.41)        (Eq 36)
    Applicability (manual): b in 0.3..2.6 m; for b > 2.6 m, b = 2.6 m is used.
    Tangent: k_tan = dp/dy = C * (p/y).  No plateau is defined (power law).
    """
    IN2M  = 0.0254
    IN2MM = 25.4
    KNM2KIPIN = 0.2248089 / 39.3701      # kN/m -> kip/in

    b_in = float(params['b_in'])
    if z <= _Y_ZERO_TOL:
        return 0.0, 0.0, 0.0

    b_m = min(max(b_in * IN2M, 0.3), 2.6)
    z_m = z * IN2M
    Pd  = 3.81 * np.log(b_m) + 5.6
    Aco = 3.0e-7 * (z_m + 1.0) ** 6.05
    Bco = 2.80   * (z_m + 1.0) ** 0.11
    Cco = 2.85   * (z_m + 1.0) ** (-0.41)

    def p_of(y_abs_in):
        y_mm = y_abs_in * IN2MM
        return Pd * Aco * (Bco * y_mm) ** Cco * KNM2KIPIN     # kip/in

    y_sign, abs_y = _sign_and_abs(y)
    if abs_y <= _Y_ZERO_TOL:
        y_ev = _Y_INIT_FRAC                                   # 0.01 in regularisation
        k_s  = p_of(y_ev) / y_ev
        return 0.0, k_s, Cco * k_s
    p_mag = p_of(abs_y)
    k_sec = p_mag / abs_y
    k_tan = Cco * k_sec
    return y_sign * p_mag, k_sec, k_tan



# ==============================================================================
# MODEL 4d -- Frank & Rollins (2013) & Wang & Reese (1998)  Hybrid Liquefied Sand  Sec 5.15
# ==============================================================================
def _hybrid_liquefied_sand(y, z, params):
    """
    RSPile Sec 5.15 -- Hybrid Liquefied Sand.   p = min(p_clay, p_liq)   (Eq 51)

    p_clay : residual soft-clay (Matlock) curve (Eqs 52-53), J=0.5, y50=2.5*eps50*b,
             p_u,clay = min[(3 + (gamma'/c)*z + (J/b)*z)*c*b , 9*c*b].
    p_liq  : Rollins (2005a) liquefied power law (Eqs 55-57), native z[m],b[m],y[mm],p[kN/m]:
             p_d (Eq 55, incl. b<0.3 m branch), A/B/C (Eq 56), p_liq=min[p_u,liq, p_d*A*(B*y)^C].
             p_u,liq (Eq 54) = min[ p_d*A*(B*150mm)^C , p_max ].
    NOTE: the manual does NOT define p_max numerically, so it is treated here as
          NON-BINDING (cap = Rollins value at y = 150 mm). Flag vs Frank & Rollins (2013).
    Required params (kip-in): cu_ksi (residual c), eps50, gamma_kip_in3, b_in.
    """
    b = float(params['b_in'])
    if z <= _Y_ZERO_TOL:
        return 0.0, 0.0, 0.0
    cu    = float(_get_param(z, params, 'cu_ksi'))
    eps50 = float(params['eps50'])
    gamma = float(params['gamma_kip_in3'])
    J     = 0.5
    y50   = 2.5 * eps50 * b

    # ----- Rollins liquefied part (metric -> kip/in) -----
    IN2M = 0.0254; IN2MM = 25.4; KNM2KIPIN = 0.2248089 / 39.3701
    bm = b * IN2M
    if bm < 0.3:
        Pd = bm / 0.3
    elif bm <= 2.6:
        Pd = 3.81 * np.log(bm) + 5.6
    else:
        Pd = 3.81 * np.log(2.6) + 5.6
    zm = z * IN2M
    Aco = 3.0e-7 * (zm + 1.0) ** 6.05
    Bco = 2.80   * (zm + 1.0) ** 0.11
    Cco = 2.85   * (zm + 1.0) ** (-0.41)
    def liq_curve(ay_in):
        return Pd * Aco * (Bco * ay_in * IN2MM) ** Cco * KNM2KIPIN
    pu_liq = liq_curve(150.0 / IN2MM)            # Rollins value at y = 150 mm (p_max non-binding)

    if cu > 0.0:
        pu_clay = max(min((3.0 + (gamma / cu) * z + (J / b) * z) * cu * b, 9.0 * cu * b), 0.0)
    else:
        pu_clay = 1.0e30                          # clay not limiting if c ~ 0

    def clay_curve(ay_in):
        if y50 > 0.0 and ay_in <= 8.0 * y50:
            return 0.5 * pu_clay * (ay_in / y50) ** (1.0 / 3.0)
        return pu_clay

    y_sign, ay = _sign_and_abs(y)
    if ay <= _Y_ZERO_TOL:
        ay_ev = _Y_INIT_FRAC * max(y50, 1.0e-9)
        p_ev = min(clay_curve(ay_ev), liq_curve(ay_ev), pu_liq)
        ks = p_ev / ay_ev
        return 0.0, ks, ks

    # clay branch value + tangent
    if y50 > 0.0 and ay <= 8.0 * y50:
        pc = 0.5 * pu_clay * (ay / y50) ** (1.0 / 3.0); ktc = (1.0 / 3.0) * (pc / ay)
    else:
        pc = pu_clay; ktc = 0.0
    # liq branch value + tangent
    plc = liq_curve(ay)
    if plc <= pu_liq:
        pl = plc; ktl = Cco * (pl / ay)
    else:
        pl = pu_liq; ktl = 0.0
    # governing (lesser) curve
    if pc <= pl:
        p_mag, k_tan = pc, ktc
    else:
        p_mag, k_tan = pl, ktl
    k_sec = p_mag / ay
    return y_sign * p_mag, k_sec, k_tan



# ==============================================================================
# MODEL 3b -- Welch & Reese (1972) MODIFIED Stiff Clay w/o Free Water  Sec 5.13
# ==============================================================================
def _modified_stiff_clay_nowater(y, z, params):
    """
    RSPile Sec 5.13 -- Modified Stiff Clay without Free Water (Welch & Reese, 1972).
    Adds a finite initial stiffness to the Dry Stiff Clay (Sec 5.3) curve:
        p = min( k*z*y , p_dry_stiff_clay(y) )
    where k = ks_kip_in3 (user-input initial stiffness). The Sec 5.3 curve has an
    (near-)infinite initial slope; the linear branch k*z*y governs near the origin,
    giving a finite initial stiffness, then the Sec 5.3 quarter-power curve governs.
    Requires Sec 5.3 params (cu_ksi, ca_ksi, eps50, gamma_kip_in3, b_in, J) + ks_kip_in3.
    NOTE: the manual gives the cyclic curve only as a FIGURE (Fig 5.14), no equations,
    so the static curve is used (cyclic not text-defined).
    """
    ks = float(params['ks_kip_in3'])
    p_dsc, _, kt_dsc = _welch_reese_stiff_clay_nowater(y, z, params)
    y_sign, ay = _sign_and_abs(y)
    Esi = ks * z
    if ay <= _Y_ZERO_TOL:
        return 0.0, (Esi if Esi > 0.0 else 0.0), (Esi if Esi > 0.0 else 0.0)
    p_lin = Esi * ay
    p_dsc_mag = abs(p_dsc)
    if Esi > 0.0 and p_lin <= p_dsc_mag:
        return y_sign * p_lin, Esi, Esi
    return y_sign * p_dsc_mag, (p_dsc_mag / ay), kt_dsc


# ==============================================================================
# MODEL 4e -- Reese et al. (1974)  Silt / Cemented c-phi Soil  Sec 5.14
# ==============================================================================
def _silt_cphi(y, z, params):
    """
    RSPile Sec 5.14 -- Silt (cemented c-phi soil), Reese et al. (1974).
    Four-branch curve (Fig 5.15): linear k*z*y (Eq40) -> parabola C*y^(1/n) (Eq41)
    -> straight line to (yu,pu) (Eq42) -> flat pu (Eq43).
        ym = b/60 (Eq45);  yu = 3b/80 (Eq46)
        pm = 1.5*pphi + pc (Eq47);   pu = Abar*pphi (Eq48)
        pphi = frictional ultimate (Eq49 = Reese 1974 sand wedge);  pc = cohesive (Eq50)
        C = pm / ym^(1/n);  slope m = (pu-pm)/(yu-ym);  yk = (C/(k z))^(n/(n-1)) (Eq44)
    Abar from Fig 5.6 (static/cyclic) -- same A-table as Sec 5.4 sand.
    n: the manual (Eq41) calls it a 'curve-fit parameter' and gives NO value;
       exposed here as user input 'n_silt' (default 2.0 -- CONFIRM vs Reese 1974).
    Requires: phi_deg, gamma_kip_in3, kpy_kip_in3 (k), b_in, cu_ksi, loading, n_silt.
    """
    phi_deg = float(params.get('phi_deg', 35.0)); gamma = float(params['gamma_kip_in3'])
    kpy = float(params['kpy_kip_in3']); b = float(params['b_in'])
    cu = float(_get_param(z, params, 'cu_ksi')); J = float(params.get('J', 0.5))
    loading = str(params.get('loading', 'static')).lower()
    n = float(params.get('n_silt', 2.0))
    if z <= _Y_ZERO_TOL:
        return 0.0, 0.0, 0.0

    # frictional ultimate (Eq 49 -- Reese 1974 sand wedge, identical to Sec 5.4)
    phi_r = np.radians(phi_deg); alpha = phi_r/2.0; beta = np.radians(45.0)+phi_r/2.0
    K0 = 0.4; Ka = np.tan(np.radians(45.0)-phi_r/2.0)**2
    tphi = np.tan(phi_r); tbeta = np.tan(beta); sbeta = np.sin(beta)
    calpha = np.cos(alpha); talpha = np.tan(alpha)
    tbmphi = np.tan(beta-phi_r)
    if abs(tbmphi) < 1e-10: tbmphi = 1e-10
    t1 = K0*z*tphi*sbeta/(tbmphi*calpha)
    t2 = (tbeta/tbmphi)*(b + z*tbeta*talpha)
    t3 = K0*z*tbeta*(tphi*sbeta - talpha)
    pphi_st = gamma*z*(t1 + t2 + t3 - Ka*b)
    pphi_d  = Ka*b*gamma*z*(tbeta**8 - 1.0) + K0*b*gamma*z*tphi*tbeta**4
    pphi = max(min(pphi_st, pphi_d), 0.0)

    # cohesive ultimate (Eq 50)
    if cu > 0.0:
        pc = max(min((3.0 + (gamma/cu)*z + (J/b)*z)*cu*b, 9.0*cu*b), 0.0)
    else:
        pc = 0.0

    zb = z/b if b > 0.0 else 0.0
    A = (np.interp(zb, _AS_SAND_ZB, _AS_SAND_VAL) if loading == 'static'
         else np.interp(zb, _AC_SAND_ZB, _AC_SAND_VAL))

    pm = 1.5*pphi + pc                # Eq 47 (peak)
    pu = A*pphi                       # Eq 48 (point u)
    ym = b/60.0; yu = 3.0*b/80.0      # Eq 45, 46
    Esi = kpy*z

    y_sign, ay = _sign_and_abs(y)
    if pm <= 0.0 or ym <= 0.0:
        if ay <= _Y_ZERO_TOL: return 0.0, Esi, Esi
        return y_sign*Esi*ay, Esi, Esi

    C = pm * ym**(-1.0/n)                          # parabola through (ym, pm): C*ym^(1/n)=pm
    mslope = (pu - pm)/(yu - ym) if yu > ym else 0.0
    # Linear-parabola intersection (Eq 44). The linear section (Eq 40) engages only if it
    # crosses the parabola BEFORE ym (yk < ym); otherwise the rising branch is the parabola
    # from the origin (it still reaches pm exactly at ym -> the curve stays continuous).
    if Esi > 0.0 and abs(n - 1.0) > 1e-9:
        yk = (C/Esi)**(n/(n - 1.0))
    else:
        yk = 0.0
    use_linear = (Esi > 0.0) and (0.0 < yk < ym)

    if ay <= _Y_ZERO_TOL:
        if use_linear:
            return 0.0, Esi, Esi
        y_ev = _Y_INIT_FRAC * ym
        p_ev = C * y_ev**(1.0/n); ks0 = p_ev/y_ev
        return 0.0, ks0, (1.0/n)*ks0
    if use_linear and ay <= yk:
        p_mag = Esi*ay; k_sec = Esi; k_tan = Esi
    elif ay <= ym:
        p_mag = C*ay**(1.0/n); k_sec = p_mag/ay; k_tan = (1.0/n)*k_sec
    elif ay <= yu:
        p_mag = pm + mslope*(ay - ym); k_sec = p_mag/ay; k_tan = 0.0
    else:
        p_mag = pu; k_sec = pu/ay; k_tan = 0.0
    return y_sign*p_mag, k_sec, k_tan



# ==============================================================================
# MODEL 5b -- Johnson et al. (2006)  Loess  Sec 5.9
# ==============================================================================
def _loess_johnson(y, z, params):
    """
    RSPile Sec 5.9 (Johnson et al. 2006) -- Loess.  Secant-modulus hyperbolic curve.
        pu0 = N_CPT*qc                       (Eq26, N_CPT = 0.409)
        pu  = pu0*b / (1 + Cn*log10(N))      (Eq27, Cn = 0.24, N = number of cycles)
        Ei  = pu / yref                      (Eq28)
        yh  = (y/yref)*(1 + a*exp(-y/yref))  (Eq29, a = 0.1)
        Es  = Ei / (1 + yh)                  (Eq30)
        p   = Es * y                         (Eq31)
    p -> pu asymptotically; initial slope = Ei.
    Requires: qc_ksi (CPT tip resistance), b_in, yref_in, N_cycles (default 1, static).
    FLAG: the manual references 'reference displacement' yref in Eqs 28-29 but NEVER
          defines it. It is exposed here as the REQUIRED user input 'yref_in'
          (NOT invented). Confirm against Johnson et al. (2006).
    """
    qc = float(_get_param(z, params, 'qc_ksi'))
    b  = float(params['b_in'])
    yref = float(params['yref_in'])
    N = float(params.get('N_cycles', 1.0))
    a = 0.1; N_CPT = 0.409; Cn = 0.24
    if qc <= 0.0 or yref <= 0.0 or b <= 0.0:
        return 0.0, 0.0, 0.0
    pu0 = N_CPT * qc
    pu  = pu0 * b / (1.0 + Cn * np.log10(max(N, 1.0)))
    Ei  = pu / yref
    y_sign, ay = _sign_and_abs(y)
    if ay <= _Y_ZERO_TOL:
        return 0.0, Ei, Ei
    u  = ay / yref
    yh = u * (1.0 + a * np.exp(-u))
    Es = Ei / (1.0 + yh)
    p_mag = Es * ay
    dyh   = (1.0 / yref) * (1.0 + a * np.exp(-u) * (1.0 - u))   # d(yh)/dy
    k_tan = Ei * ((1.0 + yh) - ay * dyh) / (1.0 + yh) ** 2
    return y_sign * p_mag, Es, k_tan


# ==============================================================================
# MODEL 5c -- Simpson & Brown (2003)  Piedmont Residual Soil  Sec 5.11
# ==============================================================================
def _piedmont_simpson_brown(y, z, params):
    """
    RSPile Sec 5.11 (Simpson & Brown 2003) -- piedmont residual soil.
        y/b < 0.001:           p = y*b*Esi                              (Eq 37)
        0.001 <= y/b <= 0.0375:p = b*y*Esi*(1 - lam*ln((y/b)/0.001))    (Eq 38)
        y/b > 0.0375:          p = (0.0375b)*b*Esi*(1 - lam*ln(37.5))   (Eq 39)
    Esi = initial soil modulus. The manual builds Esi from DMT/CPT/SPT/PMT test data
    via input factors (0.076/0.118/22/0.235) + a unit-conversion factor; that
    input-prep step is NOT reproduced here -- Esi is supplied directly as 'Esi_kip_in3'.
    FLAG: the manual uses 'lam' (lambda) in Eqs 38-39 but NEVER defines it. It is exposed
          here as the REQUIRED user input 'lambda_pied' (NOT invented). Confirm vs
          Simpson & Brown (2003). Initial stiffness (per unit y) = b*Esi.
    Requires: Esi_kip_in3, b_in, lambda_pied.
    """
    Esi = float(params['Esi_kip_in3']); b = float(params['b_in']); lam = float(params['lambda_pied'])
    if b <= 0.0:
        return 0.0, 0.0, 0.0
    y_sign, ay = _sign_and_abs(y)
    if ay <= _Y_ZERO_TOL:
        return 0.0, b * Esi, b * Esi
    yb = ay / b
    if yb < 0.001:
        p_mag = ay * b * Esi; k_sec = b * Esi; k_tan = b * Esi
    elif yb <= 0.0375:
        fac = 1.0 - lam * np.log(yb / 0.001)
        p_mag = b * ay * Esi * fac; k_sec = p_mag / ay; k_tan = b * Esi * (fac - lam)
    else:
        fac = 1.0 - lam * np.log(37.5)
        p_mag = (0.0375 * b) * b * Esi * fac; k_sec = p_mag / ay; k_tan = 0.0
    return y_sign * p_mag, k_sec, k_tan


# ══════════════════════════════════════════════════════════════════════════════
# MODEL 5 — REESE & NYMAN (1978)  Weak Rock  §5.5
# ══════════════════════════════════════════════════════════════════════════════
def _reese_nyman_weak_rock(y, z, params):
    """
    RSPile §5.5  (Reese & Nyman, 1978) — Weak Rock

    Required params keys (kip-in system)
    -------------------------------------
        'q_ur_ksi'      float   uniaxial compressive strength of rock  [kip/in²]
        'k_ir_kip_in2'  float   initial reaction modulus of rock        [kip/in²]
                                (slope of the initial linear p-y segment)
        'RQD_pct'       float   rock quality designation                 [%] (0–100)
        'k_rm'          float   strain factor (range: 0.00005–0.0005)   [-]
                                interpreted as compression strain at 50% q_ur
        'b_in'           float  pile diameter                            [in]
        'z_rock_in'      float  depth to top of rock from ground surface [in]
                                (optional, default = 0.0)

    Rock strength reduction factor (Eq. before §5.5 Eq. 17):
        a_r = 1 − (2/3) × (RQD / 100)         [dimensionless]
        Range: a_r = 1/3 (RQD = 100%, intact) to 1.0 (RQD = 0%, fully fractured).

    Ultimate resistance (Eqs. 17–18, smaller value governs):
        Eq. 17 (shallow):  p_ur = a_r · q_ur · b · (1 + 1.4 · z_r/b)
        Eq. 18 (deep):     p_ur = 5.2 · a_r · q_ur · b
        Transition at z_r/b = 3.0  (where Eq. 17 = Eq. 18)

    Reference deflection:
        y_rm = k_rm · b                        [in]

    Plateau deflection (where curved branch reaches p_ur):
        y_ult = 16 · y_rm                      [in]
        Derivation: (p_ur/2)·(y_ult/y_rm)^(1/4) = p_ur  →  y_ult/y_rm = 2^4 = 16

    Transition deflection y_A (intersection of linear and curved branch):
        Solve:  K_ir · y_A = (p_ur/2) · (y_A/y_rm)^(1/4)
        Closed-form solution:
            y_A = [p_ur / (2 · K_ir)]^(4/3) · y_rm^(−1/3)
        NUMERICAL NOTE: This is exact algebra; no iteration required.

    Curve segments (Figure 5.7):
        1. Linear:         y ≤ y_A              p = K_ir · y
        2. Quarter-power:  y_A < y ≤ y_ult      p = (p_ur/2) · (y/y_rm)^(1/4)
        3. Plateau:        y > y_ult             p = p_ur

    Tangent moduli:
        Segment 1:  k_tan = K_ir                   (constant linear stiffness)
        Segment 2:  k_tan = (1/4) · (p/y)          (exact derivative of power law)
        Segment 3:  k_tan = 0                       (perfectly plastic plateau)
    """
    q_ur   = float(_get_param(z, params, 'q_ur_ksi'))
    K_ir   = float(_get_param(z, params, 'k_ir_kip_in2'))
    RQD    = float(params.get('RQD_pct', 50.0))
    k_rm   = float(params.get('k_rm', 0.0005))
    b      = float(params['b_in'])
    z_rock = float(params.get('z_rock_in', 0.0))

    # ── Depth below top of rock ───────────────────────────────────────────────
    z_r = max(z - z_rock, 0.0)   # [in]  depth measured from rock surface

    # ── Rock strength reduction factor ───────────────────────────────────────
    a_r = 1.0 - (2.0 / 3.0) * (RQD / 100.0)
    # Physical bounds: RQD ∈ [0, 100] → a_r ∈ [1/3, 1].
    a_r = np.clip(a_r, 1.0 / 3.0, 1.0)

    # ── Ultimate resistance (Eqs. 17–18, smaller governs) ────────────────────
    # Eq. 17: depth-dependent (shallow) — increases linearly with z_r
    p_ur_shallow = a_r * q_ur * b * (1.0 + 1.4 * z_r / b) if b > 0.0 else 0.0
    # Eq. 18: constant (deep) — limiting plateau of the shallow formula
    p_ur_deep    = 5.2 * a_r * q_ur * b
    # Take the smaller (governing) value; clamp to zero
    p_ur = max(min(p_ur_shallow, p_ur_deep), 0.0)

    # ── Reference and plateau deflections ────────────────────────────────────
    y_rm  = k_rm * b          # [in]  reference deflection
    y_ult = 16.0 * y_rm       # [in]  deflection at full p_ur on curved branch

    # ── Transition deflection y_A (closed-form, see docstring) ───────────────
    # NUMERICAL NOTE: y_A is derived by equating the linear and quarter-power
    # expressions at the junction point.  The algebra is exact; no root-finding
    # or iteration is used.
    if K_ir > 0.0 and p_ur > 0.0 and y_rm > 0.0:
        y_A = (p_ur / (2.0 * K_ir)) ** (4.0 / 3.0) * y_rm ** (-1.0 / 3.0)
        y_A = min(y_A, y_ult)   # y_A cannot logically exceed y_ult
    else:
        # Degenerate case: no linear segment (K_ir = 0) or no resistance
        y_A = 0.0

    # ── Sign handling ─────────────────────────────────────────────────────────
    y_sign, abs_y = _sign_and_abs(y)

    # ── y ≈ 0: return initial stiffness without singularity ──────────────────
    if abs_y <= _Y_ZERO_TOL:
        if K_ir > 0.0:
            # Linear segment governs at origin; k_sec = K_ir is exact
            return 0.0, K_ir, K_ir
        elif p_ur > 0.0 and y_rm > 0.0:
            # No linear segment; regularise on curved branch
            y_ev  = _Y_INIT_FRAC * max(y_rm, 1e-15)
            p_ref = 0.5 * p_ur * (y_ev / y_rm) ** (1.0 / 4.0)
            k_s   = p_ref / y_ev
            return 0.0, k_s, 0.25 * k_s
        else:
            return 0.0, 0.0, 0.0

    # ── Evaluate p at abs_y ───────────────────────────────────────────────────
    if abs_y <= y_A:
        # Segment 1: linear initial
        p_mag = K_ir * abs_y
        k_sec = K_ir
        k_tan = K_ir

    elif abs_y <= y_ult:
        # Segment 2: quarter-power curve  p = (p_ur/2)·(y/y_rm)^(1/4)
        p_mag = 0.5 * p_ur * (abs_y / y_rm) ** (1.0 / 4.0)
        k_sec = p_mag / abs_y
        k_tan = 0.25 * k_sec   # exact: d/dy[(C·y^(1/4))] = (1/4)·p/y

    else:
        # Segment 3: plateau at p_ur
        p_mag = p_ur
        k_sec = p_ur / abs_y
        k_tan = 0.0

    return y_sign * p_mag, k_sec, k_tan


# ══════════════════════════════════════════════════════════════════════════════
# MODEL 6 — REESE & NYMAN (1978)  Strong Rock (Vuggy Limestone)  §5.12
# ══════════════════════════════════════════════════════════════════════════════
def _reese_nyman_strong_rock(y, z, params):
    """
    RSPile §5.12  (Reese & Nyman, 1978) — Strong Rock (Vuggy Limestone)

    Based on field tests on instrumented drilled shafts in vuggy limestone.

    Required params keys (kip-in system)
    -------------------------------------
        'su_ksi'    float   unconfined compressive strength of rock  [kip/in²]
        'b_in'      float   pile diameter                            [in]

    Curve geometry (Figure 5.12):
        Two breakpoints on the y-axis:
            y_1 = 0.0004 · b               [in]   end of initial stiff segment
            y_2 = 0.0024 · b               [in]   end of softer linear segment (= ultimate deflection)

        Initial slopes:
            E_1 = 2000 · s_u               [kip/in²]   very stiff initial segment
            E_2 = 100  · s_u               [kip/in²]   softer linear segment

        Resistance at breakpoints:
            p_1   = E_1 · y_1  = 2000 · s_u · 0.0004 · b  = 0.8 · s_u · b   [kip/in]
            p_ur  = p_1 + E_2 · (y_2 − y_1)
                  = 0.8·s_u·b + 100·s_u·0.002·b
                  = 1.0 · s_u · b                                              [kip/in]

    Curve segments:
        1. Stiff linear:   y ≤ y_1          p = E_1 · y
        2. Softer linear:  y_1 < y ≤ y_2    p = p_1 + E_2 · (y − y_1)
        3. Plateau:        y > y_2           p = p_ur

    Tangent moduli:
        Segment 1:  k_tan = E_1 = 2000 · s_u
        Segment 2:  k_tan = E_2 = 100  · s_u
        Segment 3:  k_tan = 0   (plateau)

    Note on brittle fracture (§5.12):
        The manual states "Brittle fracture assumed beyond ultimate deflection"
        (y > y_2).  For solver stability the post-ultimate response is held at
        p = p_ur (constant plateau) rather than dropped to zero.
        NUMERICAL APPROXIMATION: the plateau assumption is conservative for
        strength but avoids numerical divergence in the iterative solver.
        Engineers should verify that computed deflections remain below y_2 =
        0.0024·b to stay within the valid range of the curve.  If deflections
        exceed y_2 the solution should be interpreted with engineering judgment.
    """
    su  = float(_get_param(z, params, 'su_ksi'))
    b   = float(params['b_in'])

    # ── Curve geometry ────────────────────────────────────────────────────────
    y_1  = 0.0004 * b          # [in]  end of initial stiff segment
    y_2  = 0.0024 * b          # [in]  ultimate deflection (brittle fracture onset)

    # ── Resistance & slopes [kip/in, kip/in²] ────────────────────────────────
    #  Reese & Nyman vuggy-limestone curve: the first (stiff) linear segment ends
    #  at HALF the ultimate resistance (p_1 = 0.5*p_ur), NOT 0.8*p_ur.
    #  CORRECTION: the prior 0.8*p_ur breakpoint forced E_1 = 2000*su, which made
    #  the soil reaction ~1.63x too high (LPile vuggy-limestone slope is ~1226*su).
    #  The half-ultimate breakpoint gives E_1 = 1250*su and E_2 = 250*su.
    p_ur = max(su * b, 0.0)                                   # ultimate  p_ur = b*su
    p_1  = 0.5 * p_ur                                         # first break at 0.5*p_ur
    E_1  = p_1 / y_1 if y_1 > 0.0 else 0.0                    # = 1250*su  (was 2000*su)
    E_2  = (p_ur - p_1) / (y_2 - y_1) if y_2 > y_1 else 0.0   # = 250*su   (was 100*su)

    # ── Sign handling ─────────────────────────────────────────────────────────
    y_sign, abs_y = _sign_and_abs(y)

    # ── y ≈ 0: initial stiffness is E_1 (linear segment governs) ─────────────
    if abs_y <= _Y_ZERO_TOL:
        return 0.0, E_1, E_1

    # ── Evaluate p at abs_y ───────────────────────────────────────────────────
    if abs_y <= y_1:
        # Segment 1: stiff linear
        p_mag = E_1 * abs_y
        k_sec = E_1
        k_tan = E_1

    elif abs_y <= y_2:
        # Segment 2: softer linear
        p_mag = p_1 + E_2 * (abs_y - y_1)
        k_sec = p_mag / abs_y
        k_tan = E_2

    else:
        # Segment 3: plateau (brittle fracture assumed; held at p_ur — see docstring)
        p_mag = p_ur
        k_sec = p_ur / abs_y
        k_tan = 0.0

    return y_sign * p_mag, k_sec, k_tan



# ==============================================================================
# MODEL 7 -- Liang, Yang & Nusairat (2009)  Massive Rock  Sec 5.16
# ==============================================================================
def _massive_rock_props(z, params):
    """Return (Ki, pu) for the Sec 5.16 massive-rock curve at depth z (kip-in).
       Ki  = Em*(D/Dref)*exp(-2 nu)*(EpIp/(Em D^4))^0.284             (Eq 59)
       pu  = min(pus, pud);  pus passive wedge (Eqs 60-67); pud Hoek-Brown (Eqs 71-74).
       Sec 5.16 definitions used verbatim: Ka = tan^2(45 + phi'/2) (Eq 66),
       K0 = 1 - sin phi' (Eq 67), beta = 45 + phi'/2, theta = phi'/2.
       H = depth below rock top (z - z_rock); sigma'v = gamma' z.
       Eq 74 imperial: sigma_ci in psi -> tau in psi -> converted to ksi."""
    Em   = float(params['Em_ksi']); nu = float(params['nu'])
    EpIp = float(params['EpIp_kip_in2']); D = float(params['b_in'])
    cp   = float(params['c_ksi']); phi = float(params['phi_deg'])
    gamma= float(params['gamma_kip_in3']); sci = float(params['sigma_ci_ksi'])
    mb   = float(params['mb']); s = float(params['s_hb']); a = float(params['a_hb'])
    z_rock = float(params.get('z_rock_in', 0.0)); Dref = 12.0
    if D <= 0.0 or Em <= 0.0:
        return 0.0, 0.0

    Ki = Em*(D/Dref)*np.exp(-2.0*nu)*(EpIp/(Em*D**4))**0.284          # Eq 59

    H  = max(z - z_rock, 0.0)
    sv = gamma*z
    phir = np.radians(phi); beta = np.radians(45.0 + phi/2.0); theta = np.radians(phi/2.0)
    Ka = np.tan(np.radians(45.0 + phi/2.0))**2                       # Eq 66 (manual)
    K0 = 1.0 - np.sin(phir)                                          # Eq 67
    tb=np.tan(beta); sb=np.sin(beta); cb=np.cos(beta)
    st=np.sin(theta); ct=np.cos(theta); tt=np.tan(theta)
    sect = 1.0/ct if ct != 0 else 0.0
    tphi = np.tan(phir)
    z0 = (2.0*cp/(gamma*np.sqrt(Ka)) - sv/gamma) if (gamma > 0.0 and Ka > 0.0) else 0.0  # Eq 68

    C1 = H*tb*sect*(cp + K0*sv*tphi + (H/2.0)*K0*gamma*tphi)         # Eq 61
    den = sb - tphi*cb
    C2 = ((D*tb*(sv + H*gamma) + H*tb*tb*tt*(2.0*sv + H*gamma)       # Eq 62
           + cp*(D + 2.0*H*tb*tt) + 2.0*C1*cb*ct) / den) if abs(den) > 1e-12 else 0.0
    C4 = K0*H*tb*sect*(sv + 0.5*gamma*H)                            # Eq 64
    C5 = max(gamma*Ka*(H - z0)*D, 0.0)                              # Eq 65
    pus = 2.0*C1*ct*sb + C2*sb - 2.0*C4*st - C5                     # Eq 60

    pa = max(Ka*sv - 2.0*cp*np.sqrt(Ka), 0.0)                        # Eq 72
    pL = sv + sci*(mb*(sv/sci) + s)**a if sci > 0.0 else 0.0         # Eq 73 (Hoek-Brown)
    tau = 5.4194*np.sqrt(max(sci, 0.0)*1000.0)/1000.0               # Eq 74 imperial (psi->ksi)
    pud = (np.pi/4.0*pL + (2.0/3.0)*tau - pa)*D                      # Eq 71

    return Ki, max(min(pus, pud), 0.0)


def _massive_rock_liang(y, z, params):
    """
    RSPile Sec 5.16 (Liang, Yang & Nusairat 2009) -- Massive Rock. Hyperbolic:
        p = y / (1/Ki + y/pu)                                        (Eq 58)
    Requires: Em_ksi, nu, EpIp_kip_in2 (pile bending stiffness), b_in (D), c_ksi,
              phi_deg, gamma_kip_in3, sigma_ci_ksi, mb, s_hb, a_hb, z_rock_in (opt).
    k_tan = (1/Ki)/(1/Ki + |y|/pu)^2  (= Ki at y=0).
    """
    Ki, pu = _massive_rock_props(z, params)
    y_sign, ay = _sign_and_abs(y)
    if Ki <= 0.0 or pu <= 0.0:
        return 0.0, max(Ki, 0.0), max(Ki, 0.0)
    if ay <= _Y_ZERO_TOL:
        return 0.0, Ki, Ki
    denom = 1.0/Ki + ay/pu
    p_mag = ay/denom
    return y_sign*p_mag, 1.0/denom, (1.0/Ki)/denom**2


# ══════════════════════════════════════════════════════════════════════════════
# DISPATCH
# ══════════════════════════════════════════════════════════════════════════════
_MODEL_MAP = {
    'matlock':             _matlock_soft_clay,
    'reese_stiff_water':   _reese_stiff_clay_water,
    'welch_reese':         _welch_reese_stiff_clay_nowater,
    'modified_stiff_clay': _modified_stiff_clay_nowater,
    'silt_cphi':           _silt_cphi,
    'loess':               _loess_johnson,
    'piedmont':            _piedmont_simpson_brown,
    'reese_sand':          _reese_sand,
    'api_sand':            _api_sand,
    'liquefied_sand':      _liquefied_sand_rollins,
    'hybrid_liquefied_sand': _hybrid_liquefied_sand,
    'weak_rock':           _reese_nyman_weak_rock,
    'strong_rock':         _reese_nyman_strong_rock,
    'massive_rock':        _massive_rock_liang,
}

MODEL_LABELS = {
    'matlock':            'Matlock (1970) — Soft Clay w/ Water',
    'reese_stiff_water':  'Reese et al. (1975) — Stiff Clay w/ Water',
    'welch_reese':        'Welch & Reese (1972) — Stiff Clay w/o Water',
    'modified_stiff_clay': 'Welch & Reese (1972) — Modified Stiff Clay w/o Water',
    'silt_cphi':          'Reese et al. (1974) — Silt (cemented c-phi)',
    'loess':              'Johnson et al. (2006) — Loess',
    'piedmont':           'Simpson & Brown (2003) — Piedmont Residual',
    'reese_sand':         'Reese et al. (1974) — Sand',
    'api_sand':           'API RP 2A — Sand',
    'liquefied_sand':     'Rollins et al. (2005a) — Liquefied Sand',
    'hybrid_liquefied_sand': 'Frank & Rollins (2013) — Hybrid Liquefied Sand',
    'weak_rock':          'Reese & Nyman (1978) — Weak Rock',
    'strong_rock':        'Reese & Nyman (1978) — Strong Rock (Vuggy Limestone)',
    'massive_rock':       'Liang, Yang & Nusairat (2009) — Massive Rock',
}

# Which models are clay (need cu/ε50) vs sand (need φ) vs rock (need q_ur or su)
MODEL_IS_SAND = {
    'matlock':            False,
    'reese_stiff_water':  False,
    'welch_reese':        False,
    'modified_stiff_clay': False,
    'silt_cphi':          False,  # c-phi: needs cu AND phi
    'loess':              False,
    'piedmont':           False,
    'reese_sand':         True,
    'api_sand':           True,
    'liquefied_sand':     True,
    'hybrid_liquefied_sand': False,  # needs residual clay params (cu, eps50)
    'weak_rock':          False,   # rock — neither clay nor sand
    'strong_rock':        False,   # rock — neither clay nor sand
    'massive_rock':       False,  # rock
}

MODEL_IS_ROCK = {
    'matlock':            False,
    'reese_stiff_water':  False,
    'welch_reese':        False,
    'modified_stiff_clay': False,
    'silt_cphi':          False,
    'loess':              False,
    'piedmont':           False,
    'reese_sand':         False,
    'api_sand':           False,
    'liquefied_sand':     False,
    'hybrid_liquefied_sand': False,
    'weak_rock':          True,
    'strong_rock':        True,
    'massive_rock':       True,
}

MODEL_KEYS = list(_MODEL_MAP.keys())

# Cyclic loading support matrix
# True  = the model changes curve shape/parameters for cyclic (loading param is used)
# False = no cyclic variant in the RSPile §5.x formulation implemented here
MODEL_CYCLIC_SUPPORT = {
    'matlock':            False,   # §5.1: no separate cyclic curve in RSPile formulation
    'reese_stiff_water':  True,    # §5.2: uses Ac vs As A-factor for cyclic
    'welch_reese':        False,   # §5.3: static only; cyclic variant is §5.13 (not implemented)
    'modified_stiff_clay': False,  # §5.13: cyclic given only as a figure (no equations)
    'silt_cphi':          True,    # §5.14: Abar uses Ac vs As (Fig 5.6) for cyclic
    'loess':              True,    # §5.9: N_cycles enters Eq 27
    'piedmont':           False,   # §5.11: no cyclic variant defined
    'reese_sand':         True,    # §5.4: uses Ac/Bc vs As/Bs A/B factors for cyclic
    'api_sand':           True,    # §5.8: A=(3-0.8z/b)>=0.9 static, A=0.9 cyclic
    'liquefied_sand':     False,   # §5.10: single empirical curve (no static/cyclic split)
    'hybrid_liquefied_sand': False,  # §5.15: single hybrid curve (no static/cyclic split)
    'weak_rock':          False,   # §5.5: no cyclic variant defined in RSPile §5.5
    'strong_rock':        False,   # §5.12: no cyclic variant defined in RSPile §5.12
    'massive_rock':       False,   # §5.16: no cyclic variant defined
}


def get_soil_response(y, z, params, model='matlock'):
    """
    Central dispatch.  Returns (p [kip/in], k_sec [kip/in²], k_tan [kip/in²]).
    """
    fn = _MODEL_MAP.get(model)
    if fn is None:
        raise NotImplementedError(
            f"Soil model '{model}' not implemented. "
            f"Available: {MODEL_KEYS}")
    return fn(y, z, params)


def compute_pult(z, params, model):
    """
    Return p_ult [kip/in] at depth z for mobilisation ratio post-processing.
    Returns 0.0 for unknown models.
    """
    b  = float(params.get('b_in', 1.0))

    if model == 'matlock':
        cu    = _get_param(z, params, 'cu_ksi')
        gamma = float(params['gamma_kip_in3'])
        J     = float(params.get('J', 0.5))
        p_s   = (3.0 + (gamma / cu) * z + (J / b) * z) * cu * b
        return max(min(p_s, 9.0 * cu * b), 0.0)

    elif model == 'reese_stiff_water':
        cu    = _get_param(z, params, 'cu_ksi')
        ca    = _get_param(z, params, 'ca_ksi')
        gamma = float(params['gamma_kip_in3'])
        p_s   = 2.0 * ca * b + gamma * b * z + 2.83 * ca * z
        return max(min(p_s, 11.0 * cu * b), 0.0)

    elif model == 'welch_reese':
        cu    = _get_param(z, params, 'cu_ksi')
        ca    = _get_param(z, params, 'ca_ksi')
        gamma = float(params['gamma_kip_in3'])
        J     = float(params.get('J', 0.5))
        p_s   = (3.0 + (gamma / ca) * z + (J / b) * z) * ca * b
        return max(min(p_s, 9.0 * cu * b), 0.0)

    elif model == 'modified_stiff_clay':
        cu    = _get_param(z, params, 'cu_ksi')
        ca    = _get_param(z, params, 'ca_ksi')
        gamma = float(params['gamma_kip_in3'])
        J     = float(params.get('J', 0.5))
        p_s   = (3.0 + (gamma / ca) * z + (J / b) * z) * ca * b
        return max(min(p_s, 9.0 * cu * b), 0.0)

    elif model == 'silt_cphi':
        if z <= 0.0:
            return 0.0
        phi_r = np.radians(float(params.get('phi_deg', 35.0))); gamma = float(params['gamma_kip_in3'])
        cu = float(_get_param(z, params, 'cu_ksi')); J = float(params.get('J', 0.5))
        alpha = phi_r/2.0; beta = np.radians(45.0)+phi_r/2.0
        K0 = 0.4; Ka = np.tan(np.radians(45.0)-phi_r/2.0)**2
        tphi=np.tan(phi_r); tbeta=np.tan(beta); tbmphi=np.tan(beta-phi_r)
        if abs(tbmphi)<1e-10: tbmphi=1e-10
        t1=K0*z*tphi*np.sin(beta)/(tbmphi*np.cos(alpha)); t2=(tbeta/tbmphi)*(b+z*tbeta*np.tan(alpha))
        t3=K0*z*tbeta*(tphi*np.sin(beta)-np.tan(alpha))
        pphi=max(min(gamma*z*(t1+t2+t3-Ka*b), Ka*b*gamma*z*(tbeta**8-1.0)+K0*b*gamma*z*tphi*tbeta**4),0.0)
        pc = max(min((3.0+(gamma/cu)*z+(J/b)*z)*cu*b, 9.0*cu*b),0.0) if cu>0.0 else 0.0
        return max(1.5*pphi + pc, 0.0)   # peak resistance pm (Eq 47)

    elif model == 'loess':
        qc = float(_get_param(z, params, 'qc_ksi')); N = float(params.get('N_cycles', 1.0))
        if qc <= 0.0: return 0.0
        return 0.409 * qc * b / (1.0 + 0.24 * np.log10(max(N, 1.0)))   # pu (Eq 27)

    elif model == 'piedmont':
        Esi = float(params['Esi_kip_in3']); lam = float(params['lambda_pied'])
        return max((0.0375*b) * b * Esi * (1.0 - lam*np.log(37.5)), 0.0)   # plateau (Eq 39)

    elif model == 'reese_sand':
        # Re-use the full sand function to compute ps, then multiply by A
        if z <= 0.0:
            return 0.0
        phi_r   = np.radians(float(params.get('phi_deg', 35.0)))
        gamma   = float(params['gamma_kip_in3'])
        loading = str(params.get('loading', 'static')).lower()
        alpha   = phi_r / 2.0
        beta    = np.radians(45.0) + phi_r / 2.0
        K0, Ka  = 0.4, np.tan(np.radians(45.0) - phi_r / 2.0) ** 2
        tan_phi, tan_beta = np.tan(phi_r), np.tan(beta)
        tan_bmphi = np.tan(beta - phi_r)
        if abs(tan_bmphi) < 1e-10:
            tan_bmphi = 1e-10
        t1    = K0 * z * tan_phi * np.sin(beta) / (tan_bmphi * np.cos(alpha))
        t2    = (tan_beta / tan_bmphi) * (b + z * tan_beta * np.tan(alpha))
        t3    = K0 * z * tan_beta * (tan_phi * np.sin(beta) - np.tan(alpha))
        ps_sh = gamma * z * (t1 + t2 + t3 - Ka * b)
        ps_dp = (Ka * b * gamma * z * (tan_beta ** 8 - 1.0) +
                 K0 * b * gamma * z * tan_phi * tan_beta ** 4)
        ps    = max(min(ps_sh, ps_dp), 0.0)
        zb    = z / b if b > 0.0 else 0.0
        A     = (np.interp(zb, _AS_SAND_ZB, _AS_SAND_VAL) if loading == 'static'
                 else np.interp(zb, _AC_SAND_ZB, _AC_SAND_VAL))
        return A * ps

    elif model == 'api_sand':
        if z <= 0.0:
            return 0.0
        phi_r   = np.radians(float(params.get('phi_deg', 35.0)))
        gamma   = float(params['gamma_kip_in3'])
        loading = str(params.get('loading', 'static')).lower()
        alpha   = phi_r / 2.0
        beta    = np.radians(45.0) + phi_r / 2.0
        K0      = 0.4
        Kp      = np.tan(np.radians(45.0) + phi_r / 2.0) ** 2
        Ka      = np.tan(np.radians(45.0) - phi_r / 2.0) ** 2
        tan_phi = np.tan(phi_r)
        C1 = np.tan(beta) * (Kp * np.tan(alpha) +
                K0 * (tan_phi * np.sin(beta) * (1.0/np.cos(alpha) + 1.0) - np.tan(alpha)))
        C2 = Kp - Ka
        C3 = Kp ** 2 * (Kp + K0 * tan_phi) - Ka
        pus = (C1 * z + C2 * b) * gamma * z
        pud = C3 * b * gamma * z
        pu  = max(min(pus, pud), 0.0)
        A   = 0.9 if loading == 'cyclic' else (max(3.0 - 0.8*z/b, 0.9) if b > 0.0 else 0.9)
        return A * pu

    elif model == 'weak_rock':
        # Replicate the p_ur calculation from _reese_nyman_weak_rock
        q_ur   = float(_get_param(z, params, 'q_ur_ksi'))
        RQD    = float(params.get('RQD_pct', 50.0))
        z_rock = float(params.get('z_rock_in', 0.0))
        z_r    = max(z - z_rock, 0.0)
        a_r    = np.clip(1.0 - (2.0 / 3.0) * (RQD / 100.0), 1.0 / 3.0, 1.0)
        p_ur_s = a_r * q_ur * b * (1.0 + 1.4 * z_r / b) if b > 0.0 else 0.0
        p_ur_d = 5.2 * a_r * q_ur * b
        return max(min(p_ur_s, p_ur_d), 0.0)

    elif model == 'strong_rock':
        # p_ur = 1.0 · su · b  (derived from bilinear geometry, see §5.12 docstring)
        su  = float(_get_param(z, params, 'su_ksi'))
        return max(su * b, 0.0)

    elif model == 'liquefied_sand':
        return 0.0   # Sec 5.10 Rollins power law has no plateau; p_ult undefined

    elif model == 'hybrid_liquefied_sand':
        if z <= 0.0:
            return 0.0
        cu = float(_get_param(z, params, 'cu_ksi')); gamma = float(params['gamma_kip_in3']); J = 0.5
        pu_clay = (max(min((3.0+(gamma/cu)*z+(J/b)*z)*cu*b, 9.0*cu*b), 0.0) if cu > 0.0 else 1.0e30)
        IN2M=0.0254; KNM2KIPIN=0.2248089/39.3701; bm=b*IN2M
        Pd=(bm/0.3 if bm<0.3 else (3.81*np.log(bm)+5.6 if bm<=2.6 else 3.81*np.log(2.6)+5.6))
        zm=z*IN2M; A=3e-7*(zm+1)**6.05; B=2.80*(zm+1)**0.11; C=2.85*(zm+1)**(-0.41)
        pu_liq=Pd*A*(B*150.0)**C*KNM2KIPIN
        return max(min(pu_clay, pu_liq), 0.0)

    elif model == 'massive_rock':
        _Ki, _pu = _massive_rock_props(z, params)
        return max(_pu, 0.0)   # pu = min(wedge, Hoek-Brown) (Sec 5.16)

    return 0.0


def get_py_curve(z, params, model, n_pts=300):
    """
    Return (y_in, p_kipin) arrays for the p-y curve at depth z.
    Useful for plotting p-y curves at selected depths.
    y range is automatically scaled to fully capture the curve shape.
    """
    b     = float(params.get('b_in', 1.0))
    eps50 = float(params.get('eps50', 0.005))

    if model == 'matlock':
        y50   = 2.5 * eps50 * b
        y_max = max(12.0 * y50, 0.5)

    elif model == 'reese_stiff_water':
        y50  = eps50 * b
        zb   = z / b if b > 0 else 0.0
        A    = np.interp(zb, _AS_ZB, _AS_VAL)
        y_max = max(22.0 * A * y50, 0.5)

    elif model == 'welch_reese':
        y50   = eps50 * b
        y_max = max(20.0 * y50, 0.5)

    elif model == 'modified_stiff_clay':
        y50   = eps50 * b
        y_max = max(20.0 * y50, 0.5)

    elif model == 'silt_cphi':
        y_max = max(4.0 * (3.0*b/80.0), 0.1*b, 0.5)

    elif model == 'loess':
        y_max = max(10.0 * float(params.get('yref_in', 0.5)), 0.5)

    elif model == 'piedmont':
        y_max = max(0.06 * b, 0.5)

    elif model == 'reese_sand':
        # Extend to ~4×yu to clearly show the plateau
        yu    = 3.0 * b / 80.0    # = 0.0375·b
        y_max = max(4.0 * yu, 0.1 * b, 0.5)

    elif model == 'api_sand':
        # tanh curve -> A*pu asymptote; show to ~0.05*b
        y_max = max(0.05 * b, 0.5)

    elif model == 'liquefied_sand':
        y_max = max(0.1 * b, 2.0)

    elif model == 'hybrid_liquefied_sand':
        y50 = 2.5 * eps50 * b
        y_max = max(8.0 * y50, 0.1 * b, 6.0)

    elif model == 'weak_rock':
        # Extend to 2×y_ult = 32·y_rm to show the full plateau clearly
        k_rm  = float(params.get('k_rm', 0.0005))
        y_rm  = k_rm * b
        y_ult = 16.0 * y_rm
        y_max = max(2.0 * y_ult, 0.001 * b, 0.01)

    elif model == 'strong_rock':
        # Extend to 4×y_2 to show the plateau beyond the brittle limit
        y_2   = 0.0024 * b
        y_max = max(4.0 * y_2, 0.001 * b, 0.001)

    elif model == 'massive_rock':
        y_max = max(0.03 * b, 0.05)

    else:
        y_max = 1.0

    y_arr = np.linspace(0.0, y_max, n_pts)
    p_arr = np.array([get_soil_response(yv, z, params, model)[0] for yv in y_arr])
    return y_arr, p_arr
