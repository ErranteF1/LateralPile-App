"""
engine.py -- high-level driver that turns an engineering 'project' (US units) into a
solver run and tidy results. Wraps fem_solver (linear / secant p-y) and nonlinear_ei
(bilinear M-phi). All UI <-> solver unit conversion lives here and in defaults.py.
"""
import json, copy
import numpy as np
import fem_solver as F
import nonlinear_ei as NL
import defaults as D

# --------------------------------------------------------------------- defaults
def blank_project():
    return {
        'name': 'New project',
        'pile': dict(L_ft=40.0, D_in=36.0, n_elements=120,
                     head_fixed=False, tip_cond='free',
                     ei_mode='rc',                       # 'rc' | 'steel' | 'direct' | 'mphi'
                     fc_psi=4000.0,                      # rc
                     E_ksi=29000.0, I_in4=900.0,         # steel
                     EI_kip_in2=1.0e9,                   # direct
                     My_kip_in=1.5e5, alpha=0.05),       # mphi (bilinear)
        'layers': [   # each: z_top_ft, z_bot_ft, model, loading, fields{...}
            dict(z_top_ft=0.0, z_bot_ft=16.0, model='reese_sand', loading='static',
                 fields=dict(phi_deg=30.0, gamma_pcf=115.0, k_pci=0.0)),
            dict(z_top_ft=16.0, z_bot_ft=50.0, model='strong_rock', loading='static',
                 fields=dict(ucs_top_psi=1000.0, ucs_bot_psi=8000.0)),
        ],
        'loads': [dict(name='LC1', V_kip=50.0, M_kipft=0.0, P_kip=0.0)],
    }

# --------------------------------------------------------------- pile parameters
def build_pile_params(project):
    p = project['pile']
    L_in = p['L_ft'] * D.FT2IN
    pp = dict(L_in=L_in, n_elements=int(p['n_elements']),
              head_fixed=bool(p['head_fixed']), tip_cond=p['tip_cond'])
    mode = p.get('ei_mode', 'rc')
    if mode == 'rc':
        EI = D.EI_rc_circular(p['D_in'], p['fc_psi'])
    elif mode == 'steel':
        EI = D.EI_from_EI(p['E_ksi'], p['I_in4'])
    else:  # 'direct' or 'mphi' (mphi uses base EI as the elastic branch)
        EI = p['EI_kip_in2']
    pp['EI_kip_in2'] = float(EI)
    return pp

# ------------------------------------------------------------------ soil layers
def build_soil_layers(project):
    D_in = project['pile']['D_in']
    layers = []
    for L in project['layers']:
        z_top = L['z_top_ft'] * D.FT2IN
        z_bot = L['z_bot_ft'] * D.FT2IN
        model = L['model']
        f = L.get('fields', {})
        loading = L.get('loading', 'static')
        if model == 'strong_rock':
            # depth-varying UCS as su_ksi gradient (absolute-depth array)
            su = np.array([[z_top, f['ucs_top_psi']*D.PSI],
                           [z_bot, f['ucs_bot_psi']*D.PSI]])
            params = dict(su_ksi=su, b_in=D_in)
        else:
            params = D.SOIL_MODELS[model]['build'](f, D_in, z_top, loading)
        layers.append(dict(z_top_in=z_top, z_bot_in=z_bot, model=model, params=params))
    return layers

# ------------------------------------------------------------------- load cases
def build_load_cases(project):
    lcs = []
    for lc in project['loads']:
        lcs.append(dict(name=lc.get('name', ''),
                        H_kip=float(lc['V_kip']),
                        M_kip_in=float(lc.get('M_kipft', 0.0)) * D.FT2IN,
                        P_kip=float(lc.get('P_kip', 0.0))))
    return lcs

# ------------------------------------------------------------------------- run
def _depth_to_fixity_ft(z_in, y_in):
    """Shallowest depth (ft) where lateral deflection first reverses sign (zero crossing)."""
    y = np.asarray(y_in); z = np.asarray(z_in)
    s = np.sign(y)
    for i in range(1, len(y)):
        if s[i] != s[i-1] and s[i] != 0:
            # linear interpolation of the crossing
            z0 = z[i-1] - y[i-1]*(z[i]-z[i-1])/(y[i]-y[i-1])
            return float(z0 / D.FT2IN)
    return None

def run(project, max_iter=200, tol=1e-6):
    pp = build_pile_params(project)
    soil = build_soil_layers(project)
    lcs = build_load_cases(project)
    mode = project['pile'].get('ei_mode', 'rc')
    results = []
    if mode == 'mphi':
        My = project['pile']['My_kip_in']; alpha = project['pile'].get('alpha', 0.05)
        for lc in lcs:
            r = NL.solve_mphi(pp, soil, H_kip=lc['H_kip'], M_kip_in=lc['M_kip_in'],
                              P_axial=lc['P_kip'],
                              My_sections=[dict(z_top_in=0.0, z_bot_in=pp['L_in'], My_kip_in=My)],
                              alpha=alpha)
            r['name'] = lc['name']; results.append(r)
    else:
        results = F.run_analysis(pp, soil, lcs, max_iter=max_iter, tol=tol)
    for r in results:
        r['depth_to_fixity_ft'] = _depth_to_fixity_ft(r['z_in'], r['y_in'])
    return results, pp, soil

# ------------------------------------------------------------- summary table rows
def summary_row(r):
    z = np.asarray(r['z_in']) / D.FT2IN
    iM = int(np.argmax(np.abs(r['M_kip_in']))); iV = int(np.argmax(np.abs(r['V_kip'])))
    return dict(
        Case=r.get('name', ''),
        y_head_in=round(float(r['y_head_in']), 5),
        rotation_rad=round(float(r['theta_rad'][0]), 6),
        M_max_kipft=round(abs(r['M_kip_in'][iM]) / D.FT2IN, 1),
        z_Mmax_ft=round(float(z[iM]), 2),
        V_max_kip=round(abs(r['V_kip'][iV]), 1),
        z_Vmax_ft=round(float(z[iV]), 2),
        depth_to_fixity_ft=(round(r['depth_to_fixity_ft'], 2) if r.get('depth_to_fixity_ft') else None),
        converged=r.get('converged', True),
    )

# ----------------------------------------------------------------- save / load
def save_project(project, path):
    with open(path, 'w') as fh: json.dump(project, fh, indent=2)
def load_project(path):
    with open(path) as fh: return json.load(fh)
