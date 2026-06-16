"""
defaults.py -- units, theory-based default parameters, and the soil-model registry
that drives the UI. Engineering (US) units in the interface; everything converts to
the solver's internal kip-inch system here.
"""
import numpy as np

# ----------------------------------------------------------------------------- units
FT2IN = 12.0
PCF   = 1.0 / (1000.0 * 1728.0)   # pcf      -> kip/in^3
PCI   = 1.0 / 1000.0              # lb/in^3  -> kip/in^3
PSF   = 1.0 / (1000.0 * 144.0)    # psf      -> kip/in^2  (ksi)
PSI   = 1.0 / 1000.0              # psi      -> kip/in^2  (ksi)
KSF   = 1.0 / 144.0               # ksf      -> ksi

# ----------------------------------------- theory default subgrade modulus k(phi)
#  LPile/LAP auto-compute the sand initial modulus of subgrade reaction by
#  interpolating the API(2011)/Reese table on friction angle:
_KPHI_DEG  = np.array([25.0, 30.0, 35.0, 40.0])
_KPHI_KNM3 = np.array([5400.0, 11000.0, 22000.0, 45000.0])
KNM3_TO_PCI = 0.0036839
def default_k_sand_pci(phi_deg):
    """Default sand subgrade modulus k [pci] from friction angle (API 2011 / Reese)."""
    return float(np.interp(phi_deg, _KPHI_DEG, _KPHI_KNM3)) * KNM3_TO_PCI

# ------------------------------------------------------ structural stiffness helpers
def EI_rc_circular(D_in, fc_psi):
    """Gross (uncracked) EI of a circular RC section [kip-in^2]. Ec = 57000*sqrt(fc')."""
    Ec_ksi = 57000.0 * (fc_psi ** 0.5) / 1000.0
    Ig = np.pi * D_in ** 4 / 64.0
    return Ec_ksi * Ig
def EI_from_EI(E_ksi, I_in4):
    return E_ksi * I_in4

# ----------------------------------------------------------- soil-model registry
#  Each model: label, category, and the UI fields (engineering units) + converter to
#  the internal kip-in params dict the solver expects. `depth_grad=True` fields accept
#  a top/bottom value pair (linear with depth, absolute depth array).
#  Only the well-exercised / validated models are exposed in this 'core' version.
def _sand_params(f, b_in, z_top_in, loading):
    phi = f['phi_deg']
    k = f.get('k_pci') or default_k_sand_pci(phi)
    return dict(phi_deg=phi, gamma_kip_in3=f['gamma_pcf']*PCF,
                kpy_kip_in3=k*PCI, b_in=b_in, loading=loading)

SOIL_MODELS = {
    'reese_sand': dict(
        label='Sand — Reese et al. (1974)', category='Sand',
        fields=[('phi_deg','Friction angle phi','deg',32.0),
                ('gamma_pcf','Effective unit weight','pcf',115.0),
                ('k_pci','Subgrade modulus k (0 = auto from phi)','pci',0.0)],
        build=lambda f,b,z,ld: _sand_params(f,b,z,ld)),
    'api_sand': dict(
        label='Sand — API RP-2A', category='Sand',
        fields=[('phi_deg','Friction angle phi','deg',32.0),
                ('gamma_pcf','Effective unit weight','pcf',115.0),
                ('k_pci','Subgrade modulus k (0 = auto from phi)','pci',0.0)],
        build=lambda f,b,z,ld: _sand_params(f,b,z,ld)),
    'matlock': dict(
        label='Soft clay — Matlock (1970)', category='Clay',
        fields=[('cu_psf','Undrained shear strength cu','psf',500.0),
                ('eps50','Strain e50','-',0.010),
                ('gamma_pcf','Effective unit weight','pcf',110.0),
                ('J','Empirical J (0.25-0.5)','-',0.5)],
        build=lambda f,b,z,ld: dict(cu_ksi=f['cu_psf']*PSF, ca_ksi=f['cu_psf']*PSF,
                eps50=f['eps50'], gamma_kip_in3=f['gamma_pcf']*PCF, J=f['J'], b_in=b)),
    'welch_reese': dict(
        label='Stiff clay (no free water) — Welch & Reese', category='Clay',
        fields=[('cu_psf','Undrained shear strength cu','psf',2000.0),
                ('eps50','Strain e50','-',0.005),
                ('gamma_pcf','Effective unit weight','pcf',125.0),
                ('ks_pci','Subgrade modulus ks','pci',500.0)],
        build=lambda f,b,z,ld: dict(cu_ksi=f['cu_psf']*PSF, ca_ksi=f['cu_psf']*PSF,
                eps50=f['eps50'], gamma_kip_in3=f['gamma_pcf']*PCF, J=0.5,
                ks_kip_in3=f['ks_pci']*PCI, b_in=b)),
    'reese_stiff_water': dict(
        label='Stiff clay (with free water) — Reese et al.', category='Clay',
        fields=[('cu_psf','Undrained shear strength cu','psf',2000.0),
                ('eps50','Strain e50','-',0.005),
                ('gamma_pcf','Effective unit weight','pcf',125.0),
                ('ks_pci','Subgrade modulus ks','pci',500.0)],
        build=lambda f,b,z,ld: dict(cu_ksi=f['cu_psf']*PSF, ca_ksi=f['cu_psf']*PSF,
                eps50=f['eps50'], gamma_kip_in3=f['gamma_pcf']*PCF, J=0.5,
                ks_kip_in3=f['ks_pci']*PCI, b_in=b)),
    'weak_rock': dict(
        label='Weak rock — Reese (1997)', category='Rock',
        fields=[('qur_psi','Uniaxial compressive strength q_ur','psi',1000.0),
                ('kir_ksi','Initial rock modulus k_ir','ksi',300.0),
                ('RQD_pct','RQD','%',50.0),
                ('krm','Strain factor k_rm (1e-4..5e-4)','-',0.0005)],
        build=lambda f,b,z,ld: dict(q_ur_ksi=f['qur_psi']*PSI, k_ir_kip_in2=f['kir_ksi'],
                RQD_pct=f['RQD_pct'], k_rm=f['krm'], b_in=b, z_rock_in=z)),
    'strong_rock': dict(
        label='Strong rock / vuggy limestone — Reese & Nyman', category='Rock',
        fields=[('ucs_top_psi','UCS at layer top','psi',1000.0),
                ('ucs_bot_psi','UCS at layer bottom','psi',8000.0)],
        build=None),   # handled specially (depth gradient) in engine
}

LOADING_OPTIONS = ['static', 'cyclic']

# Convenience: ordered model keys grouped by category for the UI
def models_by_category():
    cats = {}
    for k, m in SOIL_MODELS.items():
        cats.setdefault(m['category'], []).append((k, m['label']))
    return cats
