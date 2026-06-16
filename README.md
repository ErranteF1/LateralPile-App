# Lateral Pile Analysis — local app (LPile-like)

A transparent p-y / beam-column solver for laterally loaded piles and drilled shafts,
with a Streamlit user interface you run locally in Python.

## Install (one time)
```bash
# from this folder, ideally in a fresh virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

## Run
```bash
python -m streamlit run app.py
```
…then open the URL it prints (usually http://localhost:8501).
On Windows you can also just double-click **run_app.bat**; on macOS/Linux run **./run_app.sh**.

## What you can do
- Build a **layered soil/rock profile** (sand: Reese / API; clay: Matlock, Welch-Reese,
  Reese stiff-with-water; rock: weak rock, strong/vuggy-limestone).
- Define the **pile** (length, diameter, head fixity, tip condition) and **bending stiffness**
  as RC-circular (from f'c), steel (E·I), a direct EI value, or a **nonlinear bilinear M-phi**.
- Enter **multiple load cases** (shear V, moment M, axial P) in a table.
- **Run** and view interactive **y / M / V / p depth profiles**, a **summary table**
  (y_head, rotation, M_max, V_max and their depths, depth-to-fixity), and a **p-y curve viewer**.
- **Export** profiles and the summary to CSV, and **save / load** the whole project as JSON.

A worked example is in `examples/Example_sand_over_rock_96in.json` — load it from the sidebar.

## How sand stiffness is chosen
The sand initial modulus of subgrade reaction k is **auto-computed from the friction angle**
using the API(2011)/Reese k(φ) relationship — the same default LPile applies:
φ = 25 / 30 / 35 / 40° → 19.9 / 40.5 / 81 / 166 pci. You can override k per layer.

## Files
| File | Purpose |
|---|---|
| `app.py` | Streamlit user interface |
| `engine.py` | High-level project → solver driver (units, run, depth-to-fixity, save/load) |
| `defaults.py` | Unit conversions, theory k(φ), soil-model registry that drives the UI |
| `fem_solver.py` | Beam-on-Winkler FEM, secant p-y iteration, P-delta, variable EI (core) |
| `py_models.py` | The 14 p-y constitutive models (core) |
| `nonlinear_ei.py` | Bilinear M-phi nonlinear-stiffness solver (core) |
| `examples/` | Example project file |

## Validation & scope
Benchmarked against LPile for a **96-in drilled shaft in sand over strong (vuggy-limestone)
rock**, 40 load cases: maximum moment within ~1% and the depths of maximum moment/shear
within ~0.5 ft of LPile. Pile-head deflection and any configuration **outside that envelope**
(other diameters, soil models, head conditions, layer orders) are engineering estimates that
have not yet been independently validated — cross-check against LPile or a field load test.

**This tool does not replace the judgment of the engineer of record and is not for
unverified final/stamped design.**
