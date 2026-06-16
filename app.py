"""
app.py -- Lateral Pile Analysis (LPile-like) Streamlit application.
Run locally with:   streamlit run app.py

Core build: layered p-y profiles (sand / clay / rock), multi-load-case table,
constant or nonlinear (M-phi) bending stiffness, interactive y/M/V/p profiles,
p-y curve viewer, summary table, CSV/JSON export and project save/load.
"""
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import defaults as D
import engine as E
import py_models as M

st.set_page_config(page_title="Lateral Pile Analysis", layout="wide",
                   initial_sidebar_state="expanded")

if "project" not in st.session_state:
    st.session_state.project = E.blank_project()
if "results" not in st.session_state:
    st.session_state.results = None
P = st.session_state.project

TIP_OPTS = ["free", "pinned", "fixed"]

# =================================================================== SIDEBAR
with st.sidebar:
    st.title("Lateral Pile Analysis")
    st.caption("p-y method | beam-column FEM | US units (ft, in, kip)")

    P["name"] = st.text_input("Project name", P.get("name", "New project"))

    st.header("Pile geometry")
    c1, c2 = st.columns(2)
    P["pile"]["L_ft"]  = c1.number_input("Length L (ft)", 1.0, 500.0, float(P["pile"]["L_ft"]), 1.0)
    P["pile"]["D_in"]  = c2.number_input("Diameter D (in)", 1.0, 240.0, float(P["pile"]["D_in"]), 1.0)
    P["pile"]["n_elements"] = c1.number_input("FEM elements", 20, 400, int(P["pile"]["n_elements"]), 10)
    P["pile"]["head_fixed"] = c2.selectbox("Head condition", [False, True],
        index=int(P["pile"]["head_fixed"]), format_func=lambda b: "Fixed" if b else "Free")
    P["pile"]["tip_cond"] = c1.selectbox("Tip condition", TIP_OPTS, index=TIP_OPTS.index(P["pile"]["tip_cond"]))

    st.header("Bending stiffness EI")
    modes = {"rc": "RC circular (from f'c)", "steel": "Steel (E.I)",
             "direct": "Direct EI", "mphi": "Nonlinear M-phi (bilinear)"}
    mode = st.selectbox("Stiffness model", list(modes.keys()),
                        index=list(modes.keys()).index(P["pile"].get("ei_mode", "rc")),
                        format_func=lambda k: modes[k])
    P["pile"]["ei_mode"] = mode
    if mode == "rc":
        P["pile"]["fc_psi"] = st.number_input("Concrete f'c (psi)", 2000.0, 12000.0,
                                              float(P["pile"]["fc_psi"]), 250.0)
        EI = D.EI_rc_circular(P["pile"]["D_in"], P["pile"]["fc_psi"])
    elif mode == "steel":
        cc1, cc2 = st.columns(2)
        P["pile"]["E_ksi"] = cc1.number_input("E (ksi)", 1000.0, 50000.0, float(P["pile"]["E_ksi"]), 500.0)
        P["pile"]["I_in4"] = cc2.number_input("I (in^4)", 1.0, 5.0e5, float(P["pile"]["I_in4"]), 10.0)
        EI = D.EI_from_EI(P["pile"]["E_ksi"], P["pile"]["I_in4"])
    elif mode == "direct":
        P["pile"]["EI_kip_in2"] = st.number_input("EI (kip-in2)", 1.0e5, 1.0e12,
                                                  float(P["pile"]["EI_kip_in2"]), format="%.3e")
        EI = P["pile"]["EI_kip_in2"]
    else:
        cc1, cc2 = st.columns(2)
        P["pile"]["EI_kip_in2"] = cc1.number_input("Elastic EI (kip-in2)", 1.0e5, 1.0e12,
                                                   float(P["pile"]["EI_kip_in2"]), format="%.3e")
        P["pile"]["My_kip_in"] = cc2.number_input("Yield moment My (kip-in)", 1.0e3, 1.0e7,
                                                  float(P["pile"]["My_kip_in"]), format="%.3e")
        P["pile"]["alpha"] = st.slider("Post-yield stiffness ratio alpha", 0.0, 0.5,
                                       float(P["pile"].get("alpha", 0.05)), 0.01)
        EI = P["pile"]["EI_kip_in2"]
    st.metric("Effective EI (kip-in2)", f"{EI:.3e}")

    st.divider()
    st.subheader("Project file")
    up = st.file_uploader("Load project (.json)", type=["json"])
    if up is not None:
        try:
            st.session_state.project = json.load(up); st.session_state.results = None
            st.success("Project loaded."); st.rerun()
        except Exception as ex:
            st.error(f"Could not read file: {ex}")
    st.download_button("Save project (.json)",
                       data=json.dumps(P, indent=2), file_name=f"{P['name']}.json",
                       mime="application/json", use_container_width=True)

# =================================================================== MAIN
st.title("Lateral Pile Analysis")
st.caption("Validated against LPile for a 96-in drilled shaft in sand-over-strong-rock. "
           "Results outside that envelope are engineering estimates - verify against LPile or a load test.")

tab_soil, tab_loads, tab_run, tab_py, tab_about = st.tabs(
    ["Soil profile", "Loads", "Run & results", "p-y curves", "About"])

# -------------------------------------------------------------- SOIL PROFILE
with tab_soil:
    st.subheader("Layered soil / rock profile")
    st.caption("Depths in feet below the pile head (ground surface = pile top).")
    cats = D.models_by_category()
    model_keys = [k for c in cats for (k, _) in cats[c]]
    model_label = {k: D.SOIL_MODELS[k]["label"] for k in model_keys}

    remove_idx = None
    for i, L in enumerate(P["layers"]):
        with st.expander(f"Layer {i+1}:  {model_label.get(L['model'], L['model'])}  "
                         f"({L['z_top_ft']:.0f}-{L['z_bot_ft']:.0f} ft)", expanded=(len(P['layers'])<=3)):
            a, b, c = st.columns(3)
            L["z_top_ft"] = a.number_input("Top (ft)", 0.0, 500.0, float(L["z_top_ft"]), 1.0, key=f"zt{i}")
            L["z_bot_ft"] = b.number_input("Bottom (ft)", 0.0, 600.0, float(L["z_bot_ft"]), 1.0, key=f"zb{i}")
            new_model = c.selectbox("p-y model", model_keys, index=model_keys.index(L["model"]),
                                    format_func=lambda k: model_label[k], key=f"mdl{i}")
            if new_model != L["model"]:
                L["model"] = new_model
                L["fields"] = {fk: dv for (fk, _lbl, _u, dv) in D.SOIL_MODELS[new_model]["fields"]}
                st.rerun()
            spec = D.SOIL_MODELS[L["model"]]
            if spec["category"] in ("Sand", "Clay"):
                L["loading"] = st.selectbox("Loading", D.LOADING_OPTIONS,
                    index=D.LOADING_OPTIONS.index(L.get("loading", "static")), key=f"ld{i}")
            fcols = st.columns(max(1, len(spec["fields"])))
            for j, (fk, lbl, unit, dv) in enumerate(spec["fields"]):
                cur = float(L.get("fields", {}).get(fk, dv))
                L.setdefault("fields", {})[fk] = fcols[j].number_input(
                    f"{lbl} [{unit}]", value=cur, key=f"f{i}_{fk}", format="%.5g")
            if L["model"] in ("reese_sand", "api_sand"):
                kk = L["fields"].get("k_pci", 0.0)
                shown = kk if kk > 0 else D.default_k_sand_pci(L["fields"]["phi_deg"])
                st.caption(f"Initial subgrade modulus used: k = {shown:.1f} pci "
                           f"({'user' if kk>0 else 'auto from phi, API/LPile default'}).")
            if st.button("Remove layer", key=f"rm{i}"):
                remove_idx = i
    if remove_idx is not None and len(P["layers"]) > 1:
        P["layers"].pop(remove_idx); st.rerun()

    if st.button("Add layer"):
        z0 = P["layers"][-1]["z_bot_ft"] if P["layers"] else 0.0
        P["layers"].append(dict(z_top_ft=z0, z_bot_ft=z0+10.0, model="reese_sand",
                                loading="static",
                                fields={fk: dv for (fk,_l,_u,dv) in D.SOIL_MODELS["reese_sand"]["fields"]}))
        st.rerun()

    fig = go.Figure()
    palette = {"Sand": "#E3C16F", "Clay": "#9DB17C", "Rock": "#9A8478"}
    for L in P["layers"]:
        cat = D.SOIL_MODELS[L["model"]]["category"]
        fig.add_shape(type="rect", x0=0, x1=1, y0=L["z_top_ft"], y1=L["z_bot_ft"],
                      fillcolor=palette.get(cat, "#ccc"), line=dict(color="white"))
        fig.add_annotation(x=0.5, y=(L["z_top_ft"]+L["z_bot_ft"])/2,
                           text=D.SOIL_MODELS[L["model"]]["label"], showarrow=False, font=dict(size=11))
    fig.add_shape(type="line", x0=0.5, x1=0.5, y0=0, y1=P["pile"]["L_ft"],
                  line=dict(color="black", width=4))
    fig.update_yaxes(autorange="reversed", title="Depth (ft)")
    fig.update_xaxes(visible=False, range=[0, 1])
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=20, b=10), showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------------------------- LOADS
with tab_loads:
    st.subheader("Pile-head load cases")
    st.caption("V = lateral shear at head (kip), M = applied moment (kip-ft), P = axial thrust (kip).")
    df = pd.DataFrame(P["loads"])
    df = st.data_editor(df, num_rows="dynamic", use_container_width=True,
                        column_config={
                            "name": st.column_config.TextColumn("Case"),
                            "V_kip": st.column_config.NumberColumn("V (kip)", format="%.1f"),
                            "M_kipft": st.column_config.NumberColumn("M (kip-ft)", format="%.1f"),
                            "P_kip": st.column_config.NumberColumn("P axial (kip)", format="%.1f")})
    P["loads"] = df.fillna(0).to_dict("records")

# ----------------------------------------------------------------- RUN/RESULTS
with tab_run:
    st.subheader("Run analysis")
    if st.button("Run", type="primary"):
        try:
            with st.spinner("Solving p-y system..."):
                res, pp, soil = E.run(P)
            st.session_state.results = res
            st.success(f"Done - {len(res)} load case(s).")
        except Exception as ex:
            st.session_state.results = None
            st.error(f"Solver error: {ex}")

    res = st.session_state.results
    if res:
        rows = [E.summary_row(r) for r in res]
        sdf = pd.DataFrame(rows)
        st.markdown("**Summary of pile-head responses**")
        st.dataframe(sdf, use_container_width=True, hide_index=True)
        st.download_button("Download summary (CSV)", sdf.to_csv(index=False),
                           file_name=f"{P['name']}_summary.csv", mime="text/csv")

        names = [r.get("name", f"LC{i+1}") or f"LC{i+1}" for i, r in enumerate(res)]
        sel = st.selectbox("Show profiles for case", range(len(res)), format_func=lambda i: names[i])
        r = res[sel]
        z = np.asarray(r["z_in"]) / D.FT2IN
        Mft = np.asarray(r["M_kip_in"]) / D.FT2IN
        fig = make_subplots(rows=1, cols=4, shared_yaxes=True,
                            subplot_titles=("Deflection y (in)", "Moment M (kip-ft)",
                                            "Shear V (kip)", "Soil reaction p (kip/in)"))
        fig.add_trace(go.Scatter(x=r["y_in"], y=z, mode="lines", line=dict(color="#1f77b4")), 1, 1)
        fig.add_trace(go.Scatter(x=Mft, y=z, mode="lines", line=dict(color="#C00000")), 1, 2)
        fig.add_trace(go.Scatter(x=r["V_kip"], y=z, mode="lines", line=dict(color="#2ca02c")), 1, 3)
        fig.add_trace(go.Scatter(x=r["p_kip_in"], y=z, mode="lines", line=dict(color="#7f4fc9")), 1, 4)
        for L in P["layers"]:
            for zz in (L["z_top_ft"], L["z_bot_ft"]):
                for c in range(1, 5):
                    fig.add_hline(y=zz, line=dict(color="rgba(120,90,60,0.25)", width=1), row=1, col=c)
        fig.update_yaxes(autorange="reversed", title="Depth (ft)", row=1, col=1)
        fig.update_layout(height=620, showlegend=False, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)

        prof = pd.DataFrame({"depth_ft": z, "y_in": r["y_in"], "M_kipft": Mft,
                             "V_kip": r["V_kip"], "p_kip_in": r["p_kip_in"],
                             "p_over_pult": r["p_over_pult"]})
        st.download_button("Download this profile (CSV)", prof.to_csv(index=False),
                           file_name=f"{P['name']}_{names[sel]}_profile.csv", mime="text/csv")
        if not r.get("converged", True):
            st.warning("This case did not fully converge - interpret with caution.")
    else:
        st.info("Set up the soil profile and loads, then press Run.")

# ------------------------------------------------------------------ P-Y CURVES
with tab_py:
    st.subheader("p-y curve viewer")
    li = st.selectbox("Layer", range(len(P["layers"])),
                      format_func=lambda i: f"Layer {i+1}: {D.SOIL_MODELS[P['layers'][i]['model']]['label']}")
    L = P["layers"][li]
    zc, _ = st.columns([1, 2])
    z_ft = zc.slider("Depth (ft)", float(L["z_top_ft"]), float(max(L["z_bot_ft"], L["z_top_ft"]+1)),
                     float((L["z_top_ft"]+L["z_bot_ft"]) / 2))
    try:
        if L["model"] == "strong_rock":
            f = L["fields"]
            su = np.array([[L["z_top_ft"]*D.FT2IN, f["ucs_top_psi"]*D.PSI],
                           [L["z_bot_ft"]*D.FT2IN, f["ucs_bot_psi"]*D.PSI]])
            params = dict(su_ksi=su, b_in=P["pile"]["D_in"])
        else:
            params = D.SOIL_MODELS[L["model"]]["build"](L.get("fields", {}), P["pile"]["D_in"],
                                                        L["z_top_ft"]*D.FT2IN, L.get("loading", "static"))
        yv, pv = M.get_py_curve(z_ft*D.FT2IN, params, L["model"])
        figpy = go.Figure(go.Scatter(x=yv, y=pv, mode="lines", line=dict(color="#1f77b4", width=2)))
        figpy.update_layout(height=440, xaxis_title="y (in)", yaxis_title="p (kip/in)",
                            margin=dict(l=10, r=10, t=20, b=10),
                            title=f"{D.SOIL_MODELS[L['model']]['label']} @ {z_ft:.1f} ft")
        st.plotly_chart(figpy, use_container_width=True)
    except Exception as ex:
        st.error(f"Could not build p-y curve: {ex}")

# ---------------------------------------------------------------------- ABOUT
with tab_about:
    st.markdown("""
### About this tool
A transparent p-y / beam-column solver for laterally loaded piles and drilled shafts.

**Method.** Hermitian beam-on-Winkler FEM with secant p-y iteration and optional
P-delta and nonlinear (bilinear M-phi) bending stiffness. 14 p-y constitutive models
are implemented; the **core build** exposes the validated set (sand, clay, weak/strong rock).

**Sand subgrade modulus** is auto-computed from friction angle using the API(2011)/Reese
k(phi) relationship (the same default LPile uses): phi=25/30/35/40 deg -> 19.9/40.5/81/166 pci.

**Validation.** Reproduced LPile for a 96-in drilled shaft, sand-over-strong-rock, 40 load
cases: M_max ~ 1%, depth of max moment/shear < 1 ft. Deflection and out-of-envelope
configurations are estimates - cross-check against LPile or a field load test before
relying on them for final design. Not a substitute for the engineer of record's judgment.
""")
