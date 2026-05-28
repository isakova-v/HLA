#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PURE HLA-ONLY vaccine immunity predictor
=======================================
• No age
• No sex
• No region
• Prediction uses ONLY HLA genotype

Run:
    streamlit run vaccine_prediction_app.py

Artifacts required:
    ./artifacts/<VACCINE>_classifier.cbm
    ./artifacts/<VACCINE>_shap_classif.csv
    ./artifacts/meta.json
"""

import os, re, json
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier, CatBoostRegressor, Pool


ARTIFACT_DIR = "./artifacts"


# ===================== HLA helpers =====================
def canonical_pair(a1, a2):
    a1 = "" if pd.isna(a1) else str(a1).strip()
    a2 = "" if pd.isna(a2) else str(a2).strip()
    if a2 in ["", "-"]:
        a2 = a1
    if a1 == "":
        a1 = "NA"
    if a2 == "":
        a2 = "NA"
    p = sorted([a1, a2])
    return f"{p[0]}|{p[1]}"


def build_allele_catalog(hla_df):
    cat = {}
    for c in hla_df.columns:
        m = re.match(r"HLA-(.+)_1$", str(c))
        if not m:
            continue
        g = m.group(1)
        s = pd.concat([hla_df[f"HLA-{g}_1"], hla_df[f"HLA-{g}_2"]])
        s = s.dropna().astype(str).str.strip()
        s = s[~s.isin(["", "-", "NA"])]
        cat[g] = sorted(set(s.tolist()))
    return cat


# ===================== Cache loaders =====================
@st.cache_data
def load_meta():
    return json.load(open(f"{ARTIFACT_DIR}/meta.json"))

@st.cache_resource
def load_models(meta):
    models = {}
    for v in meta["TARGETS_CLASSIF"]:
        m = CatBoostClassifier()
        m.load_model(f"{ARTIFACT_DIR}/{v}_classifier.cbm")
        models[v] = m
    return models

@st.cache_data
def load_shap(meta):
    return {v: pd.read_csv(f"{ARTIFACT_DIR}/{v}_shap_classif.csv") for v in meta["TARGETS_CLASSIF"]}


# ===================== UI =====================
st.set_page_config(page_title="HLA Vaccine Predictor", layout="wide")
st.title("🧬 HLA-only Vaccine Immunity Predictor")

meta = load_meta()
models = load_models(meta)
shap = load_shap(meta)

up = st.sidebar.file_uploader("Upload combined_hla_out.xlsx", type=["xlsx"])
allele_catalog = build_allele_catalog(pd.read_excel(up)) if up else None

tab_pred, tab_shap = st.tabs(["Prediction", "SHAP"])

# ===================== Prediction =====================
with tab_pred:
    vac = st.selectbox("Vaccine", list(models.keys()))

    feats = shap[vac]["feature"].tolist()
    genes = [f.replace("geno_", "") for f in feats]

    chosen = st.multiselect("Genes to specify", genes, default=genes[:5])

    geno = {}
    for g in chosen:
        opts = allele_catalog[g] if allele_catalog and g in allele_catalog else []
        a1 = st.selectbox(f"{g} allele 1", ["(missing)"]+opts)
        a2 = st.selectbox(f"{g} allele 2", ["(missing)"]+opts)
        geno[f"geno_{g}"] = canonical_pair(a1 if a1!="(missing)" else "",
                                           a2 if a2!="(missing)" else "")

    X = pd.DataFrame([[geno.get(f,"NA|NA") if f!="age" else 0.0 for f in feats]], columns=feats)

    cat_idx = [i for i,f in enumerate(feats) if f!="age"]

    if st.button("Predict"):
        p = models[vac].predict_proba(Pool(X, cat_features=cat_idx))[:,1][0]
        st.metric("Responder probability", f"{p:.4f}")


# ===================== SHAP =====================
with tab_shap:
    vac2 = st.selectbox("Vaccine (SHAP)", list(models.keys()))
    imp = shap[vac2].head(20).iloc[::-1]
    fig = plt.figure(figsize=(7,6))
    plt.barh(imp["feature"], imp["mean_abs_shap"])
    plt.title(vac2)
    plt.tight_layout()
    st.pyplot(fig)