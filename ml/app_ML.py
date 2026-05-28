import os
import re
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from catboost import CatBoostClassifier, CatBoostRegressor, Pool

# Optional extras
HAS_SKLEARN = True
HAS_XGB = True
try:
    from sklearn.model_selection import StratifiedKFold, KFold
    from sklearn.metrics import (
        roc_auc_score, average_precision_score, f1_score,
        balanced_accuracy_score, accuracy_score
    )
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LogisticRegression
except Exception:
    HAS_SKLEARN = False

try:
    import xgboost as xgb
except Exception:
    HAS_XGB = False


# ---------------- Constants ----------------
ID_COL_VACC = "ZLIMS ID"
ID_COL_HLA = "sample_id"

TARGETS_CLASSIF = {
    "HBV": "HBV_NoAnswer_coef",
    "diphtheria": "diphtheria_NoAnswer_coef",
    "measles": "measles_NoAnswer_coef",
    "mumps": "mumps_NoAnswer_coef",
    "rubella": "rubella_NoAnswer_coef",
}

DEFAULT_FILES = {
    "HBV": "HBV.xlsx",
    "diphtheria": "diphtheria.xlsx",
    "measles": "measles.xlsx",
    "rubella": "rubella.xlsx",
    "mumps": "mumps.xlsx",
    "hla": "combined_hla_out.xlsx",
}

# Regression: you said mumps has no titre.
REGRESSION_VACCINES = ["HBV", "diphtheria", "measles", "rubella"]

# Optional: try to auto-detect titre column candidates
TITRE_CANDIDATE_PATTERNS = [
    r"titre", r"titer", r"titer_", r"antibody", r"igg", r"ig_g", r"concentration"
]


# ---------------- Utilities ----------------
def read_xlsx(upload, fallback_path: str) -> pd.DataFrame:
    return pd.read_excel(upload) if upload is not None else pd.read_excel(fallback_path)


def canonical_pair(a1, a2) -> str:
    a1 = "" if pd.isna(a1) else str(a1).strip()
    a2 = "" if pd.isna(a2) else str(a2).strip()
    # homozygote: *_2 == "-" -> copy allele 1
    if a2 in ["-", ""]:
        a2 = a1
    if a1 == "":
        a1 = "NA"
    if a2 == "":
        a2 = "NA"
    p = sorted([a1, a2])
    return f"{p[0]}|{p[1]}"


def build_genotype_features(hla_df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({ID_COL_HLA: hla_df[ID_COL_HLA].astype(str)})
    for c in hla_df.columns:
        m = re.match(r"HLA-(.+)_1$", str(c))
        if not m:
            continue
        g = m.group(1)
        c1 = f"HLA-{g}_1"
        c2 = f"HLA-{g}_2"
        if c2 not in hla_df.columns:
            continue
        out[f"geno_{g}"] = [
            canonical_pair(a1, a2)
            for a1, a2 in zip(hla_df[c1].tolist(), hla_df[c2].tolist())
        ]
    return out


def _prepare_merged(vdf: pd.DataFrame, hla_geno: pd.DataFrame) -> pd.DataFrame:
    v = vdf.copy()
    v[ID_COL_VACC] = v[ID_COL_VACC].astype(str)

    h = hla_geno.copy()
    h[ID_COL_HLA] = h[ID_COL_HLA].astype(str)

    merged = v.merge(h, left_on=ID_COL_VACC, right_on=ID_COL_HLA, how="inner")
    return merged


def make_dataset_classif(vdf: pd.DataFrame, hla_geno: pd.DataFrame, target_col: str, use_age: bool = True):
    merged = _prepare_merged(vdf, hla_geno)
    merged = merged[~merged[target_col].isna()].copy()

    # responder=1 if NoAnswer_coef == 0, else 0
    y = (merged[target_col].astype(float).values == 0.0).astype(int)

    geno_cols = [c for c in merged.columns if c.startswith("geno_")]
    base = ["sex", "region"]
    if use_age:
        base = ["age"] + base

    X = merged[base + geno_cols].copy()

    if use_age and "age" in X.columns:
        X["age"] = pd.to_numeric(X["age"], errors="coerce")

    # all categorical except age
    for c in [c for c in X.columns if c != "age"]:
        X[c] = X[c].astype("string")

    return X, y, merged


def make_dataset_regress(vdf: pd.DataFrame, hla_geno: pd.DataFrame, titre_col: str, use_age: bool = True):
    merged = _prepare_merged(vdf, hla_geno)
    if titre_col is None or titre_col not in merged.columns:
        return None, None, merged

    # keep non-missing titre
    merged = merged[~merged[titre_col].isna()].copy()

    y = pd.to_numeric(merged[titre_col], errors="coerce").values
    mask = ~np.isnan(y)
    merged = merged.loc[mask].copy()
    y = y[mask]

    geno_cols = [c for c in merged.columns if c.startswith("geno_")]
    base = ["sex", "region"]
    if use_age:
        base = ["age"] + base

    X = merged[base + geno_cols].copy()

    if use_age and "age" in X.columns:
        X["age"] = pd.to_numeric(X["age"], errors="coerce")

    for c in [c for c in X.columns if c != "age"]:
        X[c] = X[c].astype("string")

    return X, y, merged


# ---------------- Metrics ----------------
def compute_metrics_classif(y_true, proba):
    pred = (proba >= 0.5).astype(int)
    return {
        "ROC_AUC": roc_auc_score(y_true, proba) if len(np.unique(y_true)) == 2 else np.nan,
        "PR_AUC": average_precision_score(y_true, proba),
        "F1": f1_score(y_true, pred, zero_division=0),
        "BalancedAcc": balanced_accuracy_score(y_true, pred),
        "Accuracy": accuracy_score(y_true, pred),
    }


def compute_metrics_regress(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_pred - y_true
    mse = float(np.mean(err ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(err)))
    # R2 (safe)
    denom = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - (np.sum((y_true - y_pred) ** 2) / denom)) if denom > 0 else np.nan
    return {"RMSE": rmse, "MAE": mae, "R2": r2}


def mean_std_strings(df: pd.DataFrame):
    m, s = df.mean(numeric_only=True), df.std(numeric_only=True)
    out = {}
    for col in m.index:
        out[col] = f"{m[col]:.3f} ± {s[col]:.3f}"
    return out


# ---------------- Baselines (optional) ----------------
def lr_pipeline(X: pd.DataFrame):
    if not HAS_SKLEARN:
        return None

    num_cols = ["age"] if "age" in X.columns else []
    cat_cols = [c for c in X.columns if c not in num_cols]

    transformers = []
    if num_cols:
        transformers.append(("num", Pipeline([("imp", SimpleImputer(strategy="median"))]), num_cols))
    if cat_cols:
        transformers.append(("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=True))
        ]), cat_cols))

    pre = ColumnTransformer(transformers, remainder="drop")
    model = LogisticRegression(max_iter=3000, class_weight="balanced")
    return Pipeline([("pre", pre), ("model", model)])


# ---------------- CV evaluators ----------------
def cv_eval_catboost_classif(X, y, folds=5, seed=42, iters=800):
    if not HAS_SKLEARN:
        st.warning("scikit-learn not available: CV splitting might fail. Install scikit-learn.")
        return pd.DataFrame()

    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    cat_idx = [i for i, c in enumerate(X.columns) if c != "age"]

    all_m = []
    for tr, te in skf.split(X, y):
        Xtr, Xte = X.iloc[tr], X.iloc[te]
        ytr, yte = y[tr], y[te]

        model = CatBoostClassifier(
            iterations=int(iters),
            depth=6,
            learning_rate=0.05,
            loss_function="Logloss",
            eval_metric="AUC",
            auto_class_weights="Balanced",
            random_seed=int(seed),
            verbose=False,
        )
        model.fit(Pool(Xtr, ytr, cat_features=cat_idx))
        proba = model.predict_proba(Pool(Xte, yte, cat_features=cat_idx))[:, 1]
        all_m.append(compute_metrics_classif(yte, proba))

    return pd.DataFrame(all_m)


def cv_eval_catboost_regress(X, y, folds=5, seed=42, iters=800):
    if not HAS_SKLEARN:
        st.warning("scikit-learn not available: CV splitting might fail. Install scikit-learn.")
        return pd.DataFrame()

    kf = KFold(n_splits=folds, shuffle=True, random_state=seed)
    cat_idx = [i for i, c in enumerate(X.columns) if c != "age"]

    all_m = []
    for tr, te in kf.split(X):
        Xtr, Xte = X.iloc[tr], X.iloc[te]
        ytr, yte = y[tr], y[te]

        model = CatBoostRegressor(
            iterations=int(iters),
            depth=6,
            learning_rate=0.05,
            loss_function="RMSE",
            eval_metric="RMSE",
            random_seed=int(seed),
            verbose=False,
        )
        model.fit(Pool(Xtr, ytr, cat_features=cat_idx))
        pred = model.predict(Pool(Xte, cat_features=cat_idx))
        all_m.append(compute_metrics_regress(yte, pred))

    return pd.DataFrame(all_m)


def cv_eval_logreg(X, y, folds=5, seed=42):
    if not HAS_SKLEARN:
        return None

    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    all_m = []
    pipe = lr_pipeline(X)
    for tr, te in skf.split(X, y):
        Xtr, Xte = X.iloc[tr], X.iloc[te]
        ytr, yte = y[tr], y[te]
        pipe.fit(Xtr, ytr)
        proba = pipe.predict_proba(Xte)[:, 1]
        all_m.append(compute_metrics_classif(yte, proba))
    return pd.DataFrame(all_m)


def cv_eval_xgb(X, y, folds=5, seed=42):
    if (not HAS_XGB) or (not HAS_SKLEARN):
        return None

    pipe = lr_pipeline(X)
    pre = pipe.named_steps["pre"]

    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    all_m = []
    for tr, te in skf.split(X, y):
        Xtr, Xte = X.iloc[tr], X.iloc[te]
        ytr, yte = y[tr], y[te]

        Xtr_enc = pre.fit_transform(Xtr)
        Xte_enc = pre.transform(Xte)

        model = xgb.XGBClassifier(
            n_estimators=800,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            objective="binary:logistic",
            eval_metric="auc",
            random_state=int(seed),
            n_jobs=-1,
        )
        model.fit(Xtr_enc, ytr)
        proba = model.predict_proba(Xte_enc)[:, 1]
        all_m.append(compute_metrics_classif(yte, proba))
    return pd.DataFrame(all_m)


# ---------------- Full-train model builders (for interactive prediction) ----------------
def train_full_classifier(model_name: str, X: pd.DataFrame, y: np.ndarray, seed: int, cb_iters: int):
    if model_name == "CatBoost":
        cat_idx = [i for i, c in enumerate(X.columns) if c != "age"]
        model = CatBoostClassifier(
            iterations=int(cb_iters),
            depth=6,
            learning_rate=0.05,
            loss_function="Logloss",
            eval_metric="AUC",
            auto_class_weights="Balanced",
            random_seed=int(seed),
            verbose=False,
        )
        model.fit(Pool(X, y, cat_features=cat_idx))

        def predict_proba(X_new: pd.DataFrame):
            return model.predict_proba(Pool(X_new, cat_features=cat_idx))[:, 1]

        return model, predict_proba

    if model_name == "LogReg + OneHot":
        if not HAS_SKLEARN:
            return None, None
        pipe = lr_pipeline(X)
        pipe.fit(X, y)

        def predict_proba(X_new: pd.DataFrame):
            return pipe.predict_proba(X_new)[:, 1]

        return pipe, predict_proba

    if model_name == "XGBoost + OneHot":
        if (not HAS_XGB) or (not HAS_SKLEARN):
            return None, None
        pipe = lr_pipeline(X)
        pre = pipe.named_steps["pre"]

        X_enc = pre.fit_transform(X)
        model = xgb.XGBClassifier(
            n_estimators=800,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            objective="binary:logistic",
            eval_metric="auc",
            random_state=int(seed),
            n_jobs=-1,
        )
        model.fit(X_enc, y)

        def predict_proba(X_new: pd.DataFrame):
            X_new_enc = pre.transform(X_new)
            return model.predict_proba(X_new_enc)[:, 1]

        return (pre, model), predict_proba

    return None, None


def train_full_regressor(X: pd.DataFrame, y: np.ndarray, seed: int, cb_iters: int):
    cat_idx = [i for i, c in enumerate(X.columns) if c != "age"]
    model = CatBoostRegressor(
        iterations=int(cb_iters),
        depth=6,
        learning_rate=0.05,
        loss_function="RMSE",
        eval_metric="RMSE",
        random_seed=int(seed),
        verbose=False,
    )
    model.fit(Pool(X, y, cat_features=cat_idx))

    def predict(X_new: pd.DataFrame):
        return model.predict(Pool(X_new, cat_features=cat_idx))

    return model, predict


# ---------------- SHAP helper (CatBoost built-in) ----------------
def catboost_mean_abs_shap_any(model, X: pd.DataFrame, y: np.ndarray, max_rows=300, seed=42):
    rng = np.random.default_rng(int(seed))
    idx = rng.choice(len(X), size=min(int(max_rows), len(X)), replace=False)
    Xs = X.iloc[idx].reset_index(drop=True)
    ys = y[idx]

    cat_idx = [i for i, c in enumerate(X.columns) if c != "age"]
    shap_vals = model.get_feature_importance(Pool(Xs, ys, cat_features=cat_idx), type="ShapValues")
    sv = shap_vals[:, :-1]  # last col = expected value
    mean_abs = np.mean(np.abs(sv), axis=0)

    imp = pd.DataFrame({"feature": X.columns, "mean_abs_shap": mean_abs})
    return imp.sort_values("mean_abs_shap", ascending=False)


# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="Vaccine models: quality + SHAP + prediction", layout="wide")
st.title("Vaccine response: per-vaccine tabs + summary + SHAP + (optional) titre regression")

with st.sidebar:
    st.header("Upload (or use local defaults)")
    up_hla = st.file_uploader("combined_hla_out.xlsx", type=["xlsx"])
    up_hbv = st.file_uploader("HBV.xlsx", type=["xlsx"])
    up_diph = st.file_uploader("diphtheria.xlsx", type=["xlsx"])
    up_mea = st.file_uploader("measles.xlsx", type=["xlsx"])
    up_rub = st.file_uploader("rubella.xlsx", type=["xlsx"])
    up_mum = st.file_uploader("mumps.xlsx", type=["xlsx"])

    st.divider()
    use_age = st.checkbox("Use age as feature", value=True)
    folds = st.slider("CV folds", 2, 10, 5, 1)
    seed = st.number_input("Seed", value=42, step=1)
    cb_iters = st.slider("CatBoost iterations", 200, 2000, 800, 100)

    st.divider()
    shap_rows = st.slider("SHAP sample rows", 50, 1500, 300, 50)

    st.divider()
    run_btn = st.button("Run evaluation", type="primary")

st.subheader("Environment")
st.write({"sklearn_available": HAS_SKLEARN, "xgboost_available": HAS_XGB})
if not HAS_SKLEARN:
    st.info("Install scikit-learn to enable CV splitting + LogisticRegression/XGBoost baselines: `pip install scikit-learn`.")
if HAS_SKLEARN and not HAS_XGB:
    st.info("Install xgboost to enable XGBoost baseline: `pip install xgboost`.")


@st.cache_data(show_spinner=True)
def load_data():
    hla_df = read_xlsx(up_hla, DEFAULT_FILES["hla"])
    hla_geno = build_genotype_features(hla_df)

    vdfs = {
        "HBV": read_xlsx(up_hbv, DEFAULT_FILES["HBV"]),
        "diphtheria": read_xlsx(up_diph, DEFAULT_FILES["diphtheria"]),
        "measles": read_xlsx(up_mea, DEFAULT_FILES["measles"]),
        "rubella": read_xlsx(up_rub, DEFAULT_FILES["rubella"]),
        "mumps": read_xlsx(up_mum, DEFAULT_FILES["mumps"]),
    }
    return vdfs, hla_geno


def suggest_titre_columns(df: pd.DataFrame):
    cols = list(df.columns)
    sugg = []
    for c in cols:
        c_low = str(c).lower()
        if any(re.search(pat, c_low) for pat in TITRE_CANDIDATE_PATTERNS):
            sugg.append(c)
    # Keep unique order
    seen = set()
    out = []
    for c in sugg:
        if c not in seen:
            out.append(c)
            seen.add(c)
    return out


vdfs, hla_geno = load_data()

# Let user map titre columns per vaccine (except mumps) in the sidebar-like top area.
st.subheader("Titre columns (for regression)")
st.caption("Pick which column contains antibody titre for each vaccine. Mumps is excluded (no titre).")
titre_map = {}
cols_box = st.columns(4)
for i, vac in enumerate(REGRESSION_VACCINES):
    df = vdfs[vac]
    candidates = suggest_titre_columns(df)
    options = ["(none)"] + candidates + [c for c in df.columns if c not in candidates]
    default_idx = 0
    # if there are candidates, choose first candidate by default
    if len(candidates) > 0:
        default_idx = options.index(candidates[0])
    sel = cols_box[i % 4].selectbox(f"{vac} titre column", options, index=default_idx, key=f"titre_{vac}")
    titre_map[vac] = None if sel == "(none)" else sel


# Session state caches
if "cv_summary" not in st.session_state:
    st.session_state.cv_summary = None
if "per_vac" not in st.session_state:
    st.session_state.per_vac = {}
if "shap_imp" not in st.session_state:
    st.session_state.shap_imp = {}


def run_all_evaluations():
    per_vac = {}
    rows = []

    for vac in TARGETS_CLASSIF.keys():
        target_col = TARGETS_CLASSIF[vac]

        # ---- Classification dataset ----
        Xc, yc, merged = make_dataset_classif(vdfs[vac], hla_geno, target_col, use_age=bool(use_age))
        base_c = {"vaccine": vac, "task": "classification", "n": int(len(Xc)), "rate_or_mean": float(yc.mean())}

        cb_c = cv_eval_catboost_classif(Xc, yc, folds=int(folds), seed=int(seed), iters=int(cb_iters))
        rows.append({**base_c, "model": "CatBoost", **mean_std_strings(cb_c)})

        lr_c = cv_eval_logreg(Xc, yc, folds=int(folds), seed=int(seed))
        if lr_c is not None:
            rows.append({**base_c, "model": "LogReg + OneHot", **mean_std_strings(lr_c)})

        xg_c = cv_eval_xgb(Xc, yc, folds=int(folds), seed=int(seed))
        if xg_c is not None:
            rows.append({**base_c, "model": "XGBoost + OneHot", **mean_std_strings(xg_c)})

        # ---- Regression dataset (CatBoostRegressor only), skip mumps and only if titre selected ----
        titre_col = titre_map.get(vac) if vac in REGRESSION_VACCINES else None
        Xr, yr, _ = make_dataset_regress(vdfs[vac], hla_geno, titre_col, use_age=bool(use_age))

        cb_r = None
        if (vac in REGRESSION_VACCINES) and (titre_col is not None) and (Xr is not None) and (len(Xr) >= 5):
            base_r = {"vaccine": vac, "task": "regression", "n": int(len(Xr)), "rate_or_mean": float(np.mean(yr))}
            cb_r = cv_eval_catboost_regress(Xr, yr, folds=int(folds), seed=int(seed), iters=int(cb_iters))
            rows.append({**base_r, "model": "CatBoostRegressor", **mean_std_strings(cb_r)})

        per_vac[vac] = {
            "classif": {"X": Xc, "y": yc, "cv": {"CatBoost": cb_c, "LogReg + OneHot": lr_c, "XGBoost + OneHot": xg_c}},
            "regress": {"titre_col": titre_col, "X": Xr, "y": yr, "cv": {"CatBoostRegressor": cb_r}},
        }

    summary = pd.DataFrame(rows).sort_values(["task", "vaccine", "model"]).reset_index(drop=True)
    st.session_state.per_vac = per_vac
    st.session_state.cv_summary = summary


if run_btn:
    run_all_evaluations()

tabs = st.tabs(["Summary"] + list(TARGETS_CLASSIF.keys()))


# ---------------- Summary tab ----------------
with tabs[0]:
    st.header("Summary")
    if st.session_state.cv_summary is None:
        st.info("Click **Run evaluation** in the sidebar.")
    else:
        st.subheader("CV summary")
        st.caption("`rate_or_mean` = responder_rate for classification, mean(titre) for regression.")
        st.dataframe(st.session_state.cv_summary, use_container_width=True)

        st.divider()
        st.subheader("Interactive prediction")

        vac = st.selectbox("Vaccine", list(TARGETS_CLASSIF.keys()), index=0)

        # Which tasks are available?
        per = st.session_state.per_vac[vac]
        has_reg = (vac in REGRESSION_VACCINES) and (per["regress"]["titre_col"] is not None) and (per["regress"]["X"] is not None)

        task = st.radio(
            "Task",
            ["classification (response probability)"] + (["regression (predicted titre)"] if has_reg else []),
            horizontal=True
        )

        # Choose models
        if task.startswith("classification"):
            available_models = ["CatBoost"]
            if HAS_SKLEARN:
                available_models.append("LogReg + OneHot")
            if HAS_SKLEARN and HAS_XGB:
                available_models.append("XGBoost + OneHot")
            chosen_models = st.multiselect("Models to use", available_models, default=["CatBoost"])
            Xref = per["classif"]["X"]
        else:
            chosen_models = ["CatBoostRegressor"]
            Xref = per["regress"]["X"]

        geno_cols = [c for c in Xref.columns if c.startswith("geno_")]
        genes = [c.replace("geno_", "") for c in geno_cols]
        chosen_genes = st.multiselect(
            "Genes to specify (geno_*)",
            genes,
            default=genes[:3] if len(genes) >= 3 else genes
        )

        with st.form("predict_form"):
            cols = st.columns(3)
            if "age" in Xref.columns:
                age = cols[0].number_input("age", value=30.0, step=1.0)
            else:
                age = None
            sex = cols[1].text_input("sex", value="F")
            region = cols[2].text_input("region", value="NA")

            st.markdown("**Alleles (enter allele1 and allele2; use '-' or empty for missing/homozygote)**")
            geno_values = {}
            for g in chosen_genes:
                a1 = st.text_input(f"HLA-{g} allele 1", value="", key=f"pred_{vac}_{task}_{g}_a1")
                a2 = st.text_input(f"HLA-{g} allele 2", value="", key=f"pred_{vac}_{task}_{g}_a2")
                geno_values[f"geno_{g}"] = canonical_pair(a1, a2)

            submitted = st.form_submit_button("Predict", type="primary")

        if submitted:
            # Build single-row with expected columns
            row = {}
            if "age" in Xref.columns:
                row["age"] = age
            row["sex"] = sex
            row["region"] = region
            for c in geno_cols:
                row[c] = geno_values.get(c, "NA|NA")
            X_new = pd.DataFrame([row], columns=Xref.columns)

            if task.startswith("classification"):
                results = []
                y = per["classif"]["y"]
                X = per["classif"]["X"]

                for mn in chosen_models:
                    model_obj, pred_fn = train_full_classifier(mn, X, y, seed=int(seed), cb_iters=int(cb_iters))
                    if pred_fn is None:
                        continue
                    p = float(pred_fn(X_new)[0])
                    results.append({"model": mn, "proba_responder": p, "pred_class": int(p >= 0.5)})

                st.dataframe(pd.DataFrame(results), use_container_width=True)
                st.caption("pred_class: 1 = responder (NoAnswer_coef==0), 0 = non-responder")
            else:
                y = per["regress"]["y"]
                X = per["regress"]["X"]
                model_obj, pred_fn = train_full_regressor(X, y, seed=int(seed), cb_iters=int(cb_iters))
                pred = float(pred_fn(X_new)[0])
                st.metric("Predicted titre", f"{pred:.4g}")
                st.caption(f"Using CatBoostRegressor. Titre column: `{per['regress']['titre_col']}`")


# ---------------- Per-vaccine tabs ----------------
for i, vac in enumerate(TARGETS_CLASSIF.keys(), start=1):
    with tabs[i]:
        st.header(vac)

        if st.session_state.cv_summary is None:
            st.info("Click **Run evaluation** in the sidebar.")
            continue

        per = st.session_state.per_vac[vac]

        # ---- Classification section ----
        st.subheader("1) Classification (response / non-response)")
        Xc = per["classif"]["X"]
        yc = per["classif"]["y"]
        st.write({"n": int(len(Xc)), "responder_rate": float(np.mean(yc)), "use_age": bool(use_age)})

        st.markdown("**CV results (per fold)**")
        st.write("CatBoost")
        st.dataframe(per["classif"]["cv"]["CatBoost"], use_container_width=True)

        lr = per["classif"]["cv"]["LogReg + OneHot"]
        if lr is not None:
            st.write("LogReg + OneHot")
            st.dataframe(lr, use_container_width=True)

        xg = per["classif"]["cv"]["XGBoost + OneHot"]
        if xg is not None:
            st.write("XGBoost + OneHot")
            st.dataframe(xg, use_container_width=True)

        # ---- Regression section (not for mumps) ----
        st.divider()
        st.subheader("2) Regression (predicted titre) — CatBoostRegressor")
        if vac not in REGRESSION_VACCINES:
            st.info("No titre available for this vaccine (mumps).")
        else:
            titre_col = per["regress"]["titre_col"]
            if titre_col is None:
                st.info("Select a titre column at the top of the page (Titre columns section) to enable regression.")
            else:
                Xr = per["regress"]["X"]
                yr = per["regress"]["y"]
                if Xr is None or yr is None or len(Xr) == 0:
                    st.warning("Regression dataset is empty after filtering missing titre values.")
                else:
                    st.write({"titre_col": titre_col, "n": int(len(Xr)), "mean_titre": float(np.mean(yr))})
                    cv_r = per["regress"]["cv"]["CatBoostRegressor"]
                    if cv_r is not None:
                        st.markdown("**CV results (per fold)**")
                        st.dataframe(cv_r, use_container_width=True)
                    else:
                        st.info("Run evaluation to compute regression CV metrics.")

        # ---- SHAP (classification CatBoost only; on demand) ----
        st.divider()
        st.subheader("3) SHAP (CatBoost classifier, on demand)")
        st.caption("Uses CatBoost `get_feature_importance(type='ShapValues')` (no `shap` package needed).")

        shap_key = ("classif", vac, int(seed), int(cb_iters), int(shap_rows), bool(use_age))

        if st.button(f"Compute SHAP for {vac} (classification)", key=f"shap_btn_{vac}"):
            cat_idx = [j for j, c in enumerate(Xc.columns) if c != "age"]
            model = CatBoostClassifier(
                iterations=int(cb_iters),
                depth=6,
                learning_rate=0.05,
                loss_function="Logloss",
                eval_metric="AUC",
                auto_class_weights="Balanced",
                random_seed=int(seed),
                verbose=False,
            )
            model.fit(Pool(Xc, yc, cat_features=cat_idx))
            imp = catboost_mean_abs_shap_any(model, Xc, yc, max_rows=int(shap_rows), seed=int(seed))
            st.session_state.shap_imp[shap_key] = imp

        imp = st.session_state.shap_imp.get(shap_key)
        if imp is None:
            st.info("Press **Compute SHAP** to see feature importances.")
        else:
            st.dataframe(imp.head(30), use_container_width=True)
            topk = 20
            imp_top = imp.head(topk).iloc[::-1]
            fig = plt.figure()
            plt.barh(imp_top["feature"], imp_top["mean_abs_shap"])
            plt.xlabel("mean(|SHAP|)")
            plt.title(f"{vac}: top-{topk} features by mean(|SHAP|)")
            st.pyplot(fig)