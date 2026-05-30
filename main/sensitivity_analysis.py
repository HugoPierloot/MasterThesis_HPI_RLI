# Standard library imports
import itertools
import json
import sys
import os
from datetime import datetime
from pathlib import Path
from copy import deepcopy

# Third part imports
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Path setup
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Local imports
from constants import Constant as C
from splitter import make_split
from models.baseline    import MajorityClassifier, PriorRateClassifier
from models.tree_models import get_all_models
from models.evaluation  import (
    evaluate_model, build_comparison_report,
    plot_confusion_matrix, plot_roc_curves,
    plot_precision_recall_curves, plot_metrics_comparison,
)
from models.explainability import run_explainability
from models.logit import build_logit, classify_feature_cols, run_logit_analysis


# ─────────────────────────────────────────────────────────────
#  ONE-SHOT configuration (default values)
# ─────────────────────────────────────────────────────────────

ONE_SHOT_CONFIG = {
    # Split
    "cut_year"        : 2010,

    # SMOTE
    "apply_smote"     : True,
    "smote_strategy"  : 0.3, # minority/majority ratio after SMOTE

    # Threshold
    "threshold"       : 0.5, # decision threshold for binary output

    # Random Forest
    "rf_n_estimators" : 300,
    "rf_max_depth"    : None,  # None = grow full trees
    "rf_min_samples_leaf": 2,

    # XGBoost
    "xgb_n_estimators"  : 300,
    "xgb_max_depth"     : 6,
    "xgb_learning_rate" : 0.05,
    "xgb_subsample"     : 0.8,

    # AdaBoost
    "ada_n_estimators"  : 200,
    "ada_learning_rate" : 0.1,

    # RUSBoost
    "rus_n_estimators"  : 200,
    "rus_learning_rate" : 0.1,

    # Logit (logistic regression — linear model)
    "logit_C"           : 1.0,     # inverse L2 regularisation (smaller = stronger)
    "logit_penalty"     : "l2",    # 'l2', 'l1', or 'none'

    # SHAP
    "shap_dependence_features": [
        "rocket_prior_launches",
        "org_prior_success_rate",
        "Launch Year",
    ],
    "shap_waterfall_cases": 3,
}

SENSITIVITY_GRID = {
    "smote_strategy"   : [0.2, 0.3, 0.5],
    "threshold"        : [0.3, 0.5],
    "xgb_learning_rate": [0.01, 0.05, 0.1],
    # "rf_n_estimators": [100, 300, 500],
    # "rf_max_depth"   : [None, 10, 20],
    # "ada_n_estimators": [100, 200],
}


# ─────────────────────────────────────────────────────────────
#  Parameter => model relevance map
#
#  Maps each model to the config keys it actually uses. Used both by
#  the grid de-duplication and by the per-axis sensitivity charts so
#  that a chart is produced ONLY for a model that actually consumes
#  the swept parameter (no more "AdaBoost vs rf_n_estimators").
# ─────────────────────────────────────────────────────────────

MODEL_PARAM_KEYS = {
    "RandomForest": {"rf_n_estimators", "rf_max_depth", "rf_min_samples_leaf",
                     "smote_strategy", "threshold", "cut_year", "apply_smote"},
    "XGBoost":      {"xgb_n_estimators", "xgb_max_depth", "xgb_learning_rate",
                     "xgb_subsample", "smote_strategy", "threshold",
                     "cut_year", "apply_smote"},
    "AdaBoost":     {"ada_n_estimators", "ada_learning_rate",
                     "smote_strategy", "threshold", "cut_year", "apply_smote"},
    "RUSBoost":     {"rus_n_estimators", "rus_learning_rate",
                     "smote_strategy", "threshold", "cut_year", "apply_smote"},
    "Logit":        {"logit_C", "logit_penalty",
                     "smote_strategy", "threshold", "cut_year", "apply_smote"},
}

# Type coercion sets for reconstructing a config from a saved best-row.
_INT_PARAMS = {
    "cut_year", "rf_n_estimators", "rf_min_samples_leaf",
    "xgb_n_estimators", "xgb_max_depth", "ada_n_estimators", "rus_n_estimators",
}
_FLOAT_PARAMS = {
    "smote_strategy", "threshold", "xgb_learning_rate", "xgb_subsample",
    "ada_learning_rate", "rus_learning_rate", "xgb_scale_pos_weight",
}
_BOOL_PARAMS = {"apply_smote"}


# ─────────────────────────────────────────────────────────────
#  Directory helpers
# ─────────────────────────────────────────────────────────────

def _sensitivity_dir() -> Path:
    """Root folder for all sensitivity outputs: figures/sensitivity/."""
    p = C.DATA_PATH_OUTPUTS_FIG / "sensitivity"
    p.mkdir(parents=True, exist_ok=True)
    return p

def _best_configs_dir() -> Path:
    """figures/sensitivity/best_configs/ — replaces the per-run subfolders."""
    p = _sensitivity_dir() / "best_configs"
    p.mkdir(parents=True, exist_ok=True)
    return p

def _best_model_dir(model_name: str) -> Path:
    """One subfolder per model under best_configs/ (best-F1 config)."""
    p = _best_configs_dir() / model_name.lower()
    p.mkdir(parents=True, exist_ok=True)
    return p

def _best_model_shap_dir(model_name: str) -> Path:
    p = _best_model_dir(model_name) / "shap"
    p.mkdir(parents=True, exist_ok=True)
    return p

def _run_fig_dir(run_id: str) -> Path:
    """Single folder for a one-shot run (figures/sensitivity/<run_id>/)."""
    p = _sensitivity_dir() / run_id
    p.mkdir(parents=True, exist_ok=True)
    return p

def _run_shap_dir(run_id: str) -> Path:
    p = _run_fig_dir(run_id) / "shap"
    p.mkdir(parents=True, exist_ok=True)
    return p

def _reports_dir() -> Path:
    p = C.DATA_PATH_OUTPUTS_EVL_REP
    p.mkdir(parents=True, exist_ok=True)
    return p


# ─────────────────────────────────────────────────────────────
#  Parameter => model builder
# ─────────────────────────────────────────────────────────────

def _build_models(
    cfg: dict,
    feature_cols: list = None,
    classes: dict = None,
    random_state: int = 42,
) -> dict:
    """
    Instantiate all models using parameters from cfg dict.
    Overrides defaults in tree_models.py with run-specific values.

    feature_cols / classes are required to build the Logit model (it needs to
    know each column's name and kind for its internal preprocessing). When they
    are None, the Logit model is omitted (e.g. for callers that only want trees).
    """
    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
    from sklearn.tree import DecisionTreeClassifier
    from xgboost import XGBClassifier
    from imblearn.ensemble import RUSBoostClassifier

    rf = RandomForestClassifier(
        n_estimators    = cfg.get("rf_n_estimators",    300),
        max_depth       = cfg.get("rf_max_depth",       None),
        min_samples_leaf= cfg.get("rf_min_samples_leaf", 2),
        max_features    = "sqrt",
        class_weight    = "balanced",
        n_jobs          = -1,
        random_state    = random_state,
    )

    xgb = XGBClassifier(
        n_estimators     = cfg.get("xgb_n_estimators",   300),
        max_depth        = cfg.get("xgb_max_depth",        6),
        learning_rate    = cfg.get("xgb_learning_rate",  0.05),
        subsample        = cfg.get("xgb_subsample",       0.8),
        colsample_bytree = 0.8,
        scale_pos_weight = cfg.get("xgb_scale_pos_weight", 3.0),
        eval_metric      = "aucpr",
        use_label_encoder= False,
        random_state     = random_state,
        n_jobs           = -1,
        verbosity        = 0,
    )

    base_dt = DecisionTreeClassifier(
        max_depth    = 2,
        class_weight = "balanced",
        random_state = random_state,
    )
    ada = AdaBoostClassifier(
        estimator     = base_dt,
        n_estimators  = cfg.get("ada_n_estimators",  200),
        learning_rate = cfg.get("ada_learning_rate", 0.1),
        random_state  = random_state,
    )

    rus = RUSBoostClassifier(
        n_estimators      = cfg.get("rus_n_estimators",  200),
        learning_rate     = cfg.get("rus_learning_rate", 0.1),
        sampling_strategy = "auto",
        random_state      = random_state,
    )

    models = {
        "Baseline_Majority" : MajorityClassifier(),
        "Baseline_PriorRate": PriorRateClassifier(),
        "RandomForest"      : rf,
        "XGBoost"           : xgb,
        "AdaBoost"          : ada,
        "RUSBoost"          : rus,
    }

    # Logit needs the column names/kinds for its own preprocessing
    if feature_cols is not None and classes is not None:
        models["Logit"] = build_logit(
            feature_cols = feature_cols,
            classes      = classes,
            C_reg        = cfg.get("logit_C", 1.0),
            penalty      = cfg.get("logit_penalty", "l2"),
            random_state = random_state,
        )

    return models


# ─────────────────────────────────────────────────────────────
#  Single run
# ─────────────────────────────────────────────────────────────

def run_single(
    cfg        : dict,
    run_id     : str,
    run_shap   : bool = True,
    make_figures: bool = True,
    random_state: int = 42,
    verbose    : bool = True,
) -> dict:
    """
    One complete training + evaluation (+ optional SHAP) run.

    Args:
        cfg         : parameter dict (merge ONE_SHOT_CONFIG + overrides)
        run_id      : unique string label - used for filenames and subfolder
        run_shap    : whether to run SHAP explainability (ignored if make_figures
                      is False)
        make_figures: whether to render per-run figures (confusion matrices,
                      ROC/PR curves, metrics comparison, SHAP). Set False during
                      a grid sweep — figures for the grid are produced once, for
                      the best config of each model, by render_best_configs().
        random_state: seed
        verbose     : print progress

    Returns:
        dict with keys: results, df_report, fitted_models, split, shap_dict
    """
    fig_dir = _run_fig_dir(run_id) if make_figures else None

    if verbose:
        print(f"\n{'='*60}")
        print(f"RUN: {run_id}")
        print(f"Config: {json.dumps({k:v for k,v in cfg.items() if 'shap' not in k and k != '_run_for_models'}, indent=2, default=str)}")
        print(f"{'='*60}")

    # 1. Split + SMOTE
    split = make_split(
        cut_year       = cfg["cut_year"],
        apply_smote    = cfg["apply_smote"],
        smote_strategy = cfg["smote_strategy"],
        random_state   = random_state,
        verbose        = verbose,
    )
    X_train = split["X_train"]
    X_test  = split["X_test"]
    y_train = split["y_train"]
    y_test  = split["y_test"]
    y_raw   = split["df_train"]["launch_success_binary"].values
    X_raw   = split["X_train_raw"].values

    ir = (y_raw == 0).sum() / max((y_raw == 1).sum(), 1)

    # 2. Build + train models
    feature_cols = split["feature_cols"]
    col_classes  = classify_feature_cols(split["df_train"], feature_cols)
    models = _build_models(cfg, feature_cols, col_classes, random_state)
    models["XGBoost"].set_params(scale_pos_weight=float(ir))

    # Only train models flagged as needing a fresh run for this config.	
    run_for = cfg.get("_run_for_models", None) 

    fitted_models = {}
    for name, model in models.items():
        is_baseline = "Baseline" in name
        if run_for is not None and not is_baseline:
            # Extract base model name ("RandomForest", "XGBoost", etc.)
            base_name = name.replace("Baseline_Majority","").replace("Baseline_PriorRate","")
            if name not in run_for:
                if verbose:
                    print(f"  Skipping {name} (params unchanged for this run)")
                continue 
        if verbose:
            print(f"  Training {name}...", end=" ", flush=True)
        if is_baseline:
            model.fit(X_raw, y_raw)
        else:
            model.fit(X_train, y_train)
        fitted_models[name] = model
        if verbose:
            print("done")

    # 3. Evaluate
    threshold = cfg.get("threshold", 0.5)
    results   = []
    for name, model in fitted_models.items():
        m = evaluate_model(model, X_test, y_test,
                           model_name=name, threshold=threshold)
        m["run_id"]    = run_id
        m["threshold"] = threshold
        # Key config params to each row for later comparison
        skip_keys = {"_run_for_models", "shap_dependence_features", 
                     "shap_waterfall_cases"}
        for key, val in cfg.items():
            if key not in skip_keys:
                m[key] = val
        results.append(m)

    df_report = build_comparison_report(results)

    if verbose:
        display_cols = ["precision","recall","f1","f2","auc_pr","mcc"]
        print(f"\n  Results (run={run_id}):")
        print(df_report[display_cols].round(3).to_string())

    # 4. Save evaluation figures (skipped during grid sweeps)
    # Override C paths temporarily via monkey-patch on evaluation module
    if make_figures:
        import models.evaluation as ev_mod
        _orig_fig_dir_fn = ev_mod._get_fig_dir
        ev_mod._get_fig_dir = lambda: fig_dir # redirect all saves to run subfolder

        try:
            for name, model in fitted_models.items():
                if "Baseline" not in name:
                    plot_confusion_matrix(model, X_test, y_test,
                                          model_name=name,
                                          threshold=threshold)
            plot_roc_curves(results, y_test, fitted_models, X_test)
            plot_precision_recall_curves(fitted_models, X_test, y_test)
            plot_metrics_comparison(df_report)
        finally:
            ev_mod._get_fig_dir = _orig_fig_dir_fn
            plt.close("all")

    # 5. SHAP (optional [parameter]) — only when figures are being produced
    shap_dict = {}
    if make_figures and run_shap:
        import models.explainability as ex_mod
        _orig_fig_dir_fn_shap = ex_mod._fig_dir
        ex_mod._fig_dir = lambda: _run_shap_dir(run_id)

        try:
            shap_dict = run_explainability(
                fitted_models       = fitted_models,
                split               = split,
                top_n               = 20,
                waterfall_cases     = cfg.get("shap_waterfall_cases", 3),
                dependence_features = cfg.get("shap_dependence_features", []),
                run_permutation     = True,
            )
        finally:
            ex_mod._fig_dir = _orig_fig_dir_fn_shap
            plt.close("all")

    return {
        "run_id"       : run_id,
        "cfg"          : cfg,
        "results"      : results,
        "df_report"    : df_report,
        "fitted_models": fitted_models,
        "split"        : split,
        "shap_dict"    : shap_dict,
    }


# ─────────────────────────────────────────────────────────────
#  Grid builder
# ─────────────────────────────────────────────────────────────

def _build_grid(grid: dict, base_cfg: dict) -> list[dict]:
    """
    Returns a list of config dicts - one per combination in the grid.
    Each dict is base_cfg with the grid overrides applied.
    """
    keys   = list(grid.keys())
    values = list(grid.values())
    combos = list(itertools.product(*values))

    # Mapping each model to the config keys it actually uses (module-level).
    # Keys that belong to a different model are ignored when de-duplicating.
    MODEL_KEYS = MODEL_PARAM_KEYS
    all_configs = []
    for combo in combos:
        cfg = deepcopy(base_cfg)
        for k, v in zip(keys, combo):
            cfg[k] = v
        all_configs.append(cfg)
 
    # Deduplicate: for each model, only keep configs that differ on params relevant to that model.
    seen_per_model  = {m: set() for m in MODEL_KEYS}
    dedup_configs   = []
    dedup_per_model = {m: [] for m in MODEL_KEYS}
 
    for cfg in all_configs:
        for model, relevant_keys in MODEL_KEYS.items():
            sig = tuple(sorted(
                (k, cfg.get(k)) for k in relevant_keys if k in cfg
            ))
            if sig not in seen_per_model[model]:
                seen_per_model[model].add(sig)
                dedup_per_model[model].append(cfg)
 
    # Keeping configs that are unique for AT LEAST ONE model.
    # Tagging each config with which models need a fresh run.
    seen_globally = set()
    configs = []
    for cfg in all_configs:
        needed = set()
        for model, relevant_keys in MODEL_KEYS.items():
            sig = tuple(sorted(
                (k, cfg.get(k)) for k in relevant_keys if k in cfg
            ))
            sig_with_model = (model,) + sig
            if sig_with_model not in seen_globally:
                seen_globally.add(sig_with_model)
                needed.add(model)
        if needed:
            cfg = deepcopy(cfg)
            # tag: only these models need fresh training
            cfg["_run_for_models"] = needed
            configs.append(cfg)
 
    n_original = len(all_configs)
    n_dedup    = len(configs)
    print(f"  Grid deduplication: {n_original} combinations -> {n_dedup} unique runs")
    return configs


# ─────────────────────────────────────────────────────────────
#  Per-axis sensitivity scatter (F1 only)
# ─────────────────────────────────────────────────────────────

def plot_sensitivity_scatter(df_all: pd.DataFrame, sweep_param: str):
    """
    For a single swept parameter, produce one F1 scatter plot PER MODEL.

    X axis = parameter value, Y axis = F1-score. No other metric and no
    legend: comparison is made on F1 alone for visibility.

    A chart is produced for a model ONLY when the swept parameter is one
    the model actually consumes (see MODEL_PARAM_KEYS) and when that model
    was evaluated at more than one value of it. This removes meaningless
    pairs such as "AdaBoost vs rf_n_estimators".

    Each point is one evaluated configuration; several points can share an
    x value when other relevant parameters also vary, so the scatter shows
    the spread of F1 at each parameter value.
    """
    fig_dir = _sensitivity_dir()

    models_in = [m for m in df_all["model"].unique() if "Baseline" not in m]

    for model_name in models_in:
        relevant = MODEL_PARAM_KEYS.get(model_name, set())
        if sweep_param not in relevant:
            continue  # parameter not used by this model -> no chart

        sub = df_all[df_all["model"] == model_name].copy()
        if sweep_param not in sub.columns or sub[sweep_param].nunique() < 2:
            continue  # nothing swept for this model on this axis

        sub = sub.sort_values(sweep_param)
        x_labels = sub[sweep_param].astype(str)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(x_labels, sub["f1"], s=70, color="#e74c3c",
                   alpha=0.8, edgecolor="white", linewidth=1, zorder=3)

        # Annotate the single best-F1 point
        best_idx = sub["f1"].idxmax()
        ax.annotate(
            f"{sub.loc[best_idx, 'f1']:.3f}",
            xy=(str(sub.loc[best_idx, sweep_param]), sub.loc[best_idx, "f1"]),
            xytext=(0, 9), textcoords="offset points",
            ha="center", fontsize=9, fontweight="bold", color="#c0392b",
        )

        ax.set_xlabel(sweep_param)
        ax.set_ylabel("F1-score")
        ax.set_ylim(0, 1.05)
        ax.set_title(f"Sensitivity — {model_name}\n"
                     f"F1 vs {sweep_param}",
                     fontweight="bold")
        ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.5)
        plt.tight_layout()
        fname = fig_dir / f"sensitivity_{sweep_param}_{model_name.lower()}.png"
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved => {fname}")


# ─────────────────────────────────────────────────────────────
#  Best-config aggregation + rendering
# ─────────────────────────────────────────────────────────────

def compute_best_configs(df_all: pd.DataFrame) -> pd.DataFrame:
    """
    For each (non-baseline) model, keep the single row that maximises F1.

    Returns a DataFrame indexed by model name, holding every metric and
    every config parameter for that best-F1 configuration.
    """
    metric_cols = ["f1", "recall", "precision", "f2", "auc_pr", "mcc",
                   "accuracy", "auc_roc", "TP", "FP", "TN", "FN",
                   "run_id", "threshold"]
    skip_keys = {"_run_for_models", "shap_dependence_features", "shap_waterfall_cases"}
    cfg_cols  = [k for k in ONE_SHOT_CONFIG.keys()
                 if k not in skip_keys and k in df_all.columns]
    # Preserve order while dropping duplicates (e.g. "threshold" is in both lists)
    seen = set()
    keep_cols = [c for c in metric_cols + cfg_cols
                 if c in df_all.columns and not (c in seen or seen.add(c))]

    best = (
        df_all[~df_all["model"].str.contains("Baseline")]
        .sort_values("f1", ascending=False)
        .groupby("model")
        .first()
        [keep_cols]
    )
    return best


def _cfg_from_best_row(row: pd.Series, base_cfg: dict) -> dict:
    """
    Rebuild a full config dict from a saved best-config row, coercing each
    parameter back to its proper type (ints stay ints, rf_max_depth=None when
    absent, etc.). base_cfg supplies defaults for anything not in the row.
    """
    cfg = deepcopy(base_cfg)
    for k in list(base_cfg.keys()):
        if k not in row.index:
            continue
        v = row[k]
        if pd.isna(v):
            if k == "rf_max_depth":
                cfg[k] = None
            continue
        if k in _INT_PARAMS or k == "rf_max_depth":
            cfg[k] = int(round(float(v)))
        elif k in _FLOAT_PARAMS:
            cfg[k] = float(v)
        elif k in _BOOL_PARAMS:
            cfg[k] = bool(v)
        else:
            cfg[k] = v
    return cfg


def _train_one(model_name: str, cfg: dict, random_state: int = 42):
    """
    Build the split for cfg, train just `model_name`, and return
    (fitted_model, split). XGBoost's scale_pos_weight is set from the
    raw (pre-SMOTE) imbalance ratio, exactly as in run_single.
    """
    split = make_split(
        cut_year       = cfg["cut_year"],
        apply_smote    = cfg["apply_smote"],
        smote_strategy = cfg["smote_strategy"],
        random_state   = random_state,
        verbose        = False,
    )
    y_raw  = split["df_train"]["launch_success_binary"].values
    ir     = (y_raw == 0).sum() / max((y_raw == 1).sum(), 1)

    feature_cols = split["feature_cols"]
    col_classes  = classify_feature_cols(split["df_train"], feature_cols)
    models = _build_models(cfg, feature_cols, col_classes, random_state)
    if "XGBoost" in models:
        models["XGBoost"].set_params(scale_pos_weight=float(ir))

    model = models[model_name]
    model.fit(split["X_train"], split["y_train"])
    return model, split


_CURVE_COLORS = ["#2ecc71", "#3498db", "#e74c3c", "#9b59b6", "#f39c12", "#1abc9c"]


def _plot_roc_best(trained: dict, out_dir: Path):
    """ROC overlay across each model's best-F1 configuration."""
    from sklearn.metrics import roc_curve, roc_auc_score

    fig, ax = plt.subplots(figsize=(8, 6))
    for i, (name, (model, split, _thr)) in enumerate(trained.items()):
        if not hasattr(model, "predict_proba"):
            continue
        X_test, y_test = split["X_test"], split["y_test"]
        proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, proba)
        auc = roc_auc_score(y_test, proba)
        ax.plot(fpr, tpr, color=_CURVE_COLORS[i % len(_CURVE_COLORS)],
                linewidth=2, label=f"{name}  (AUC={auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random baseline")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate (Recall)")
    ax.set_title("ROC Curves — best F1 config per model", fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    path = out_dir / "roc_curves_best_configs.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved => {path}")


def _plot_pr_best(trained: dict, out_dir: Path):
    """Precision-Recall overlay across each model's best-F1 configuration."""
    from sklearn.metrics import precision_recall_curve, average_precision_score

    fig, ax = plt.subplots(figsize=(8, 6))
    baseline_precision = None
    for i, (name, (model, split, _thr)) in enumerate(trained.items()):
        if not hasattr(model, "predict_proba"):
            continue
        X_test, y_test = split["X_test"], split["y_test"]
        if baseline_precision is None:
            baseline_precision = float(np.mean(y_test))
        proba = model.predict_proba(X_test)[:, 1]
        prec, rec, _ = precision_recall_curve(y_test, proba)
        ap = average_precision_score(y_test, proba)
        ax.plot(rec, prec, color=_CURVE_COLORS[i % len(_CURVE_COLORS)],
                linewidth=2, label=f"{name}  (AP={ap:.3f})")

    if baseline_precision is not None:
        ax.axhline(y=baseline_precision, color="grey", linestyle="--",
                   linewidth=1, label=f"Random baseline ({baseline_precision:.3f})")
    ax.set_xlabel("Recall (coverage of failures)")
    ax.set_ylabel("Precision (accuracy of failure alerts)")
    ax.set_title("Precision-Recall Curves — best F1 config per model\n"
                 "(more informative than ROC for imbalanced data)",
                 fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    path = out_dir / "pr_curves_best_configs.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved => {path}")


def render_best_configs(
    df_all   : pd.DataFrame,
    best     : pd.DataFrame,
    base_cfg : dict,
    run_shap : bool = False,
    random_state: int = 42,
    verbose  : bool = True,
):
    """
    Produce the aggregated figures into figures/sensitivity/best_configs/,
    using, for each model, the configuration that maximised its F1 score.

    Outputs:
        best_configs/
            metrics_comparison_all_models.png   (all metrics, best-F1 row/model)
            roc_curves_best_configs.png
            pr_curves_best_configs.png
            shap_importance_comparison_all_models.png   (if run_shap)
            <model>/
                cm_<model>.png
                shap/   (if run_shap: beeswarm, importance, 5 waterfalls,
                         10 dependence, permutation importance)
    """
    import models.evaluation as ev_mod
    import models.explainability as ex_mod

    best_dir = _best_configs_dir()
    df_cmp   = best.sort_values("f1", ascending=False)

    print(f"\n{'='*60}")
    print("RENDERING BEST-CONFIG FIGURES")
    print(f"  => {best_dir}")
    print(f"{'='*60}")

    # 1. Aggregated metrics comparison (best-F1 config of each model)
    _orig = ev_mod._get_fig_dir
    ev_mod._get_fig_dir = lambda: best_dir
    try:
        plot_metrics_comparison(df_cmp)
    finally:
        ev_mod._get_fig_dir = _orig
        plt.close("all")

    # 2. Re-train each model at its best config (needed for curves/CM/SHAP)
    trained = {}
    for model_name, row in df_cmp.iterrows():
        cfg = _cfg_from_best_row(row, base_cfg)
        if verbose:
            print(f"  Re-training {model_name} at best config "
                  f"(F1={row['f1']:.3f}, run={row.get('run_id','?')})...",
                  end=" ", flush=True)
        model, split = _train_one(model_name, cfg, random_state)
        threshold = (float(row["threshold"]) if "threshold" in row.index
                     and pd.notna(row["threshold"]) else cfg.get("threshold", 0.5))
        trained[model_name] = (model, split, threshold)
        if verbose:
            print("done")

    # 3. ROC + PR overlays (best config per model)
    _plot_roc_best(trained, best_dir)
    _plot_pr_best(trained, best_dir)

    # 4. Per-model: confusion matrix + SHAP
    shap_dict  = {}
    feat_names = None
    for model_name, (model, split, threshold) in trained.items():
        mdir = _best_model_dir(model_name)
        X_test, y_test = split["X_test"], split["y_test"]

        _orig = ev_mod._get_fig_dir
        ev_mod._get_fig_dir = lambda d=mdir: d
        try:
            plot_confusion_matrix(model, X_test, y_test,
                                  model_name=model_name, threshold=threshold)
        finally:
            ev_mod._get_fig_dir = _orig
            plt.close("all")

        if run_shap:
            sdir = _best_model_shap_dir(model_name)
            _origs = ex_mod._fig_dir
            ex_mod._fig_dir = lambda d=sdir: d
            try:
                sv = ex_mod.run_explainability_single_model(
                    model               = model,
                    model_name          = model_name,
                    split               = split,
                    top_n               = 20,
                    n_waterfall         = 5,
                    n_dependence        = 10,
                    dependence_features = base_cfg.get("shap_dependence_features", []),
                    run_permutation     = True,
                )
                shap_dict[model_name] = sv
                feat_names = split["feature_cols"]
            except Exception as e:
                print(f"    [WARNING] SHAP failed for {model_name}: {e}")
            finally:
                ex_mod._fig_dir = _origs
                plt.close("all")

    # 5. Cross-model SHAP comparison + report into best_configs/
    if run_shap and len(shap_dict) >= 2 and feat_names is not None:
        _origs = ex_mod._fig_dir
        ex_mod._fig_dir = lambda: best_dir
        try:
            ex_mod.plot_feature_importance_comparison(shap_dict, feat_names, top_n=20)
            ts = datetime.today().strftime("%Y%m%d_%Hh%M")
            ex_mod.save_shap_report(shap_dict, feat_names, f"shap_report_{ts}.csv")
        finally:
            ex_mod._fig_dir = _origs
            plt.close("all")

    # 6. Logit interpretation text reports (one per swept config found in df_all)
    #    For each distinct (logit_C, logit_penalty) pair that appeared across the
    #    grid, a labelled plain-English .txt is saved under best_configs/logit/.
    #    Reports are numbered by descending F1, so "LOGIT 1" is always the winner.
    if "Logit" in df_all.get("model", pd.Series()).values if hasattr(df_all, "get") else "Logit" in df_all["model"].values:
        logit_dir = _best_model_dir("logit")
        logit_rows = df_all[df_all["model"] == "Logit"].copy()

        # Collect distinct (C, penalty) pairs, sorted by F1 descending so
        # "LOGIT 1" is the best-performing config.
        c_col   = "logit_C"       if "logit_C"       in logit_rows.columns else None
        pen_col = "logit_penalty" if "logit_penalty"  in logit_rows.columns else None

        if c_col and pen_col:
            distinct = (logit_rows
                        .sort_values("f1", ascending=False)
                        [[c_col, pen_col, "f1"]]
                        .drop_duplicates(subset=[c_col, pen_col])
                        .reset_index(drop=True))
        else:
            # No sweep on logit params — use base_cfg values only
            distinct = pd.DataFrame([{
                "logit_C":       base_cfg.get("logit_C", 1.0),
                "logit_penalty": base_cfg.get("logit_penalty", "l2"),
                "f1": logit_rows["f1"].max() if len(logit_rows) else float("nan"),
            }])

        best_f1_c   = float(distinct.iloc[0]["logit_C"])
        best_f1_pen = str(distinct.iloc[0]["logit_penalty"])

        for report_n, row in distinct.iterrows():
            c_val       = float(row["logit_C"])
            penalty_val = str(row["logit_penalty"])
            is_best     = (c_val == best_f1_c and penalty_val == best_f1_pen)
            label = (
                f"LOGIT {report_n + 1}"
                f" - logit_C={c_val}, logit_penalty={penalty_val}"
                + (" [BEST F1 configuration]" if is_best else "")
            )
            print(f"\n  Generating Logit interpretation report ({report_n + 1}/"
                  f"{len(distinct)}): {label}")
            try:
                run_logit_analysis(
                    save         = True,
                    verbose      = False,
                    config_label = label,
                    out_dir      = logit_dir,
                )
            except Exception as e:
                print(f"    [WARNING] Logit interpretation report failed: {e}")

    print(f"\nBest-config figures complete => {best_dir}")


# ─────────────────────────────────────────────────────────────
#  Master runners
# ─────────────────────────────────────────────────────────────

def run_one_shot(
    cfg      : dict = None,
    run_shap : bool = True,
    verbose  : bool = True,
):
    """
    Run a single training + evaluation (+ optional SHAP).

    Args:
        cfg      : dict of parameter overrides (merged on top of ONE_SHOT_CONFIG).
                   Pass only the keys you want to change - all others use defaults.
        run_shap : include SHAP explainability (default True). Set False for speed.
        verbose  : print progress.

    Returns:
        dict with keys: run_id, cfg, results, df_report,
                        fitted_models, split, shap_dict
    """
    from copy import deepcopy
    ts            = datetime.today().strftime("%Y%m%d_%Hh%M")
    run_id        = f"oneshot_{ts}"
    effective_cfg = deepcopy(ONE_SHOT_CONFIG)
    if cfg:
        effective_cfg.update(cfg)

    result = run_single(
        cfg      = effective_cfg,
        run_id   = run_id,
        run_shap = run_shap,
        verbose  = verbose,
    )
    path = _reports_dir() / f"report_{run_id}.csv"
    result["df_report"].to_csv(path)
    print(f"\nReport saved => {path}")
    return result


def run_sensitivity(
    grid     : dict = None,
    base_cfg : dict = None,
    run_shap : bool = False,
    verbose  : bool = True,
):
    """
    Sweep a parameter grid, then render figures only for the best
    configuration of each model.

    Two phases:
      1. Grid sweep — one training+evaluation per (deduplicated) config.
         NO per-run figures are written; only metric rows are collected.
         Per-axis F1 scatter plots are then produced directly under
         figures/sensitivity/ (one per model x relevant parameter).
      2. Best-config rendering — for each model the F1-maximising config is
         retrained once and its figures (metrics comparison, ROC, PR,
         confusion matrix, and optional SHAP) are written under
         figures/sensitivity/best_configs/. The old per-run "run{N}/"
         subfolders are no longer created (every run is preserved in the
         sensitivity_grid_*.csv report instead).

    Args:
        grid     : dict mapping parameter name -> list of values.
                   Grid = cartesian product of all lists.
                   Defaults to SENSITIVITY_GRID defined at top of file.
        base_cfg : base config dict. Grid values are overlaid on top.
                   Defaults to ONE_SHOT_CONFIG.
        run_shap : run SHAP for the best config of each model (default False).
                   SHAP is no longer run at every grid point.
        verbose  : print progress for each run.

    Returns:
        pd.DataFrame - one row per (run x model).
        Saved files:
          evaluation_reports/sensitivity_grid_{ts}.csv
          evaluation_reports/sensitivity_best_configs_{ts}.csv
          figures/sensitivity/sensitivity_{param}_{model}.png   (F1 scatter)
          figures/sensitivity/best_configs/...                  (aggregated)
    """
    from copy import deepcopy
    grid     = grid     or SENSITIVITY_GRID
    base_cfg = base_cfg or ONE_SHOT_CONFIG
    configs  = _build_grid(grid, base_cfg)
    ts       = datetime.today().strftime("%Y%m%d_%Hh%M")

    print(f"\n{'='*60}")
    print(f"SENSITIVITY ANALYSIS — {len(configs)} runs")
    print(f"Grid axes : {list(grid.keys())}")
    print(f"Values    : { {k: v for k, v in grid.items()} }")
    print(f"{'='*60}")

    # ── Phase 1: metric collection only (no per-run figures) ──
    all_rows = []
    for i, cfg in enumerate(configs):
        run_id = f"run{i:03d}"
        result = run_single(
            cfg         = cfg,
            run_id      = run_id,
            run_shap    = False,
            make_figures= False,
            verbose     = verbose,
        )
        all_rows.extend(result["results"])
        plt.close("all")

    df_all = pd.DataFrame(all_rows)

    # Save full grid report (every run is preserved here)
    grid_path = _reports_dir() / f"sensitivity_grid_{ts}.csv"
    df_all.to_csv(grid_path, index=False)
    print(f"\nFull grid report saved => {grid_path}")

    # Per-axis F1 scatter charts (only relevant model x parameter pairs)
    for param in grid.keys():
        if param in df_all.columns:
            plot_sensitivity_scatter(df_all, sweep_param=param)

    # ── Best config per model ──
    best = compute_best_configs(df_all)
    print(f"\n{'='*60}")
    print("BEST CONFIG PER MODEL (by F1)")
    print(f"{'='*60}")
    print(best.round(3).to_string())

    best_path = _reports_dir() / f"sensitivity_best_configs_{ts}.csv"
    best.to_csv(best_path)
    print(f"\nBest configs saved => {best_path}")

    # ── Phase 2: render aggregated best-config figures ──
    render_best_configs(
        df_all   = df_all,
        best     = best,
        base_cfg = base_cfg,
        run_shap = run_shap,
        verbose  = verbose,
    )

    return df_all