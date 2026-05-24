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
#  Directory helpers
# ─────────────────────────────────────────────────────────────

def _run_fig_dir(run_id: str) -> Path:
    """One subfolder per run under figures/sensitivity/."""
    p = C.DATA_PATH_OUTPUTS_FIG / "sensitivity" / run_id
    p.mkdir(parents=True, exist_ok=True)
    return p

def _run_shap_dir(run_id: str) -> Path:
    p = C.DATA_PATH_OUTPUTS_FIG / "sensitivity" / run_id / "shap"
    p.mkdir(parents=True, exist_ok=True)
    return p

def _reports_dir() -> Path:
    p = C.DATA_PATH_OUTPUTS_EVL_REP
    p.mkdir(parents=True, exist_ok=True)
    return p


# ─────────────────────────────────────────────────────────────
#  Parameter => model builder
# ─────────────────────────────────────────────────────────────

def _build_models(cfg: dict, random_state: int = 42) -> dict:
    """
    Instantiate all models using parameters from cfg dict.
    Overrides defaults in tree_models.py with run-specific values.
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

    return {
        "Baseline_Majority" : MajorityClassifier(),
        "Baseline_PriorRate": PriorRateClassifier(),
        "RandomForest"      : rf,
        "XGBoost"           : xgb,
        "AdaBoost"          : ada,
        "RUSBoost"          : rus,
    }


# ─────────────────────────────────────────────────────────────
#  Single run
# ─────────────────────────────────────────────────────────────

def run_single(
    cfg        : dict,
    run_id     : str,
    run_shap   : bool = True,
    random_state: int = 42,
    verbose    : bool = True,
) -> dict:
    """
    One complete training + evaluation (+ optional SHAP) run.

    Args:
        cfg         : parameter dict (merge ONE_SHOT_CONFIG + overrides)
        run_id      : unique string label - used for filenames and subfolder
        run_shap    : whether to run SHAP explainability
        random_state: seed
        verbose     : print progress

    Returns:
        dict with keys: results, df_report, fitted_models, split, shap_dict
    """
    fig_dir = _run_fig_dir(run_id)

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
    models = _build_models(cfg, random_state)
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

    # 4. Save evaluation figures
    # Override C paths temporarily via monkey-patch on evaluation module
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

    # 5. SHAP (optional [parameter])
    shap_dict = {}
    if run_shap:
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
 
    # Mapping each model to the config keys it actually uses.
    # Keys that belong to a different model are ignored and so sweeping.
    MODEL_KEYS = {
        "RandomForest" : {"rf_n_estimators", "rf_max_depth", "rf_min_samples_leaf",
                          "smote_strategy", "threshold", "cut_year", "apply_smote"},
        "XGBoost"      : {"xgb_n_estimators", "xgb_max_depth", "xgb_learning_rate",
                          "xgb_subsample", "smote_strategy", "threshold",
                          "cut_year", "apply_smote"},
        "AdaBoost"     : {"ada_n_estimators", "ada_learning_rate",
                          "smote_strategy", "threshold", "cut_year", "apply_smote"},
        "RUSBoost"     : {"rus_n_estimators", "rus_learning_rate",
                          "smote_strategy", "threshold", "cut_year", "apply_smote"},
    }
 
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
#  Sensitivity summary chart
# ─────────────────────────────────────────────────────────────

def plot_sensitivity_summary(df_all: pd.DataFrame, sweep_param: str):
    """
    For a single swept parameter, plot F1 / Recall / AUC-PR
    of each model across its values.
    One chart per model - overlay metrics as lines.
    """
    fig_dir = C.DATA_PATH_OUTPUTS_FIG / "sensitivity"
    fig_dir.mkdir(parents=True, exist_ok=True)

    models_in = [m for m in df_all["model"].unique()
                 if "Baseline" not in m]
    metrics   = ["f1", "recall", "auc_pr"]
    colors    = {"f1": "#e74c3c", "recall": "#2ecc71", "auc_pr": "#3498db"}

    for model_name in models_in:
        sub = df_all[df_all["model"] == model_name].copy()
        sub = sub.sort_values(sweep_param)

        fig, ax = plt.subplots(figsize=(8, 4))
        for metric in metrics:
            ax.plot(sub[sweep_param].astype(str), sub[metric],
                    marker="o", linewidth=2, color=colors[metric],
                    label=metric.upper())
            # Annotate max
            best_idx = sub[metric].idxmax()
            ax.annotate(
                f"{sub.loc[best_idx, metric]:.3f}",
                xy=(sub.loc[best_idx, sweep_param], sub.loc[best_idx, metric]),  # type: ignore
                xytext=(0, 8), textcoords="offset points",
                ha="center", fontsize=8, color=colors[metric],
            )

        ax.set_xlabel(sweep_param)
        ax.set_ylabel("Score")
        ax.set_title(f"Sensitivity — {model_name}\n"
                     f"Sweep: {sweep_param}",
                     fontweight="bold")
        ax.legend(loc="lower right")
        ax.set_ylim(0, 1.05)
        plt.tight_layout()
        fname = fig_dir / f"sensitivity_{sweep_param}_{model_name.lower()}.png"
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved => {fname}")


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
    Sweep a parameter grid - one full training run per combination.

    Args:
        grid     : dict mapping parameter name -> list of values.
                   Grid = cartesian product of all lists.
                   Defaults to SENSITIVITY_GRID defined at top of file.
        base_cfg : base config dict. Grid values are overlaid on top.
                   Defaults to ONE_SHOT_CONFIG.
        run_shap : run SHAP at each grid point (default False for speed).
        verbose  : print progress for each run.

    Returns:
        pd.DataFrame - one row per (run x model), columns include all
        swept parameters + f1 / recall / auc_pr / mcc / run_id.
        Saved files:
          evaluation_reports/sensitivity_grid_{ts}.csv
          evaluation_reports/sensitivity_best_configs_{ts}.csv
          figures/sensitivity/{param}_{model}.png  (one per swept axis)
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

    all_rows = []

    for i, cfg in enumerate(configs):
        run_id = f"run{i:03d}"

        result = run_single(
            cfg      = cfg,
            run_id   = run_id,
            run_shap = run_shap,
            verbose  = verbose,
        )
        all_rows.extend(result["results"])
        plt.close("all")

    df_all = pd.DataFrame(all_rows)

    # Save full grid report
    grid_path = _reports_dir() / f"sensitivity_grid_{ts}.csv"
    df_all.to_csv(grid_path, index=False)
    print(f"\nFull grid report saved => {grid_path}")

    # Per-axis summary charts
    for param in grid.keys():
        if param in df_all.columns:
            plot_sensitivity_summary(df_all, sweep_param=param)

    # Best config per model
    metric_cols = ["f1","recall","precision","f2","auc_pr","mcc",
                   "accuracy","auc_roc","TP","FP","TN","FN","run_id"]
    # Config param cols = all ONE_SHOT_CONFIG keys that made it into the DataFrame
    skip_keys   = {"_run_for_models", "shap_dependence_features", "shap_waterfall_cases"}
    cfg_cols    = [k for k in ONE_SHOT_CONFIG.keys()
                   if k not in skip_keys and k in df_all.columns]
    keep_cols   = [c for c in metric_cols + cfg_cols if c in df_all.columns]
    best = (
        df_all[~df_all["model"].str.contains("Baseline")]
        .sort_values("f1", ascending=False)
        .groupby("model")
        .first()
        [keep_cols]
    )
    print(f"\n{'='*60}")
    print("BEST CONFIG PER MODEL (by F1)")
    print(f"{'='*60}")
    print(best.round(3).to_string())

    best_path = _reports_dir() / f"sensitivity_best_configs_{ts}.csv"
    best.to_csv(best_path)
    print(f"\nBest configs saved => {best_path}")

    return df_all