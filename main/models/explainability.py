# ─────────────────────────────────────────────────────────────
#  Explainability
#
#  SHAP-based explainability for all trained classifiers.
#
#  SHAP (SHapley Additive exPlanations) by Lundberg & Lee (2017)
#  computes each feature's contribution to each individual
#  prediction using game-theory Shapley values.
#
#  Note on sentinel -1 values:
#    Several features use -1 to encode "missing data" (e.g. Price_cfg
#    is -1 for 63% of rows). SHAP will correctly attribute influence
#    to those features when -1 causes predictions to shift — so a
#    large SHAP value for Price_cfg_missing or Price_cfg=-1 means
#    "the absence of price information itself is predictive".
# ─────────────────────────────────────────────────────────────

# Standard library imports
from datetime import datetime

# Third-party imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import shap
from pathlib import Path

# Local imports
from constants import Constant as C


# ─────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────

def _fig_dir() -> Path:
    p = C.DATA_PATH_OUTPUTS_FIG / "shap"
    p.mkdir(parents=True, exist_ok=True)
    return p

def _rep_dir() -> Path:
    p = C.DATA_PATH_OUTPUTS_EVL_REP
    p.mkdir(parents=True, exist_ok=True)
    return p

def _save(fig, name: str):
    path = _fig_dir() / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved => {path}")


# ─────────────────────────────────────────────────────────────
#  Explainer factory
# ─────────────────────────────────────────────────────────────

TREE_TYPES = (
    "RandomForestClassifier",
    "XGBClassifier",
    "AdaBoostClassifier",
    "GradientBoostingClassifier",
)

def _is_tree(model) -> bool:
    class_name = type(model).__name__
    # RUSBoost wraps AdaBoost internally
    return (class_name in TREE_TYPES or
            "Boost" in class_name or
            "Forest" in class_name)

def build_explainer(
    model,
    X_background: np.ndarray,
    feature_names: list,
    n_background: int = 200,
):
    """
    Build the appropriate SHAP explainer for the given model.

    Args:
        model        : fitted sklearn-compatible estimator
        X_background : training data (numpy array) — used as background
                       for TreeExplainer / KernelExplainer
        feature_names: list of column names (for labelling)
        n_background : number of background samples for KernelExplainer

    Returns:
        shap.Explainer or shap.TreeExplainer instance
    """
    if _is_tree(model):
        # TreeExplainer: exact and fast for tree-based models
        explainer = shap.TreeExplainer(
            model,
            data=shap.sample(X_background, min(500, len(X_background))),
            feature_perturbation="interventional",
        )
    else:
        # KernelExplainer more slow
        background = shap.kmeans(X_background, min(n_background, 50))
        explainer  = shap.KernelExplainer(
            model.predict_proba if hasattr(model, "predict_proba")
            else model.predict,
            background,
        )
    return explainer


def compute_shap_values(
    explainer,
    X_test: np.ndarray,
    model_name: str,
    max_samples: int = 500,
) -> np.ndarray:
    X = X_test[:max_samples]
    print(f"    Computing SHAP for {model_name} on {len(X)} samples...",
          end=" ", flush=True)
    raw = explainer.shap_values(X)
    # OLD SHAP API
    if isinstance(raw, list):
        if len(raw) == 2:
            sv = raw[1]   # class 1
        else:
            sv = raw[0]
    # NEW SHAP API => ndarray
    elif isinstance(raw, np.ndarray):
        # Binary/multiclass classifier:
        # shape = (samples, features, classes)
        if raw.ndim == 3:
            sv = raw[:, :, 1]
        # Standard regression/binary output
        elif raw.ndim == 2:
            sv = raw
        else:
            raise ValueError(
                f"Unexpected SHAP shape: {raw.shape}"
            )
    else:
        raise ValueError(
            f"Unsupported SHAP output type: {type(raw)}"
        )
    print("done")
    print("    SHAP shape:", sv.shape)
    return np.array(sv)


# ─────────────────────────────────────────────────────────────
#  Individual SHAP plots
# ─────────────────────────────────────────────────────────────

def plot_beeswarm(
    shap_values: np.ndarray,
    X_test: np.ndarray,
    feature_names: list,
    model_name: str,
    max_display: int = 20,
):
    """
    Beeswarm (summary) plot: best overview of feature importance
    AND direction of effect for all features simultaneously.

    How to read:
      - Y axis     : features ranked by mean |SHAP| (most important at top)
      - X axis     : SHAP value (positive = pushes toward Success prediction)
      - Dot colour : feature value (red=high, blue=low)
      - Dot spread : distribution across all test samples
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    shap.summary_plot(
        shap_values,
        X_test[:len(shap_values)],
        feature_names=feature_names,
        max_display=max_display,
        show=False,
        plot_type="dot",
    )
    plt.title(f"SHAP Beeswarm — {model_name}\n"
              f"(positive SHAP = pushes toward Success prediction)",
              fontsize=11, fontweight="bold", pad=10)
    plt.tight_layout()
    _save(plt.gcf(), f"shap_beeswarm_{model_name.lower()}")


def plot_bar_importance(
    shap_values: np.ndarray,
    feature_names: list,
    model_name: str,
    top_n: int = 20,
):
    """
    Bar chart of mean |SHAP| per feature.
    Pure importance, then no direction information.
    Use alongside beeswarm to get the full picture.
    """
    mean_abs = np.abs(shap_values).mean(axis=0)
    if mean_abs.ndim != 1:
        raise ValueError( f"Expected 1D feature importance vector, got shape {mean_abs.shape}" )   # force 1-D numpy
    idx      = np.array(np.argsort(mean_abs)[-top_n:]).flatten().astype(int)  # force int
    feat     = [feature_names[i] for i in idx]
    vals     = mean_abs[idx]

    fig, ax = plt.subplots(figsize=(9, max(5, top_n * 0.38)))
    ax.barh(feat, vals, color="#3498db", alpha=0.85, edgecolor="white")
    ax.set_xlabel("Mean |SHAP value|  (average impact on prediction)")
    ax.set_title(f"SHAP Feature Importance — {model_name}\n"
                 f"(top {top_n} features by mean absolute SHAP)",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    _save(fig, f"shap_importance_{model_name.lower()}")


def plot_waterfall_single(
    explainer,
    X_test: np.ndarray,
    feature_names: list,
    model_name: str,
    sample_idx: int = 0,
    label: str = "sample",
):
    """
    Waterfall plot for ONE individual prediction.
    Shows exactly why the model predicted what it did for that launch.
    Useful for case-study analysis in your thesis.

    Args:
        sample_idx : index in X_test of the launch to explain
        label      : description of the launch (e.g. 'first failure in test set')
    """
    sv = explainer(X_test[[sample_idx]])
    if hasattr(sv, "values") and sv.values.ndim == 3:
        # Binary classifier returns (n, features, 2) — take class 1
        explanation = shap.Explanation(
            values    = sv.values[0, :, 1],
            base_values = sv.base_values[0, 1],
            data      = sv.data[0],
            feature_names = feature_names,
        )
    else:
        explanation = sv[0]

    fig = plt.figure(figsize=(10, 6))
    shap.waterfall_plot(explanation, max_display=15, show=False)
    plt.title(f"SHAP Waterfall — {model_name}\n({label})",
              fontsize=11, fontweight="bold")
    plt.tight_layout()
    _save(fig, f"shap_waterfall_{model_name.lower()}_{label.replace(' ','_')}")


def plot_dependence(
    shap_values: np.ndarray,
    X_test: np.ndarray,
    feature_names: list,
    model_name: str,
    feature: str,
    interaction_feature: str = "auto",
):
    """
    Dependence plot: SHAP value of one feature vs its raw value.
    Reveals non-linear effects and interactions.
    """
    feat_idx = feature_names.index(feature)
    fig, ax  = plt.subplots(figsize=(8, 5))
    shap.dependence_plot(
        feat_idx,
        shap_values,
        X_test[:len(shap_values)],
        feature_names=feature_names,
        interaction_index=interaction_feature,
        ax=ax,
        show=False,
    )
    ax.set_title(f"SHAP Dependence — {feature}\n({model_name})",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    fname = f"shap_dependence_{model_name.lower()}_{feature.replace(' ','_')[:30]}"
    _save(fig, fname)


# ─────────────────────────────────────────────────────────────
#  Cross-model comparison
# ─────────────────────────────────────────────────────────────

def plot_feature_importance_comparison(
    shap_dict: dict,
    feature_names: list,
    top_n: int = 15,
):
    """
    Side-by-side bar chart comparing mean |SHAP| rankings across
    all models. Reveals which features are robustly important
    (high in all models) vs model-specific.

    Args:
        shap_dict : {model_name: shap_values_array}
    """
    importance_df = pd.DataFrame(
        {name: np.abs(sv).mean(axis=0) for name, sv in shap_dict.items()},
        index=feature_names,
    )
    # Rank by average importance across models
    importance_df["mean_rank"] = importance_df.mean(axis=1)
    top_features = importance_df.nlargest(top_n, "mean_rank").index.tolist()
    plot_df = importance_df.loc[top_features].drop(columns="mean_rank")

    fig, ax = plt.subplots(figsize=(12, max(6, top_n * 0.45)))
    colors  = ["#3498db","#2ecc71","#e74c3c","#9b59b6","#f39c12"]
    x       = np.arange(len(top_features))
    n_models = len(plot_df.columns)
    width   = 0.8 / n_models

    for i, col in enumerate(plot_df.columns):
        ax.barh(x + i * width, plot_df[col].values,
                height=width, label=col,
                color=colors[i % len(colors)], alpha=0.85, edgecolor="white")

    ax.set_yticks(x + width * (n_models - 1) / 2)
    ax.set_yticklabels(top_features, fontsize=9)
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(f"SHAP Feature Importance — Cross-model comparison\n"
                 f"(top {top_n} features by average rank)",
                 fontsize=11, fontweight="bold")
    ax.legend(loc="lower right")
    plt.tight_layout()
    _save(fig, "shap_importance_comparison_all_models")


def save_shap_report(shap_dict: dict, feature_names: list, filename: str = "shap_report.csv"):
    """
    Save mean |SHAP| per feature per model as a CSV.
    Enables quantitative comparison in your thesis tables.
    """
    df = pd.DataFrame(
        {name: np.abs(sv).mean(axis=0) for name, sv in shap_dict.items()},
        index=feature_names,
    )
    df["mean_across_models"] = df.mean(axis=1)
    df = df.sort_values("mean_across_models", ascending=False)
    path = _rep_dir() / filename
    df.to_csv(path)
    print(f"    SHAP report saved => {path}")
    return df


# ─────────────────────────────────────────────────────────────
#  Statistical significance of features (permutation importance)
# ─────────────────────────────────────────────────────────────

def permutation_importance_test(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list,
    metric: str = "f1",
    n_repeats: int = 30,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Permutation feature importance with confidence intervals.

    For each feature: shuffle its values randomly n_repeats times,
    measure how much model performance drops on average.
    A feature that does NOT matter will show near-zero drop.
    A feature that matters greatly will show a large drop.

    This is the closest approximation to statistical significance
    for black-box models (no p-value from a coefficient like logistic
    regression, but the CI overlap with zero serves the same purpose).

    Args:
        model        : fitted estimator
        X_test       : test features (numpy array)
        y_test       : test labels
        feature_names: list of column names
        metric       : 'f1', 'recall', 'precision', or 'accuracy'
        n_repeats    : number of permutations per feature
        random_state : seed

    Returns:
        pd.DataFrame with columns:
            feature, mean_drop, std_drop, ci_lower, ci_upper, significant
        sorted by mean_drop descending.
        'significant' = True when ci_lower > 0
        (performance drop is reliably above zero across all shuffles)
    """
    from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score

    metric_fn = {
        "f1"       : lambda y, yp: f1_score(y, yp, zero_division=0),
        "recall"   : lambda y, yp: recall_score(y, yp, zero_division=0),
        "precision": lambda y, yp: precision_score(y, yp, zero_division=0),
        "accuracy" : accuracy_score,
    }[metric]

    baseline_score = metric_fn(y_test, model.predict(X_test))

    rng     = np.random.default_rng(random_state)
    records = []

    for j, feat in enumerate(feature_names):
        drops = []
        for _ in range(n_repeats):
            X_perm      = X_test.copy()
            X_perm[:, j] = rng.permutation(X_perm[:, j])
            perm_score  = metric_fn(y_test, model.predict(X_perm))
            drops.append(baseline_score - perm_score)

        drops    = np.array(drops)
        mean_d   = drops.mean()
        std_d    = drops.std()
        ci_lower = mean_d - 1.96 * std_d / np.sqrt(n_repeats)
        ci_upper = mean_d + 1.96 * std_d / np.sqrt(n_repeats)

        records.append({
            "feature"    : feat,
            "mean_drop"  : round(mean_d,   4),
            "std_drop"   : round(std_d,    4),
            "ci_lower"   : round(ci_lower, 4),
            "ci_upper"   : round(ci_upper, 4),
            "significant": bool(ci_lower > 0),
        })

    df = pd.DataFrame(records).sort_values("mean_drop", ascending=False)
    path = _rep_dir() / f"permutation_importance_{type(model).__name__}.csv"
    df.to_csv(path, index=False)
    print(f"    Permutation importance saved => {path}")
    return df


def plot_permutation_importance(df: pd.DataFrame, model_name: str, top_n: int = 20):
    """
    Plot permutation importance with 95% confidence intervals.
    Features where CI includes zero are NOT statistically significant.
    """
    df_plot = df.head(top_n).iloc[::-1]   # reverse for horizontal bar

    fig, ax = plt.subplots(figsize=(9, max(5, top_n * 0.38)))

    colors = ["#2ecc71" if s else "#e74c3c" for s in df_plot["significant"]]
    ax.barh(df_plot["feature"], df_plot["mean_drop"],
            color=colors, alpha=0.85, edgecolor="white")

    # 95% CI error bars
    xerr = np.array([
        df_plot["mean_drop"] - df_plot["ci_lower"],
        df_plot["ci_upper"]  - df_plot["mean_drop"],
    ])
    ax.errorbar(df_plot["mean_drop"], df_plot["feature"],
                xerr=xerr, fmt="none", color="black",
                linewidth=1.2, capsize=3)

    ax.axvline(x=0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel(f"Mean drop in F1-score when feature is shuffled\n"
                  f"(positive = feature matters; CI crossing 0 = not significant)")
    ax.set_title(f"Permutation Feature Importance — {model_name}\n"
                 f"(green = significant at 95% CI, red = not significant)",
                 fontsize=11, fontweight="bold")

    green_patch = mpatches.Patch(color="#2ecc71", label="Significant (CI > 0)")
    red_patch   = mpatches.Patch(color="#e74c3c", label="Not significant (CI includes 0)")
    ax.legend(handles=[green_patch, red_patch], loc="lower right")

    plt.tight_layout()
    _save(fig, f"permutation_importance_{model_name.lower()}")


# ─────────────────────────────────────────────────────────────
#  Master runner
# ─────────────────────────────────────────────────────────────

def run_explainability(
    fitted_models  : dict,
    split          : dict,
    top_n          : int  = 20,
    waterfall_cases: int  = 3,
    dependence_features: list = None,
    run_permutation: bool = True,
) -> dict:
    """
    Run full SHAP + permutation importance analysis for all models.

    Args:
        fitted_models       : {name: fitted estimator} from train.py
        split               : dict from make_split()
        top_n               : top N features to display
        waterfall_cases     : number of individual predictions to explain
        dependence_features : features for dependence plots (default: top 3)
        run_permutation     : also compute permutation importance

    Returns:
        dict {model_name: shap_values_array}
    """
    X_train    = split["X_train_raw"].values   # pre-SMOTE original distribution
    X_test     = split["X_test"].values if isinstance(split["X_test"], pd.DataFrame) \
                 else split["X_test"]
    y_test     = split["y_test"]
    feat_names = split["feature_cols"]

    # Default dependence features if not provided
    if dependence_features is None:
        dependence_features = [
            "rocket_prior_launches",
            "org_prior_success_rate",
            "Launch Year",
        ]
    dependence_features = [f for f in dependence_features if f in feat_names]

    shap_dict   = {}
    skip_models = {"Baseline_Majority", "Baseline_PriorRate"}

    print("=" * 60)
    print("EXPLAINABILITY PIPELINE (SHAP)")
    print("=" * 60)

    for name, model in fitted_models.items():
        if name in skip_models:
            continue

        print(f"\n[{name}]")

        # Build explainer
        print("    Building explainer...")
        try:
            explainer = build_explainer(model, X_train, feat_names)
        except Exception as e:
            print(f"    [WARNING] Could not build explainer: {e}")
            continue

        # Compute SHAP values
        sv = compute_shap_values(explainer, X_test, name)
        shap_dict[name] = sv

        # ── Per-model plots ──
        print("    Plotting beeswarm...")
        plot_beeswarm(sv, X_test, feat_names, name, max_display=top_n)

        print("    Plotting bar importance...")
        plot_bar_importance(sv, feat_names, name, top_n=top_n)

        # Waterfall: explain a few individual predictions
        # Pick first N failures in test set for interpretability
        failure_indices = np.where(y_test == 0)[0][:waterfall_cases]
        for i, idx in enumerate(failure_indices):
            print(f"    Waterfall for failure sample {i+1}...")
            try:
                plot_waterfall_single(
                    explainer, X_test, feat_names, name,
                    sample_idx=idx,
                    label=f"failure_{i+1}",
                )
            except Exception as e:
                print(f"    [WARNING] Waterfall failed: {e}")

        # Dependence plots for key features
        for feat in dependence_features:
            print(f"    Dependence plot: {feat}...")
            try:
                plot_dependence(sv, X_test, feat_names, name, feat)
            except Exception as e:
                print(f"    [WARNING] Dependence failed for {feat}: {e}")

        # Permutation importance
        if run_permutation:
            print("    Permutation importance...")
            perm_df = permutation_importance_test(
                model, X_test, y_test, feat_names,
                metric="f1", n_repeats=30,
            )
            plot_permutation_importance(perm_df, name, top_n=top_n)

    # Cross-model comparison
    if len(shap_dict) >= 2:
        print("\n[Cross-model comparison]")
        plot_feature_importance_comparison(shap_dict, feat_names, top_n=top_n)
        ts = datetime.today().strftime("%Y%m%d_%Hh%M")
        save_shap_report(shap_dict, feat_names, f"shap_report_{ts}.csv")

    print("\n" + "=" * 60)
    print("Explainability complete.")
    print(f"  Figures => {_fig_dir()}")
    print(f"  Reports => {_rep_dir()}")
    print("=" * 60)

    return shap_dict