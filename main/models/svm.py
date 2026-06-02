# ─────────────────────────────────────────────────────────────
#  SVM
#
#  Support Vector Machine (SVM) for binary rocket-launch failure
#  classification. Like LOGIT, an SVM is a margin / distance-based model and
#  therefore needs the SAME careful preprocessing the tree ensembles do not.
#
#  Why an SVM needs Logit-style preprocessing (and a bit more):
#    1. SENTINEL VALUES (-1). The same config/derived columns use -1 to mean
#       "missing" (Price_cfg == -1 <=> Price_cfg_missing == 1). An SVM reads
#       -1 as a real coordinate in feature space, so the margin and the RBF
#       kernel distances would be corrupted. We replace -1 with NaN and
#       median-impute, keeping the existing *_missing indicator columns. Rows
#       are NOT deleted (Price alone is missing for ~63% of launches).
#       NB: only columns whose minimum is -1 are sentinel-coded; latitude and
#       longitude (legitimately negative) are never touched.
#
#    2. SCALING IS MANDATORY — even more critical than for LOGIT. The SVM
#       margin and the RBF / polynomial kernels are computed from raw
#       Euclidean distances between samples. A single unscaled feature with a
#       large spread (e.g. org_prior_launches, SD ~ 748) would dominate every
#       distance and the kernel would effectively ignore everything else.
#       Standardising every continuous feature to mean 0 / SD 1 puts them on a
#       common footing. (Trees are scale-invariant, which is why this lives
#       here and in logit.py, not in splitter.py.)
#
#    3. CONTINUOUS vs CATEGORICAL vs CYCLICAL — handled identically to LOGIT:
#       continuous -> impute + standardise; binary 0/1 dummies -> left as-is;
#       launch_hour / launch_month -> sin/cos pairs. We reuse logit.py's
#       preprocessor so the two linear-family models share one definition.
#
#    4. PROBABILITIES. A plain SVM outputs a signed distance to the margin
#       (decision_function), not a probability. To plug into the shared
#       ROC / PR / threshold machinery we enable Platt scaling
#       (probability=True), which fits a calibrated sigmoid on top of the SVM
#       via internal cross-validation. This is slower to train but gives a
#       genuine predict_proba. (LinearSVC, by contrast, has no predict_proba;
#       we therefore use SVC, which supports every kernel and Platt scaling.)
#
#  INTERPRETATION — permutation importance (any kernel) + coefficients (linear
#  kernel only). This is the key difference from the other models:
#    * A NON-LINEAR kernel (RBF / poly) has NO coefficients and lives in an
#      implicit high-dimensional space, so neither odds-ratio reading (LOGIT)
#      nor a clean SHAP TreeExplainer (the tree models) applies. SHAP's
#      KernelExplainer technically works but is very slow on an RBF SVM.
#    * The robust, kernel-agnostic technique is PERMUTATION IMPORTANCE: shuffle
#      one feature at a time on the test set and measure how much the F1 score
#      drops. A large drop => the model relied on that feature. This needs no
#      coefficients and works for any kernel.
#    * For the LINEAR kernel ONLY, the SVM additionally exposes a weight vector
#      `coef_`, whose magnitude (on standardised inputs) ranks feature
#      influence and whose sign gives the direction toward success/failure —
#      analogous to a LOGIT coefficient but WITHOUT the odds-ratio
#      probabilistic reading (an SVM weight is not a log-odds).
# ─────────────────────────────────────────────────────────────

# Standard library
from datetime import datetime
from pathlib import Path

# Third-party
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.inspection import permutation_importance

# Local — reuse the exact same preprocessing the LOGIT model uses
from constants import Constant as C
from splitter import TARGET, get_feature_cols
from models.logit import (
    classify_feature_cols,
    build_logit_preprocessor,
    get_transformed_feature_names,
)


# ─────────────────────────────────────────────────────────────
#  1. Integrated model (plugs into the unified pipeline / charts)
# ─────────────────────────────────────────────────────────────

def build_svm(
    feature_cols: list,
    classes: dict,
    C_reg: float = 1.0,
    kernel: str = "rbf",
    gamma="scale",
    degree: int = 3,
    class_weight="balanced",
    random_state: int = 42,
) -> Pipeline:
    """
    A self-contained, sklearn-compatible SVM that consumes the SAME raw numeric
    feature matrix as every other model. It does its own sentinel handling,
    median imputation, standardisation and cyclical encoding internally (shared
    with the LOGIT preprocessor), then fits an SVC with Platt-scaled
    probabilities so it exposes predict / predict_proba and drops straight into
    evaluate_model(), the ROC/PR curves, the confusion matrix and best_configs.

    Args:
        feature_cols : ordered list of the raw feature column names
        classes      : output of classify_feature_cols()
        C_reg        : SVM regularisation (smaller = wider margin, more reg.)
        kernel       : 'rbf' (default), 'linear', 'poly', or 'sigmoid'
        gamma        : kernel coefficient for rbf/poly/sigmoid ('scale'/'auto'/float)
        degree       : polynomial degree (only used by kernel='poly')
        class_weight : 'balanced' (default) to counter the 10:1 imbalance, or None
        random_state : seed

    Note: probability=True enables Platt scaling, which calibrates a sigmoid via
    internal 5-fold CV — this is what makes predict_proba available but also
    makes SVC noticeably slower to train than the tree models. Within the shared
    pipeline SMOTE runs BEFORE the SVM, so synthetic minority rows may carry
    blended (non -1) values the sentinel step cannot flag; this is the same
    minor, shared limitation noted for LOGIT.
    """
    pre = build_logit_preprocessor(feature_cols, classes)
    clf = SVC(
        C            = C_reg,
        kernel       = kernel,
        gamma        = gamma,
        degree       = degree,
        class_weight = class_weight,
        probability  = True,        # Platt scaling -> predict_proba
        random_state = random_state,
    )
    return Pipeline([("prep", pre), ("clf", clf)])


# ─────────────────────────────────────────────────────────────
#  Interpretation helpers
# ─────────────────────────────────────────────────────────────

def _svm_fig_dir() -> Path:
    p = C.DATA_PATH_OUTPUTS_FIG / "svm"
    p.mkdir(parents=True, exist_ok=True)
    return p


def svm_permutation_importance(
    pipeline: Pipeline,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list,
    n_repeats: int = 10,
    max_samples: int = 400,
    metric: str = "f1",
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Kernel-agnostic interpretation: permutation importance on the RAW feature
    matrix (the pipeline does its own preprocessing, so we permute the inputs
    the user actually understands, not the standardised internals).

    For each feature we shuffle its column `n_repeats` times and record how far
    the score drops. A large mean drop => the model relied on that feature.

    The test set is capped at `max_samples` rows because permutation importance
    re-scores the whole set once per feature per repeat (here it would be
    n_features * n_repeats SVM evaluations), which is expensive for a
    Platt-scaled SVM.

    Returns:
        DataFrame[feature, importance_mean, importance_std, ci_lower, ci_upper]
        sorted by importance_mean descending.
    """
    n = min(max_samples, len(X_test))
    Xs = np.asarray(X_test)[:n]
    ys = np.asarray(y_test)[:n]

    result = permutation_importance(
        pipeline, Xs, ys,
        n_repeats=n_repeats, random_state=random_state,
        scoring=metric, n_jobs=-1,
    )
    df = pd.DataFrame({
        "feature": feature_names,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    })
    df["ci_lower"] = df["importance_mean"] - 1.96 * df["importance_std"]
    df["ci_upper"] = df["importance_mean"] + 1.96 * df["importance_std"]
    return df.sort_values("importance_mean", ascending=False).reset_index(drop=True)


def svm_linear_weights(
    pipeline: Pipeline,
    classes: dict,
) -> pd.DataFrame:
    """
    Linear-kernel ONLY: extract the SVM weight vector as a signed importance.

    On standardised inputs the magnitude |w_j| ranks how strongly feature j
    pushes the decision, and the sign tells the direction: with labels
    (1 = Success, 0 = Failure), a positive weight pushes toward SUCCESS.

    IMPORTANT: an SVM weight is NOT a log-odds, so there is no odds-ratio /
    exp(w) interpretation here (that belongs to LOGIT). The weight is only a
    relative, directional influence score.

    Returns:
        DataFrame[feature, weight, abs_weight] sorted by abs_weight desc, or an
        empty DataFrame if the fitted kernel is not linear.
    """
    clf = pipeline.named_steps["clf"]
    if getattr(clf, "kernel", None) != "linear" or not hasattr(clf, "coef_"):
        return pd.DataFrame(columns=["feature", "weight", "abs_weight"])

    names = get_transformed_feature_names(classes)
    w = np.asarray(clf.coef_).ravel()
    df = pd.DataFrame({"feature": names, "weight": w, "abs_weight": np.abs(w)})
    return df.sort_values("abs_weight", ascending=False).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────
#  Plain-English interpretation report (saved as .txt)
# ─────────────────────────────────────────────────────────────

def _classify_feature_kind(feature: str, classes: dict) -> str:
    """Return 'binary', 'cyclical' or 'continuous' for a transformed feature."""
    if feature in set(classes.get("binary", [])):
        return "binary"
    if feature.endswith("_sin") or feature.endswith("_cos"):
        return "cyclical"
    return "continuous"


def _weight_sentence(feature: str, weight: float, rank: int, kind: str) -> str:
    """
    One plain-English sentence for a LINEAR-kernel SVM weight.

    The weight is a signed, relative influence on the decision function on
    standardised inputs. It is NOT a log-odds, so we deliberately avoid any
    odds-ratio / probability-multiplier phrasing (that belongs to LOGIT).
    """
    direction = "towards SUCCESS" if weight > 0 else "towards FAILURE"
    push = "raises" if weight > 0 else "lowers"

    if kind == "binary":
        label = (feature
                 .replace("_co", " (country dummy)")
                 .replace("_cfg", " (config flag)")
                 .replace("_missing", " (spec absent)")
                 .replace("_", " "))
        move = "when this flag is present (1 vs 0)"
    elif kind == "cyclical":
        label = feature.replace("_", " ")
        move = "as this cyclical (sin/cos) component increases"
    else:
        label = feature
        move = "for a +1 standard-deviation increase"

    return (
        f"  {rank}. [{label}] — weight = {weight:+.3f}\n"
        f"     The decision function moves {direction} {move}; i.e. this feature "
        f"{push} the model's confidence that the launch SUCCEEDS.\n"
        f"     |weight| = {abs(weight):.3f} ranks its relative influence "
        f"(larger magnitude = stronger pull on the margin).\n"
    )


def _perm_sentence(feature: str, importance: float, std: float, rank: int) -> str:
    """One plain-English sentence for a permutation-importance entry."""
    if importance <= 0:
        effect = ("shuffling it did NOT hurt (and may have slightly helped) the "
                  "F1 score, so the model does not rely on it")
    else:
        effect = (f"shuffling it lowers the test F1 score by {importance:.3f} on "
                  f"average (+/- {1.96 * std:.3f}), so the model relies on it")
    return (
        f"  {rank}. [{feature}] — mean F1 drop = {importance:+.3f}\n"
        f"     When this feature's values are randomly permuted, {effect}.\n"
    )


def generate_svm_interpretation_report(
    pipeline: Pipeline,
    classes: dict,
    perm_df: pd.DataFrame,
    weights_df: pd.DataFrame,
    config_label: str,
    out_dir: Path,
    top_n: int = 20,
) -> Path:
    """
    Write a plain-English interpretation of the SVM as a .txt file.

    The content depends on the kernel of the fitted model:

      • LINEAR kernel  -> the model exposes a signed weight per feature. The
        report explains each weight's direction (towards success / failure) and
        relative magnitude, and ALSO lists permutation importance as a
        cross-check. It states explicitly that an SVM weight is not a log-odds
        (no odds-ratio reading).

      • NON-LINEAR kernel (rbf / poly / sigmoid) -> there are NO coefficients.
        The report says so and interprets the model purely via permutation
        importance (the kernel-agnostic technique).

    Args:
        pipeline     : the fitted SVM Pipeline
        classes      : output of classify_feature_cols()
        perm_df      : output of svm_permutation_importance()
        weights_df   : output of svm_linear_weights() (empty if non-linear)
        config_label : title line, e.g. "SVM 1 - svm_C=1.0, svm_kernel=linear"
        out_dir      : directory for the .txt file
        top_n        : number of features to describe

    Returns:
        Path to the saved .txt file.
    """
    clf = pipeline.named_steps["clf"]
    kernel = getattr(clf, "kernel", "rbf")
    is_linear = (kernel == "linear") and not weights_df.empty

    lines = []
    lines.append("=" * 70)
    lines.append(config_label)
    lines.append("=" * 70)
    lines.append(
        "\nThis report interprets a Support Vector Machine (SVM) for binary launch\n"
        "classification (1 = Success, 0 = Failure). All continuous features were\n"
        "standardised (z-score) before fitting, and the -1 missing sentinels were\n"
        "median-imputed (the *_missing indicator columns are kept).\n"
    )
    lines.append(f"Kernel: {kernel}\n")

    if is_linear:
        lines.append(
            "INTERPRETATION METHOD — signed weights (linear kernel).\n"
            "A linear SVM exposes one weight per feature. On standardised inputs the\n"
            "MAGNITUDE |w| ranks how strongly a feature pulls the decision boundary,\n"
            "and the SIGN gives the direction: a positive weight pushes the model\n"
            "towards predicting SUCCESS, a negative weight towards FAILURE.\n"
            "\n"
            "IMPORTANT: an SVM weight is NOT a log-odds. Unlike the LOGIT model there\n"
            "is no odds-ratio / exp(w) reading — the weight is only a relative,\n"
            "directional influence score, not a probability multiplier.\n"
        )
        lines.append("-" * 70)
        lines.append("FEATURE WEIGHTS (sorted by absolute influence, strongest first)\n")
        lines.append("-" * 70)
        topw = weights_df.head(top_n).reset_index(drop=True)
        for i, row in topw.iterrows():
            kind = _classify_feature_kind(row["feature"], classes)
            lines.append(_weight_sentence(row["feature"], row["weight"], i + 1, kind))

        # Summary: strongest pulls in each direction
        pos = weights_df[weights_df["weight"] > 0].head(3)["feature"].tolist()
        neg = weights_df[weights_df["weight"] < 0].head(3)["feature"].tolist()
        lines.append("-" * 70)
        lines.append("SUMMARY\n")
        lines.append(
            f"  Strongest pull towards SUCCESS: {', '.join(pos) if pos else 'n/a'}\n"
            f"  Strongest pull towards FAILURE: {', '.join(neg) if neg else 'n/a'}\n"
        )
        lines.append("-" * 70)
        lines.append("CROSS-CHECK — permutation importance (model-agnostic)\n")
        lines.append("-" * 70)
        topp = perm_df.head(min(10, top_n)).reset_index(drop=True)
        for i, row in topp.iterrows():
            lines.append(_perm_sentence(row["feature"], row["importance_mean"],
                                        row["importance_std"], i + 1))
    else:
        lines.append(
            "INTERPRETATION METHOD — permutation importance (non-linear kernel).\n"
            f"A '{kernel}' kernel maps the data into an implicit high-dimensional space,\n"
            "so the model has NO per-feature coefficients to read directly. The\n"
            "kernel-agnostic way to interpret it is permutation importance: each\n"
            "feature is randomly shuffled on the test set and we measure how much the\n"
            "F1 score drops. A large drop means the model relied on that feature.\n"
            "\n"
            "Note: permutation importance shows MAGNITUDE of reliance, not direction —\n"
            "it cannot say whether a feature pushes towards success or failure. For a\n"
            "signed, directional reading, refit the SVM with kernel='linear'.\n"
        )
        lines.append("-" * 70)
        lines.append("FEATURE IMPORTANCE (sorted by mean F1 drop, strongest first)\n")
        lines.append("-" * 70)
        topp = perm_df.head(top_n).reset_index(drop=True)
        for i, row in topp.iterrows():
            lines.append(_perm_sentence(row["feature"], row["importance_mean"],
                                        row["importance_std"], i + 1))

    lines.append("=" * 70)
    lines.append(f"Report generated: {datetime.today().strftime('%Y-%m-%d %H:%M')}")
    lines.append("=" * 70)

    out_dir.mkdir(parents=True, exist_ok=True)
    raw_label = config_label.split(" - ")[0].strip().lower()
    safe_label = "".join(ch if ch.isalnum() else "_" for ch in raw_label)
    safe_label = "_".join(filter(None, safe_label.split("_")))  # collapse repeats
    path = out_dir / f"interpretation_{safe_label}.txt"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Interpretation report => {path}")
    return path


# ─────────────────────────────────────────────────────────────
#  Plots
# ─────────────────────────────────────────────────────────────

def plot_svm_permutation_importance(
    perm_df: pd.DataFrame,
    model_name: str = "SVM",
    top_n: int = 20,
    out_dir: Path = None,
):
    """Horizontal bar chart of permutation importance with 95% CI."""
    df = perm_df.head(top_n).iloc[::-1]   # largest at top
    fig, ax = plt.subplots(figsize=(9, max(4, len(df) * 0.4)))
    y = np.arange(len(df))
    ax.barh(y, df["importance_mean"],
            xerr=1.96 * df["importance_std"],
            color="#3498db", alpha=0.85, edgecolor="white",
            error_kw=dict(ecolor="black", elinewidth=1, capsize=3))
    ax.axvline(x=0, color="grey", linestyle="--", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(df["feature"], fontsize=9)
    ax.set_xlabel("Mean F1 drop when feature is shuffled (higher = more important)")
    ax.set_title(f"Permutation importance — {model_name}", fontweight="bold")
    plt.tight_layout()
    out = (out_dir or _svm_fig_dir())
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"permutation_importance_{model_name.lower()}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    print(f"  Saved => {path}")
    return path


def plot_svm_linear_weights(
    weights_df: pd.DataFrame,
    model_name: str = "SVM",
    top_n: int = 20,
    out_dir: Path = None,
):
    """Signed weight bar chart (linear kernel only). Green = toward success."""
    if weights_df.empty:
        print("  [SVM] Non-linear kernel — no coefficients to plot "
              "(use permutation importance instead).")
        return None
    df = weights_df.head(top_n).iloc[::-1]
    colors = ["#2ecc71" if w > 0 else "#e74c3c" for w in df["weight"]]
    fig, ax = plt.subplots(figsize=(9, max(4, len(df) * 0.4)))
    y = np.arange(len(df))
    ax.barh(y, df["weight"], color=colors, alpha=0.85, edgecolor="white")
    ax.axvline(x=0, color="grey", linestyle="--", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(df["feature"], fontsize=9)
    ax.set_xlabel("SVM weight (standardised inputs; >0 pushes toward SUCCESS)")
    ax.set_title(f"Linear SVM weights — {model_name}", fontweight="bold")
    plt.tight_layout()
    out = (out_dir or _svm_fig_dir())
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"linear_weights_{model_name.lower()}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    print(f"  Saved => {path}")
    return path


# ─────────────────────────────────────────────────────────────
#  Interpretation runner (called from render_best_configs for the best SVM)
# ─────────────────────────────────────────────────────────────

def run_svm_interpretation(
    pipeline: Pipeline,
    split: dict,
    classes: dict,
    model_name: str = "SVM",
    top_n: int = 20,
    n_repeats: int = 10,
    out_dir: Path = None,
    config_label: str = None,
    write_report: bool = True,
    verbose: bool = True,
) -> dict:
    """
    Produce the SVM interpretation artefacts and save them to `out_dir`
    (defaults to figures/svm/). Always runs permutation importance; if the
    fitted kernel is linear, also saves the signed-weight chart and CSV. When
    `write_report` is True, also writes a plain-English .txt interpreting the
    coefficients (linear) or the permutation importance (non-linear).

    Args:
        config_label : title for the .txt report, e.g.
                       "SVM 1 - svm_C=1.0, svm_kernel=linear". Auto-generated
                       from the fitted kernel/C when None.
        write_report : set False to skip the .txt (charts/CSVs still saved).

    Returns dict with: permutation_importance (DataFrame), linear_weights
    (DataFrame, possibly empty), report_path (Path or None).
    """
    X_test = split["X_test"]
    X_test = X_test.values if isinstance(X_test, pd.DataFrame) else X_test
    y_test = split["y_test"]
    feature_names = split["feature_cols"]

    out_dir = out_dir or _svm_fig_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"\n[{model_name}] interpretation")
        print("    Permutation importance (kernel-agnostic)...")
    perm_df = svm_permutation_importance(
        pipeline, X_test, y_test, feature_names,
        n_repeats=n_repeats,
    )
    plot_svm_permutation_importance(perm_df, model_name, top_n=top_n, out_dir=out_dir)
    perm_df.round(5).to_csv(out_dir / f"permutation_importance_{model_name.lower()}.csv",
                            index=False)

    weights_df = svm_linear_weights(pipeline, classes)
    if not weights_df.empty:
        if verbose:
            print("    Linear kernel detected -> saving signed weights...")
        plot_svm_linear_weights(weights_df, model_name, top_n=top_n, out_dir=out_dir)
        weights_df.round(5).to_csv(out_dir / f"linear_weights_{model_name.lower()}.csv",
                                   index=False)
    elif verbose:
        print("    Non-linear kernel -> coefficients not available "
              "(permutation importance is the interpretation).")

    report_path = None
    if write_report:
        if config_label is None:
            clf = pipeline.named_steps["clf"]
            config_label = (f"{model_name} - svm_C={getattr(clf, 'C', '?')}, "
                            f"svm_kernel={getattr(clf, 'kernel', '?')}")
        report_path = generate_svm_interpretation_report(
            pipeline, classes, perm_df, weights_df,
            config_label=config_label, out_dir=out_dir, top_n=top_n,
        )

    return {"permutation_importance": perm_df, "linear_weights": weights_df,
            "report_path": report_path}


# ─────────────────────────────────────────────────────────────
#  Standalone smoke test
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = pd.read_csv(C.DATA_PATH_INPUTS_CLEAN / C.MODEL_DATA_FILENAME)
    feature_cols = get_feature_cols(df)
    classes = classify_feature_cols(df, feature_cols)

    df_train = df[df["Launch Year"] < 2010]
    df_test  = df[df["Launch Year"] >= 2010]

    pipe = build_svm(feature_cols, classes, kernel="rbf")
    pipe.fit(df_train[feature_cols].values, df_train[TARGET].values)

    split = {
        "X_test": df_test[feature_cols].values,
        "y_test": df_test[TARGET].values,
        "feature_cols": feature_cols,
    }
    run_svm_interpretation(pipe, split, classes, model_name="SVM")