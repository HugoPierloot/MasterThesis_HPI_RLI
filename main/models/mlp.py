# ─────────────────────────────────────────────────────────────
#  MLP
#
#  Multi-Layer Perceptron (feed-forward neural network) for binary
#  rocket-launch failure classification. Like LOGIT and SVM, an MLP is a
#  gradient-based, weighted-sum model and therefore needs the SAME careful
#  preprocessing the tree ensembles do not.
#
#  Why an MLP needs Logit/SVM-style preprocessing:
#    1. SENTINEL VALUES (-1). The same config/derived columns use -1 to mean
#       "missing" (Price_cfg == -1 <=> Price_cfg_missing == 1). A neuron reads
#       -1 as a real input magnitude, so we replace -1 with NaN and median-
#       impute, keeping the existing *_missing indicator columns. Rows are NOT
#       deleted (Price alone is missing for ~63% of launches). Only columns
#       whose minimum is -1 are sentinel-coded; lat/lon are never touched.
#
#    2. SCALING IS MANDATORY. An MLP trained by gradient descent is extremely
#       sensitive to input scale: a feature with a large spread (e.g.
#       org_prior_launches, SD ~ 748) produces huge activations and gradients
#       that dominate training and stall the optimiser. Standardising every
#       continuous feature to mean 0 / SD 1 is what lets the network converge.
#       (Trees are scale-invariant, which is why this lives here / in logit.py
#       / in svm.py, not in splitter.py.)
#
#    3. CONTINUOUS vs CATEGORICAL vs CYCLICAL — handled identically to LOGIT:
#       continuous -> impute + standardise; binary 0/1 dummies -> left as-is;
#       launch_hour / launch_month -> sin/cos pairs. We REUSE logit.py's
#       preprocessor so the whole linear/NN family shares one definition.
#
#    4. PROBABILITIES come for free: MLPClassifier.predict_proba returns the
#       softmax/logistic output, so it plugs straight into evaluate_model(),
#       the ROC/PR curves, the confusion matrix and best_configs.
#
#  INTERPRETATION — permutation importance (magnitude) + a directional
#  sensitivity probe (sign). This is the key point for a thesis:
#    * An MLP has NO coefficients and is a non-linear black box, so neither
#      the odds-ratio reading (LOGIT) nor a clean SHAP TreeExplainer (trees)
#      applies. SHAP's KernelExplainer works but is slow on a neural net.
#    * PERMUTATION IMPORTANCE is the robust, architecture-agnostic technique:
#      shuffle one feature on the test set, measure how far the F1 score drops.
#      A large drop => the model relied on that feature. (Magnitude only — it
#      cannot say the direction of the effect.)
#    * To recover DIRECTION intuitively we add a one-at-a-time SENSITIVITY
#      PROBE: shift each feature by +/-1 SD (or flip a 0/1 dummy) across the
#      whole test set and measure how the MEAN predicted success-probability
#      moves. This gives plain-English sentences like "a +1 SD increase in X
#      raises the average predicted success probability by ~Y percentage
#      points", WITHOUT pretending the model is linear (the probe reports the
#      model's own averaged response, partial-dependence style).
#    * Optionally, SHAP KernelExplainer can be enabled for a fuller local
#      explanation, but it is OFF by default for speed.
# ─────────────────────────────────────────────────────────────

# Standard library
from datetime import datetime
from pathlib import Path

# Third-party
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
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

def build_mlp(
    feature_cols: list,
    classes: dict,
    hidden_layer_sizes=(64, 32),
    alpha: float = 1e-3,
    learning_rate_init: float = 1e-3,
    activation: str = "relu",
    max_iter: int = 500,
    early_stopping: bool = True,
    random_state: int = 42,
) -> Pipeline:
    """
    A self-contained, sklearn-compatible MLP that consumes the SAME raw numeric
    feature matrix as every other model. It does its own sentinel handling,
    median imputation, standardisation and cyclical encoding internally (shared
    with the LOGIT preprocessor), then fits an MLPClassifier. predict_proba is
    native, so it drops straight into evaluate_model(), the ROC/PR curves, the
    confusion matrix and best_configs.

    Args:
        feature_cols       : ordered list of the raw feature column names
        classes            : output of classify_feature_cols()
        hidden_layer_sizes : tuple of hidden-layer widths, e.g. (64, 32)
        alpha              : L2 regularisation strength (larger = more reg.)
        learning_rate_init : initial learning rate for the adam optimiser
        activation         : 'relu' (default), 'tanh', 'logistic'
        max_iter           : maximum training epochs
        early_stopping     : hold out 10% of train as validation and stop when
                             it stops improving (guards against overfitting on
                             this small, imbalanced dataset)
        random_state       : seed

    Note: MLP does NOT have a class_weight option. Within the shared pipeline
    SMOTE runs BEFORE the MLP, which is how the minority class is balanced;
    that is also why a few synthetic minority rows may carry blended (non -1)
    values the sentinel step cannot flag — the same minor, shared limitation
    noted for LOGIT and SVM. For a clean directional probe, the interpretation
    runner re-imputes on real data via the shared preprocessor.
    """
    pre = build_logit_preprocessor(feature_cols, classes)
    clf = MLPClassifier(
        hidden_layer_sizes = hidden_layer_sizes,
        activation         = activation,
        alpha              = alpha,
        learning_rate_init = learning_rate_init,
        max_iter           = max_iter,
        early_stopping     = early_stopping,
        n_iter_no_change   = 15,
        random_state       = random_state,
    )
    return Pipeline([("prep", pre), ("clf", clf)])


# ─────────────────────────────────────────────────────────────
#  Interpretation helpers
# ─────────────────────────────────────────────────────────────

def _mlp_fig_dir() -> Path:
    p = C.DATA_PATH_OUTPUTS_FIG / "mlp"
    p.mkdir(parents=True, exist_ok=True)
    return p


def mlp_permutation_importance(
    pipeline: Pipeline,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list,
    n_repeats: int = 10,
    max_samples: int = 400,
    metric: str = "roc_auc",
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Architecture-agnostic interpretation: permutation importance on the RAW
    feature matrix (the pipeline does its own preprocessing, so we permute the
    inputs the user understands, not the standardised internals).

    For each feature we shuffle its column n_repeats times and record how far
    the score drops. A large mean drop => the model relied on that feature.

    Scoring defaults to roc_auc, NOT f1: on this highly imbalanced, high-
    accuracy problem the MLP rarely flips a hard 0.5-threshold prediction when
    a single feature is shuffled, so an F1-based score saturates near zero and
    looks falsely uninformative. roc_auc measures reliance on the probability
    RANKING, which is far more discriminative for a probabilistic black box
    (empirically ~47/56 features show signal under roc_auc vs ~6/56 under f1).
    Pass metric="f1" to score on the thresholded decision instead.

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


def mlp_directional_sensitivity(
    pipeline: Pipeline,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list,
    classes: dict,
    max_samples: int = 600,
) -> pd.DataFrame:
    """
    Recover the DIRECTION of each feature's effect, which permutation
    importance cannot give. This is a partial-dependence-style probe on the
    RAW inputs:

      • Continuous feature: shift the whole test column by +1 SD and by -1 SD
        (SD measured on the test sample), re-predict, and record how the MEAN
        predicted success-probability changes. The reported `delta_pp` is the
        average of (effect of +1 SD) and (-1)*(effect of -1 SD), in percentage
        points — a symmetric local slope of the model's averaged response.

      • Binary / dummy feature: set the whole column to 1, then to 0, and take
        the difference in mean predicted success-probability (the average
        marginal effect of the flag being present).

    A positive delta_pp => increasing the feature raises the model's average
    predicted probability of SUCCESS; negative => lowers it. This is the
    model's own averaged response, so it stays valid for a non-linear net (we
    never claim a constant per-unit effect).

    Returns:
        DataFrame[feature, kind, delta_pp, abs_delta_pp] sorted by
        abs_delta_pp descending. delta_pp is in percentage points.
    """
    n = min(max_samples, len(X_test))
    X = np.asarray(X_test, dtype=float)[:n].copy()

    binset = set(classes.get("binary", []))
    base_p = pipeline.predict_proba(X)[:, 1].mean()

    rows = []
    for j, feat in enumerate(feature_names):
        col = X[:, j]
        if feat in binset:
            Xup = X.copy(); Xup[:, j] = 1.0
            Xdn = X.copy(); Xdn[:, j] = 0.0
            p_up = pipeline.predict_proba(Xup)[:, 1].mean()
            p_dn = pipeline.predict_proba(Xdn)[:, 1].mean()
            delta = (p_up - p_dn) * 100.0
            kind = "binary"
        else:
            sd = np.nanstd(col)
            if not np.isfinite(sd) or sd == 0:
                rows.append({"feature": feat, "kind": "continuous",
                             "delta_pp": 0.0, "abs_delta_pp": 0.0})
                continue
            Xup = X.copy(); Xup[:, j] = col + sd
            Xdn = X.copy(); Xdn[:, j] = col - sd
            p_up = pipeline.predict_proba(Xup)[:, 1].mean()
            p_dn = pipeline.predict_proba(Xdn)[:, 1].mean()
            # symmetric local slope per +1 SD, in percentage points
            delta = ((p_up - base_p) + (base_p - p_dn)) / 2.0 * 100.0
            kind = "continuous"
        rows.append({"feature": feat, "kind": kind,
                     "delta_pp": delta, "abs_delta_pp": abs(delta)})

    df = pd.DataFrame(rows)
    return df.sort_values("abs_delta_pp", ascending=False).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────
#  Plain-English interpretation report (saved as .txt)
# ─────────────────────────────────────────────────────────────

def _perm_sentence(feature: str, importance: float, std: float, rank: int,
                   metric_label: str = "ROC-AUC") -> str:
    """One plain-English sentence for a permutation-importance entry."""
    if importance <= 0:
        effect = (f"shuffling it did NOT lower the {metric_label} (and may have "
                  f"slightly raised it), so the network does not rely on it")
    else:
        effect = (f"shuffling it lowers the test {metric_label} by {importance:.3f} on "
                  f"average (+/- {1.96 * std:.3f}), so the network relies on it")
    return (
        f"  {rank}. [{feature}] — mean {metric_label} drop = {importance:+.3f}\n"
        f"     When this feature's values are randomly permuted, {effect}.\n"
    )


def _direction_sentence(feature: str, kind: str, delta_pp: float, rank: int) -> str:
    """One plain-English sentence for the directional sensitivity probe."""
    raises = delta_pp > 0
    word = "raises" if raises else "lowers"
    if kind == "binary":
        label = (feature
                 .replace("_co", " (country dummy)")
                 .replace("_cfg", " (config flag)")
                 .replace("_missing", " (spec absent)")
                 .replace("_", " "))
        move = "its presence (1 vs 0)"
    else:
        label = feature
        move = "a +1 standard-deviation increase"
    return (
        f"  {rank}. [{label}] — average effect = {delta_pp:+.2f} pp\n"
        f"     On average across the test set, {move} {word} the network's "
        f"predicted probability of SUCCESS by ~{abs(delta_pp):.2f} percentage points.\n"
    )


def generate_mlp_interpretation_report(
    pipeline: Pipeline,
    classes: dict,
    perm_df: pd.DataFrame,
    direction_df: pd.DataFrame,
    config_label: str,
    out_dir: Path,
    top_n: int = 20,
) -> Path:
    """
    Write a plain-English interpretation of the MLP as a .txt file.

    Because an MLP has no coefficients, the report combines two complementary,
    intuitive views:
      • IMPORTANCE (magnitude) from permutation importance — which features the
        network relies on.
      • DIRECTION (sign + size) from the sensitivity probe — whether increasing
        a feature raises or lowers the average predicted success probability,
        and by how many percentage points.

    Args:
        pipeline     : the fitted MLP Pipeline
        classes      : output of classify_feature_cols()
        perm_df      : output of mlp_permutation_importance()
        direction_df : output of mlp_directional_sensitivity()
        config_label : title line, e.g.
                       "MLP (best F1 config) - hidden=(64, 32), alpha=0.001"
        out_dir      : directory for the .txt file
        top_n        : number of features to describe

    Returns:
        Path to the saved .txt file.
    """
    clf = pipeline.named_steps["clf"]
    arch = getattr(clf, "hidden_layer_sizes", "?")
    act  = getattr(clf, "activation", "?")

    lines = []
    lines.append("=" * 70)
    lines.append(config_label)
    lines.append("=" * 70)
    lines.append(
        "\nThis report interprets a Multi-Layer Perceptron (MLP / neural network)\n"
        "for binary launch classification (1 = Success, 0 = Failure). All\n"
        "continuous features were standardised (z-score) before training, and the\n"
        "-1 missing sentinels were median-imputed (the *_missing indicator columns\n"
        "are kept).\n"
    )
    lines.append(f"Architecture: hidden layers = {arch}, activation = {act}\n")
    lines.append(
        "INTERPRETATION METHOD — an MLP is a non-linear black box with no\n"
        "coefficients, so it is interpreted with two model-agnostic techniques:\n"
        "  (A) PERMUTATION IMPORTANCE — shuffles each feature and measures the\n"
        "      drop in ROC-AUC (reliance on the probability ranking). Shows\n"
        "      MAGNITUDE of reliance, not direction. (ROC-AUC is used instead of\n"
        "      F1 because on this imbalanced, high-accuracy problem an F1-based\n"
        "      score saturates and looks falsely flat.)\n"
        "  (B) DIRECTIONAL SENSITIVITY PROBE — shifts each feature by +/-1 SD\n"
        "      (or flips a 0/1 flag) across the test set and measures how the\n"
        "      network's MEAN predicted success probability moves. Shows the\n"
        "      DIRECTION and approximate size, in percentage points (pp).\n"
        "\n"
        "Note: these describe the network's averaged response; unlike LOGIT there\n"
        "is no single per-unit coefficient, and the effect can differ for an\n"
        "individual launch (the net is non-linear).\n"
    )

    # ── (A) Importance ──
    lines.append("-" * 70)
    lines.append("(A) FEATURE IMPORTANCE — permutation importance (ROC-AUC drop)\n")
    lines.append("-" * 70)
    topp = perm_df.head(top_n).reset_index(drop=True)
    for i, row in topp.iterrows():
        lines.append(_perm_sentence(row["feature"], row["importance_mean"],
                                    row["importance_std"], i + 1))

    # ── (B) Direction ──
    lines.append("-" * 70)
    lines.append("(B) FEATURE DIRECTION — sensitivity probe (sign + size, pp)\n")
    lines.append("-" * 70)
    topd = direction_df.head(top_n).reset_index(drop=True)
    for i, row in topd.iterrows():
        lines.append(_direction_sentence(row["feature"], row["kind"],
                                         row["delta_pp"], i + 1))

    # ── Summary ──
    pos = direction_df[direction_df["delta_pp"] > 0].head(3)["feature"].tolist()
    neg = direction_df[direction_df["delta_pp"] < 0].head(3)["feature"].tolist()
    lines.append("-" * 70)
    lines.append("SUMMARY\n")
    lines.append(
        f"  Most relied-upon features (importance): "
        f"{', '.join(perm_df.head(3)['feature'].tolist())}\n"
        f"  Strongest push towards SUCCESS (direction): {', '.join(pos) if pos else 'n/a'}\n"
        f"  Strongest push towards FAILURE (direction): {', '.join(neg) if neg else 'n/a'}\n"
    )

    lines.append("=" * 70)
    lines.append(f"Report generated: {datetime.today().strftime('%Y-%m-%d %H:%M')}")
    lines.append("=" * 70)

    out_dir.mkdir(parents=True, exist_ok=True)
    raw_label = config_label.split(" - ")[0].strip().lower()
    safe_label = "".join(ch if ch.isalnum() else "_" for ch in raw_label)
    safe_label = "_".join(filter(None, safe_label.split("_")))
    path = out_dir / f"interpretation_{safe_label}.txt"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Interpretation report => {path}")
    return path


# ─────────────────────────────────────────────────────────────
#  Plots
# ─────────────────────────────────────────────────────────────

def plot_mlp_permutation_importance(
    perm_df: pd.DataFrame,
    model_name: str = "MLP",
    top_n: int = 20,
    out_dir: Path = None,
):
    """Horizontal bar chart of permutation importance with 95% CI."""
    df = perm_df.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, max(4, len(df) * 0.4)))
    y = np.arange(len(df))
    ax.barh(y, df["importance_mean"],
            xerr=1.96 * df["importance_std"],
            color="#9b59b6", alpha=0.85, edgecolor="white",
            error_kw=dict(ecolor="black", elinewidth=1, capsize=3))
    ax.axvline(x=0, color="grey", linestyle="--", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(df["feature"], fontsize=9)
    ax.set_xlabel("Mean ROC-AUC drop when feature is shuffled (higher = more important)")
    ax.set_title(f"Permutation importance — {model_name}", fontweight="bold")
    plt.tight_layout()
    out = (out_dir or _mlp_fig_dir()); out.mkdir(parents=True, exist_ok=True)
    path = out / f"permutation_importance_{model_name.lower()}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    print(f"  Saved => {path}")
    return path


def plot_mlp_directional_sensitivity(
    direction_df: pd.DataFrame,
    model_name: str = "MLP",
    top_n: int = 20,
    out_dir: Path = None,
):
    """Signed bar chart of the directional probe. Green = raises success prob."""
    df = direction_df.head(top_n).iloc[::-1]
    colors = ["#2ecc71" if d > 0 else "#e74c3c" for d in df["delta_pp"]]
    fig, ax = plt.subplots(figsize=(9, max(4, len(df) * 0.4)))
    y = np.arange(len(df))
    ax.barh(y, df["delta_pp"], color=colors, alpha=0.85, edgecolor="white")
    ax.axvline(x=0, color="grey", linestyle="--", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(df["feature"], fontsize=9)
    ax.set_xlabel("Δ mean predicted success probability (percentage points)\n"
                  "+1 SD increase (continuous) or presence (binary); >0 → success")
    ax.set_title(f"Directional sensitivity — {model_name}", fontweight="bold")
    plt.tight_layout()
    out = (out_dir or _mlp_fig_dir()); out.mkdir(parents=True, exist_ok=True)
    path = out / f"directional_sensitivity_{model_name.lower()}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    print(f"  Saved => {path}")
    return path


# ─────────────────────────────────────────────────────────────
#  Interpretation runner (called from render_best_configs for the best MLP)
# ─────────────────────────────────────────────────────────────

def run_mlp_interpretation(
    pipeline: Pipeline,
    split: dict,
    classes: dict,
    model_name: str = "MLP",
    top_n: int = 20,
    n_repeats: int = 10,
    out_dir: Path = None,
    config_label: str = None,
    write_report: bool = True,
    verbose: bool = True,
) -> dict:
    """
    Produce the MLP interpretation artefacts and save them to `out_dir`
    (defaults to figures/mlp/): permutation importance (magnitude) and the
    directional sensitivity probe (sign + size), each as a chart + CSV, plus a
    plain-English .txt report combining the two.

    Args:
        config_label : title for the .txt report, e.g.
                       "MLP 1 - hidden=(64, 32), alpha=0.001". Auto-generated
                       from the fitted architecture when None.
        write_report : set False to skip the .txt (charts/CSVs still saved).

    Returns dict with: permutation_importance, directional_sensitivity
    (DataFrames), report_path (Path or None).
    """
    X_test = split["X_test"]
    X_test = X_test.values if isinstance(X_test, pd.DataFrame) else X_test
    y_test = split["y_test"]
    feature_names = split["feature_cols"]

    out_dir = out_dir or _mlp_fig_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"\n[{model_name}] interpretation")
        print("    Permutation importance (architecture-agnostic)...")
    perm_df = mlp_permutation_importance(
        pipeline, X_test, y_test, feature_names, n_repeats=n_repeats,
    )
    plot_mlp_permutation_importance(perm_df, model_name, top_n=top_n, out_dir=out_dir)
    perm_df.round(5).to_csv(out_dir / f"permutation_importance_{model_name.lower()}.csv",
                            index=False)

    if verbose:
        print("    Directional sensitivity probe (+/-1 SD)...")
    direction_df = mlp_directional_sensitivity(
        pipeline, X_test, y_test, feature_names, classes,
    )
    plot_mlp_directional_sensitivity(direction_df, model_name, top_n=top_n, out_dir=out_dir)
    direction_df.round(5).to_csv(
        out_dir / f"directional_sensitivity_{model_name.lower()}.csv", index=False)

    report_path = None
    if write_report:
        if config_label is None:
            clf = pipeline.named_steps["clf"]
            config_label = (f"{model_name} - hidden={getattr(clf,'hidden_layer_sizes','?')}, "
                            f"alpha={getattr(clf,'alpha','?')}")
        report_path = generate_mlp_interpretation_report(
            pipeline, classes, perm_df, direction_df,
            config_label=config_label, out_dir=out_dir, top_n=top_n,
        )

    return {"permutation_importance": perm_df,
            "directional_sensitivity": direction_df,
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

    pipe = build_mlp(feature_cols, classes)
    pipe.fit(df_train[feature_cols].values, df_train[TARGET].values)

    split = {
        "X_test": df_test[feature_cols].values,
        "y_test": df_test[TARGET].values,
        "feature_cols": feature_cols,
        "df_train": df_train,
    }
    run_mlp_interpretation(pipe, split, classes, model_name="MLP")