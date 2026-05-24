# ─────────────────────────────────────────────────────────────
#  Evaluation
#
#  Unified evaluation for all classifiers.
#  Same metrics are computed for every model so results are directly comparable in the thesis and in saved reports.
#
#  Metrics chosen for imbalanced binary classification:
#    Accuracy        = included but flagged as misleading (91.2%
#                      baseline trap). Reported for completeness.
#    Precision       = of all predicted failures, how many were real?
#    Recall          = of all real failures, how many did we catch?
#                      This is the MOST IMPORTANT metric for this
#                      problem: a missed failure is more costly than
#                      a false alarm.
#    F1-score        = harmonic mean of precision and recall.
#                      Primary metric for model comparison.
#    F2-score        = weights recall twice vs precision
#                      (Fbeta with beta=2). Used when missing a
#                      failure is more costly than a false alarm.
#    AUC-ROC         = discrimination ability across all thresholds.
#    AUC-PR          = area under precision-recall curve.
#                      More informative than AUC-ROC for imbalanced data
#                      because it focuses on the minority class.
#    MCC             = Matthews Correlation Coefficient.
#                      Single metric that accounts for all 4 cells
#                      of the confusion matrix. Range [-1, 1].
#    Confusion Matrix = raw TP/FP/TN/FN for full picture.
# ─────────────────────────────────────────────────────────────

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)

# local imports
from constants import Constant as C


# ─────────────────────────────────────────────────────────────
#  Core metric computation
# ─────────────────────────────────────────────────────────────

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray = None,
    model_name: str = "model",
    threshold: float = 0.5,
) -> dict:
    """
    Compute all evaluation metrics for one model.

    Args:
        y_true     : true binary labels
        y_pred     : predicted binary labels
        y_proba    : predicted probabilities for class 1 (optional)
        model_name : label used in the report
        threshold  : classification threshold (default 0.5)

    Returns:
        dict of metric_name => value
    """
    metrics = {
        "model"      : model_name,
        "threshold"  : threshold,
        "accuracy"   : accuracy_score(y_true, y_pred),
        "precision"  : precision_score(y_true, y_pred, zero_division=0),
        "recall"     : recall_score(y_true, y_pred, zero_division=0),
        "f1"         : f1_score(y_true, y_pred, zero_division=0),
        "f2"         : fbeta_score(y_true, y_pred, beta=2, zero_division=0),
        "mcc"        : matthews_corrcoef(y_true, y_pred),
    }

    if y_proba is not None:
        metrics["auc_roc"] = roc_auc_score(y_true, y_proba)
        metrics["auc_pr"]  = average_precision_score(y_true, y_proba)
    else:
        metrics["auc_roc"] = np.nan
        metrics["auc_pr"]  = np.nan

    # Confusion matrix cells
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    metrics.update({"TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn)})

    return metrics


def evaluate_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str = "model",
    threshold: float = 0.5,
) -> dict:
    """
    Evaluate a fitted sklearn-compatible model on test data.
    Automatically extracts probabilities if available.
    """
    y_pred  = model.predict(X_test)
    y_proba = None

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        # SVM / linear models - convert scores to probabilities via sigmoid
        scores  = model.decision_function(X_test)
        y_proba = 1 / (1 + np.exp(-scores))

    # Apply custom threshold if not 0.5
    if threshold != 0.5 and y_proba is not None:
        y_pred = (y_proba >= threshold).astype(int)

    return compute_metrics(y_test, y_pred, y_proba, model_name, threshold)


# ─────────────────────────────────────────────────────────────
#  Comparison report
# ─────────────────────────────────────────────────────────────

def build_comparison_report(results: list[dict]) -> pd.DataFrame:
    """
    Build a DataFrame comparing all models side by side.

    Args:
        results: list of dicts from evaluate_model()

    Returns:
        pd.DataFrame sorted by F1 descending
    """
    df = pd.DataFrame(results)
    metric_cols = ["accuracy", "precision", "recall", "f1", "f2",
                   "auc_roc", "auc_pr", "mcc"]
    df = df.set_index("model")[metric_cols + ["TP","FP","TN","FN","threshold"]]
    df = df.sort_values("f1", ascending=False)
    return df


def save_report(df: pd.DataFrame, filename: str = "model_comparison.csv"):
    """Save the comparison report to DATA_PATH_OUTPUTS_EVL_REP."""
    out_dir = C.DATA_PATH_OUTPUTS_EVL_REP
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename
    df.to_csv(path)
    print(f"  Report saved => {path}")
    return path


# ─────────────────────────────────────────────────────────────
#  Visualisations
# ─────────────────────────────────────────────────────────────

FIG_DIR = None


def _get_fig_dir() -> Path:
    fig_dir = C.DATA_PATH_OUTPUTS_FIG / "models"
    fig_dir.mkdir(parents=True, exist_ok=True)
    return fig_dir


def plot_confusion_matrix(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str,
    threshold: float = 0.5,
):
    """Plot and save a labelled confusion matrix for one model."""
    y_pred = model.predict(X_test)
    if threshold != 0.5 and hasattr(model, "predict_proba"):
        y_pred = (model.predict_proba(X_test)[:, 1] >= threshold).astype(int)

    cm     = confusion_matrix(y_test, y_pred)
    labels = ["Failure (0)", "Success (1)"]

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels, ax=ax,
        linewidths=0.5,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {model_name}", fontweight="bold")
    plt.tight_layout()
    path = _get_fig_dir() / f"cm_{model_name.lower().replace(' ','_')}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    print(f"  Saved => {path}")


def plot_roc_curves(
    models_results: list[dict],
    y_test: np.ndarray,
    fitted_models: dict,
    X_test: np.ndarray,
):
    """Plot ROC curves for all models on one chart."""
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#2ecc71","#3498db","#e74c3c","#9b59b6","#f39c12","#1abc9c"]

    for i, (name, model) in enumerate(fitted_models.items()):
        if not hasattr(model, "predict_proba"):
            continue
        proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, proba)
        auc = roc_auc_score(y_test, proba)
        ax.plot(fpr, tpr, color=colors[i % len(colors)],
                linewidth=2, label=f"{name}  (AUC={auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random baseline")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate (Recall)")
    ax.set_title("ROC Curves — all models", fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    path = _get_fig_dir() / "roc_curves_all_models.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    print(f"  Saved => {path}")


def plot_precision_recall_curves(
    fitted_models: dict,
    X_test: np.ndarray,
    y_test: np.ndarray,
):
    """
    Plot Precision-Recall curves for all models.
    More informative than ROC for imbalanced data.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    colors  = ["#2ecc71","#3498db","#e74c3c","#9b59b6","#f39c12","#1abc9c"]
    baseline_precision = y_test.mean()   # random classifier

    for i, (name, model) in enumerate(fitted_models.items()):
        if not hasattr(model, "predict_proba"):
            continue
        proba = model.predict_proba(X_test)[:, 1]
        prec, rec, _ = precision_recall_curve(y_test, proba)
        auc_pr = average_precision_score(y_test, proba)
        ax.plot(rec, prec, color=colors[i % len(colors)],
                linewidth=2, label=f"{name}  (AP={auc_pr:.3f})")

    ax.axhline(y=baseline_precision, color="grey", linestyle="--",
               linewidth=1, label=f"Random baseline ({baseline_precision:.3f})")
    ax.set_xlabel("Recall (coverage of failures)")
    ax.set_ylabel("Precision (accuracy of failure alerts)")
    ax.set_title("Precision-Recall Curves — all models\n"
                 "(more informative than ROC for imbalanced data)",
                 fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    path = _get_fig_dir() / "pr_curves_all_models.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    print(f"  Saved => {path}")


def plot_metrics_comparison(df_report: pd.DataFrame):
    """
    Grouped bar chart comparing F1, Recall, Precision, AUC-PR
    across all models.
    """
    metrics_to_plot = ["precision", "recall", "f1", "f2", "auc_pr", "mcc"]
    df_plot = df_report[metrics_to_plot].copy()

    fig, ax = plt.subplots(figsize=(12, 5))
    x       = np.arange(len(df_plot))
    width   = 0.13
    colors  = ["#3498db","#2ecc71","#e74c3c","#f39c12","#9b59b6","#1abc9c"]

    for i, metric in enumerate(metrics_to_plot):
        ax.bar(x + i * width, df_plot[metric], width=width,
               label=metric.upper(), color=colors[i], alpha=0.85, edgecolor="white")

    ax.set_xticks(x + width * (len(metrics_to_plot) - 1) / 2)
    ax.set_xticklabels(df_plot.index, rotation=15, ha="right")
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title("Model comparison — key metrics\n"
                 "(focus on F1/Recall/AUC-PR for imbalanced data)",
                 fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    ax.axhline(y=0.9, color="grey", linestyle=":", linewidth=1, alpha=0.6)
    plt.tight_layout()
    path = _get_fig_dir() / "metrics_comparison_all_models.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    print(f"  Saved => {path}")