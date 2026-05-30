# ─────────────────────────────────────────────────────────────
#  Logit
#
#  Logistic Regression (LOGIT) — a LINEAR classifier for rocket launch
#  failure prediction. Unlike the tree ensembles, a linear model is very
#  sensitive to how the features are encoded and scaled, so this module
#  carries its own preprocessing.
#
#  Why linear models need extra care here:
#    1. SENTINEL VALUES (-1). Several config/derived columns use -1 to mean
#       "missing" (e.g. Price_cfg == -1  <=>  Price_cfg_missing == 1). A tree
#       can split on -1 harmlessly, but a linear model reads -1 as a real
#       magnitude on the same axis as a genuine price/thrust — which is
#       nonsense and corrupts the coefficient. We therefore replace each -1
#       with NaN and median-impute it. We do NOT delete rows (Price alone is
#       missing for ~63% of launches — deletion would destroy the dataset),
#       and we KEEP the existing *_missing indicator columns so the model can
#       still learn "the absence of the spec is itself predictive". This is
#       the textbook "impute + missing-indicator" pattern, and the indicators
#       already exist in model_data.csv.
#       NB: -1 is only a sentinel for columns whose minimum is -1; latitude
#       and longitude legitimately go negative and are NEVER touched.
#
#    2. CONTINUOUS vs CATEGORICAL.
#         - Continuous (prices, masses, rates, counts): median-impute, then
#           standardise (z-score) so each coefficient is "per 1 SD".
#         - Binary / dummy (0/1 flags, one-hot country, *_missing): left as
#           0/1 and NOT scaled — scaling a dummy destroys the clean
#           "presence vs absence" reading of its coefficient.
#         - Cyclical (launch_hour 0-23, launch_month 1-12): a linear model
#           must not treat month 12 as "12x" month 1, so these are encoded as
#           sin/cos pairs (the correct linear treatment of a cycle).
#
#    3. SCALING is required for LOGIT (the L2 penalty is scale-sensitive and
#       standardised coefficients are comparable). It is NOT needed for the
#       tree models, which is why scaling lives here and not in splitter.py.
#
#    4. FEATURE SELECTION. With ~50 candidate features most are not
#       statistically relevant. We prune extreme multicollinearity (VIF) and
#       then run backward elimination on the Wald p-value until every kept
#       feature is significant at p < 0.05.
#
#  INTERPRETATION — coefficients (odds ratios), not SHAP.
#    For a linear model the coefficients ARE the explanation, so odds-ratio
#    interpretation is preferred over SHAP (SHAP would be largely redundant:
#    for a linear model the SHAP value of feature j is ~ coef_j * (x_j - E[x_j])).
#    A coefficient b on a feature means the log-odds of SUCCESS change by b for
#    a one-unit increase in that feature (one standard deviation for the
#    standardised continuous features; presence-vs-absence for the 0/1 dummies),
#    holding everything else fixed. exp(b) is the ODDS RATIO.
#
#    Worked (fake) example:
#       Suppose the fitted coefficient for the standardised feature
#       `org_prior_success_rate` is  b = +0.69.
#         odds ratio = exp(0.69) ≈ 2.0
#       => each +1 SD in an organisation's prior success rate roughly DOUBLES
#          the odds of the launch succeeding, all else equal.
#       Suppose the dummy `rocket_is_maiden_flight` has  b = -0.51.
#         odds ratio = exp(-0.51) ≈ 0.60
#       => a maiden flight (1 vs 0) multiplies the odds of success by ~0.60,
#          i.e. about a 40% reduction in the odds, all else equal.
#       The Wald p-value tests b = 0 (odds ratio = 1); p < 0.05 means the
#       effect is statistically distinguishable from "no effect".
# ─────────────────────────────────────────────────────────────

# Standard library
from datetime import datetime
from pathlib import Path

# Third-party
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.linear_model import LogisticRegression

# Local
from constants import Constant as C
from splitter import TARGET, STR_REFS, get_feature_cols, make_split


# ─────────────────────────────────────────────────────────────
#  Feature classification
# ─────────────────────────────────────────────────────────────

CYCLICAL_PERIODS = {"launch_hour": 24, "launch_month": 12}


def classify_feature_cols(df: pd.DataFrame, feature_cols: list) -> dict:
    """
    Split the candidate feature columns into the four kinds that need
    different linear-model preprocessing.

    Returns dict with keys:
        continuous : numeric, scaled
        binary     : 0/1 dummies, left as-is
        cyclical   : launch_hour / launch_month, sin/cos encoded
        sentinel   : subset of `continuous` whose -1 means "missing"
                     (detected as: contains -1 AND its minimum is -1, so
                     lat/lon — which go below -1 — are never flagged)
    """
    continuous, binary, cyclical, sentinel = [], [], [], []
    for c in feature_cols:
        vals = pd.unique(df[c].dropna())
        if set(np.unique(vals)).issubset({0, 1}):
            binary.append(c)
        elif c in CYCLICAL_PERIODS:
            cyclical.append(c)
        else:
            continuous.append(c)
            col = df[c]
            if (col == -1).any() and (col >= -1).all():
                sentinel.append(c)
    return {"continuous": continuous, "binary": binary,
            "cyclical": cyclical, "sentinel": sentinel}


# ─────────────────────────────────────────────────────────────
#  Preprocessing transformers (for the sklearn Pipeline)
# ─────────────────────────────────────────────────────────────

class SentinelToNaN(BaseEstimator, TransformerMixin):
    """
    Replace the -1 missing-sentinel with NaN, but only in the given column
    positions (positions are relative to the block of columns this transformer
    receives). Other columns pass through untouched so a downstream imputer can
    fill the NaNs. Stateless (fit is a no-op) → safe in train/test pipelines.
    """

    def __init__(self, sentinel_positions=None):
        self.sentinel_positions = sentinel_positions or []

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float).copy()
        for pos in self.sentinel_positions:
            col = X[:, pos]
            col[col == -1] = np.nan
            X[:, pos] = col
        return X


def _cyclical_expand(X, periods):
    """
    Turn each cyclical column into a (sin, cos) pair.
    X has one column per cyclical feature, in the order given by `periods`.
    """
    X = np.asarray(X, dtype=float)
    out = []
    for j, period in enumerate(periods):
        ang = 2.0 * np.pi * X[:, j] / period
        out.append(np.sin(ang))
        out.append(np.cos(ang))
    return np.column_stack(out) if out else X


def build_logit_preprocessor(feature_cols: list, classes: dict) -> ColumnTransformer:
    """
    Build the column-aware preprocessor that turns the RAW numeric feature
    matrix (same columns/order the tree models receive) into a linear-model
    ready matrix:
        continuous -> sentinel(-1)->NaN -> median impute -> standardise
        binary     -> passthrough (kept 0/1)
        cyclical   -> sin/cos
    Columns not in any group are dropped.
    """
    cont = classes["continuous"]
    binr = classes["binary"]
    cyc  = classes["cyclical"]
    sent = classes["sentinel"]

    cont_idx = [feature_cols.index(c) for c in cont]
    bin_idx  = [feature_cols.index(c) for c in binr]
    cyc_idx  = [feature_cols.index(c) for c in cyc]

    # sentinel positions *within the continuous block* (that is the order the
    # continuous sub-pipeline receives its columns in)
    sent_pos = [cont.index(c) for c in sent]

    cont_pipe = Pipeline([
        ("sentinel", SentinelToNaN(sentinel_positions=sent_pos)),
        ("impute",   SimpleImputer(strategy="median")),
        ("scale",    StandardScaler()),
    ])

    cyc_periods = [CYCLICAL_PERIODS[c] for c in cyc]
    cyc_pipe = FunctionTransformer(
        _cyclical_expand, kw_args={"periods": cyc_periods}, validate=False,
    )

    transformers = [("cont", cont_pipe, cont_idx),
                    ("bin", "passthrough", bin_idx)]
    if cyc_idx:
        transformers.append(("cyc", cyc_pipe, cyc_idx))

    return ColumnTransformer(transformers, remainder="drop")


def get_transformed_feature_names(classes: dict) -> list:
    """Names of the columns produced by build_logit_preprocessor (in order)."""
    names = list(classes["continuous"]) + list(classes["binary"])
    for c in classes["cyclical"]:
        names += [f"{c}_sin", f"{c}_cos"]
    return names


# ─────────────────────────────────────────────────────────────
#  1. Integrated model (plugs into the unified pipeline / charts)
# ─────────────────────────────────────────────────────────────

def build_logit(
    feature_cols: list,
    classes: dict,
    C_reg: float = 1.0,
    penalty: str = "l2",
    class_weight=None,
    random_state: int = 42,
) -> Pipeline:
    """
    A self-contained, sklearn-compatible LOGIT estimator that consumes the SAME
    raw numeric feature matrix as the tree models (it does its own sentinel
    handling, imputation, scaling and cyclical encoding internally). It exposes
    predict / predict_proba, so it drops straight into evaluate_model(), the
    ROC/PR curves, the confusion matrix and the best-config charts.

    Args:
        feature_cols : ordered list of the raw feature column names
        classes      : output of classify_feature_cols()
        C_reg        : inverse L2 regularisation strength (smaller = stronger)
        penalty      : 'l2' (default), 'l1', or 'none'
        class_weight : None (rely on the pipeline's SMOTE) or 'balanced'
        random_state : seed

    Note: within the shared training pipeline SMOTE is applied BEFORE this
    estimator sees the data, so synthetic minority rows may carry blended
    (non -1) values that the sentinel step cannot flag. This is a minor,
    shared limitation; for statistically rigorous coefficients/odds ratios use
    run_logit_analysis(), which imputes on real data and does not resample.
    """
    pre = build_logit_preprocessor(feature_cols, classes)
    solver = "liblinear" if penalty == "l1" else "lbfgs"
    pen = None if penalty in (None, "none") else penalty
    clf = LogisticRegression(
        C=C_reg,
        penalty=pen,
        solver=solver,
        class_weight=class_weight,
        max_iter=1000,
        random_state=random_state,
    )
    return Pipeline([("prep", pre), ("clf", clf)])


# ─────────────────────────────────────────────────────────────
#  2. Inferential workflow (the statistically rigorous path)
#
#     Impute on REAL data, NO resampling (SMOTE distorts standard errors and
#     understates p-values because synthetic rows are not independent
#     observations), then VIF prune + backward p-value selection.
# ─────────────────────────────────────────────────────────────

def prepare_logit_frame(
    df: pd.DataFrame,
    classes: dict,
    artifacts: dict = None,
):
    """
    Turn a raw feature frame into a clean, named, linear-model-ready DataFrame.

    Statistics (medians for imputation, mean/std for scaling) are learned on
    the TRAIN frame and re-applied to the TEST frame to avoid leakage: pass the
    returned `artifacts` from the train call into the test call.

    Returns:
        (X_df, artifacts)
        X_df      : DataFrame with standardised continuous, 0/1 binary,
                    and sin/cos cyclical columns
        artifacts : dict of fitted medians / means / stds
    """
    cont, binr, cyc, sent = (classes["continuous"], classes["binary"],
                             classes["cyclical"], classes["sentinel"])
    f = df.copy()

    # sentinel -> NaN
    for c in sent:
        f[c] = f[c].replace(-1, np.nan)

    fit = artifacts is None
    if fit:
        artifacts = {"median": {}, "mean": {}, "std": {}}

    # median impute (train medians)
    for c in cont:
        if fit:
            artifacts["median"][c] = float(f[c].median())
        f[c] = f[c].fillna(artifacts["median"][c])

    # cyclical sin/cos
    for c in cyc:
        period = CYCLICAL_PERIODS[c]
        ang = 2.0 * np.pi * f[c] / period
        f[f"{c}_sin"] = np.sin(ang)
        f[f"{c}_cos"] = np.cos(ang)
    f = f.drop(columns=list(cyc))

    # standardise continuous (train mean/std)
    for c in cont:
        if fit:
            mu = float(f[c].mean())
            sd = float(f[c].std(ddof=0))
            artifacts["mean"][c] = mu
            artifacts["std"][c] = sd if sd != 0 else 1.0
        f[c] = (f[c] - artifacts["mean"][c]) / artifacts["std"][c]

    ordered = get_transformed_feature_names(classes)
    return f[ordered], artifacts


def drop_rare_binaries(X_df: pd.DataFrame, classes: dict, min_count: int = 10) -> list:
    """
    Drop binary columns whose smaller class has fewer than `min_count` rows.
    Such near-constant dummies invite (quasi-)complete separation, which blows
    up coefficients and standard errors. Returns the surviving column list.
    """
    binset = set(classes["binary"])
    keep = []
    for c in X_df.columns:
        if c in binset:
            col = X_df[c].values
            if min(int((col == 0).sum()), int((col == 1).sum())) < min_count:
                continue
        keep.append(c)
    return keep


def vif_prune(X_df: pd.DataFrame, columns: list, threshold: float = 10.0) -> list:
    """
    Iteratively drop the single highest-VIF feature until all VIF <= threshold.

    VIF_i measures how much feature i is explained by the others; with
    standardised inputs it equals the i-th diagonal of the inverse correlation
    matrix. High VIF (>10 is the common rule) means the coefficient and its
    p-value are unstable, so we prune before inference.
    """
    cols = list(columns)
    while len(cols) > 1:
        R = np.corrcoef(X_df[cols].values, rowvar=False)
        try:
            inv = np.linalg.inv(R)
        except np.linalg.LinAlgError:
            inv = np.linalg.pinv(R)
        vifs = np.diag(inv)
        i = int(np.argmax(vifs))
        if vifs[i] > threshold:
            dropped = cols.pop(i)
            print(f"    VIF drop: {dropped} (VIF={vifs[i]:.1f})")
        else:
            break
    return cols


def wald_table(X: np.ndarray, y: np.ndarray, names: list) -> tuple:
    """
    Fit an (effectively unpenalised) logistic regression and return a Wald-test
    table of coefficients with standard errors, z-statistics and p-values.

    The covariance of the coefficients is (X' W X)^-1 with W = diag(p(1-p)),
    the standard maximum-likelihood (Fisher information) estimator — the same
    SEs a statsmodels Logit summary reports. A large C makes the fit
    effectively unpenalised while staying valid across sklearn versions.

    Returns:
        (DataFrame[feature, coef, std_err, z, p_value], fitted LogisticRegression)
        The first row is the intercept.
    """
    clf = LogisticRegression(C=1e12, solver="lbfgs", max_iter=5000)
    clf.fit(X, y)

    Xd = np.column_stack([np.ones(len(X)), X])               # design + intercept
    beta = np.concatenate([clf.intercept_, clf.coef_.ravel()])
    p = clf.predict_proba(X)[:, 1]
    W = p * (1.0 - p)
    cov = np.linalg.pinv(Xd.T @ (Xd * W[:, None]))
    se = np.sqrt(np.clip(np.diag(cov), 1e-12, None))
    z = beta / se
    pval = 2.0 * (1.0 - stats.norm.cdf(np.abs(z)))

    table = pd.DataFrame({
        "feature": ["intercept"] + list(names),
        "coef": beta,
        "std_err": se,
        "z": z,
        "p_value": pval,
    })
    return table, clf


def select_features_by_pvalue(
    X_df: pd.DataFrame,
    y: np.ndarray,
    columns: list,
    threshold: float = 0.05,
    verbose: bool = True,
) -> tuple:
    """
    Backward elimination: repeatedly fit the model and drop the feature with
    the largest p-value while it exceeds `threshold`, until every remaining
    feature is significant.

    Returns:
        (kept_columns, final_wald_table)
    """
    cols = list(columns)
    step = 0
    while cols:
        table, _ = wald_table(X_df[cols].values, y, cols)
        body = table[table["feature"] != "intercept"]
        worst = body.loc[body["p_value"].idxmax()]
        if worst["p_value"] > threshold:
            cols.remove(worst["feature"])
            step += 1
            if verbose:
                print(f"    [{step:02d}] drop {worst['feature']:<32} "
                      f"(p={worst['p_value']:.3f})  -> {len(cols)} left")
        else:
            break
    table, _ = wald_table(X_df[cols].values, y, cols)
    return cols, table


def compute_odds_ratios(table: pd.DataFrame) -> pd.DataFrame:
    """
    Add odds ratio = exp(coef) and its 95% Wald confidence interval
    exp(coef +/- 1.96*SE) to a Wald table, sorted by effect size.
    An odds ratio > 1 raises the odds of SUCCESS, < 1 lowers them; a CI that
    crosses 1 means the feature is not significant.
    """
    df = table.copy()
    df["odds_ratio"] = np.exp(df["coef"])
    df["or_ci_lower"] = np.exp(df["coef"] - 1.96 * df["std_err"])
    df["or_ci_upper"] = np.exp(df["coef"] + 1.96 * df["std_err"])
    body = df[df["feature"] != "intercept"].copy()
    body = body.reindex(body["coef"].abs().sort_values(ascending=False).index)
    intercept = df[df["feature"] == "intercept"]
    return pd.concat([intercept, body], ignore_index=True)


def interpret_odds_ratio(feature: str, coef: float, is_binary: bool) -> str:
    """Return a plain-English one-liner for a single coefficient."""
    orr = np.exp(coef)
    direction = "raises" if coef > 0 else "lowers"
    if is_binary:
        change = abs(orr - 1.0) * 100
        unit = "its presence (1 vs 0)"
    else:
        change = abs(orr - 1.0) * 100
        unit = "a +1 SD increase"
    return (f"{feature}: OR={orr:.2f} — {unit} {direction} the odds of success "
            f"by ~{change:.0f}% (coef={coef:+.3f}), all else equal.")


# ─────────────────────────────────────────────────────────────
#  Plots
# ─────────────────────────────────────────────────────────────

def _logit_fig_dir() -> Path:
    p = C.DATA_PATH_OUTPUTS_FIG / "logit"
    p.mkdir(parents=True, exist_ok=True)
    return p


def plot_odds_ratios(or_df: pd.DataFrame, title: str = "LOGIT — odds ratios (95% CI)"):
    """
    Forest plot of odds ratios with 95% CI on a log axis. Features above the
    OR=1 line raise the odds of success; below lower them.
    """
    body = or_df[or_df["feature"] != "intercept"].copy()
    body = body.sort_values("odds_ratio")

    fig, ax = plt.subplots(figsize=(9, max(4, len(body) * 0.4)))
    y = np.arange(len(body))
    colors = ["#2ecc71" if o > 1 else "#e74c3c" for o in body["odds_ratio"]]

    ax.errorbar(
        body["odds_ratio"], y,
        xerr=[body["odds_ratio"] - body["or_ci_lower"],
              body["or_ci_upper"] - body["odds_ratio"]],
        fmt="none", ecolor="black", elinewidth=1.1, capsize=3, zorder=2,
    )
    ax.scatter(body["odds_ratio"], y, c=colors, s=55,
               edgecolor="white", zorder=3)
    ax.axvline(x=1.0, color="grey", linestyle="--", linewidth=1)
    ax.set_xscale("log")
    ax.set_yticks(y)
    ax.set_yticklabels(body["feature"], fontsize=9)
    ax.set_xlabel("Odds ratio  (log scale; >1 raises success odds, <1 lowers)")
    ax.set_title(title, fontweight="bold")
    plt.tight_layout()
    path = _logit_fig_dir() / "logit_odds_ratios.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    print(f"  Saved => {path}")
    return path


# ─────────────────────────────────────────────────────────────
#  Plain-English interpretation report (saved as .txt)
# ─────────────────────────────────────────────────────────────

# Real standard-deviation values for every continuous feature that survived
# backward elimination (computed on the pre-2010 training set, sentinels
# replaced). These are used to translate "+1 SD" into a concrete real-world
# quantity in the text report.
_FEATURE_SD = {
    "Launch Year"              :  13.3,
    "Price_cfg"                : 203.5,
    "Liftoff Thrust (kN)_cfg"  : 5667.3,
    "Payload to LEO (kg)_cfg"  : 10376.7,
    "Payload to GTO (kg)_cfg"  : 2308.7,
    "Stages_cfg"               :   0.7,
    "Rocket Height (m)_cfg"    :  11.2,
    "Fairing Diameter (m)_cfg" :   0.8,
    "total_payloads"           :   1.3,
    "total_mass_kg"            : 4232.4,
    "mission_count"            :   0.6,
    "Launch Site Lat_loc"      :  16.2,
    "Launch Site Lon_loc"      :  74.6,
    "rocket_prior_launches"    : 162.1,
    "rocket_prior_success_rate":   0.2,
    "org_prior_launches"       : 747.6,
    "org_prior_success_rate"   :   0.1,
    "payload_per_thrust"       :   0.6,
    "fairing_slenderness"      :   1.2,
    "rocket_slenderness"       :   2.5,
    "site_prior_successes"     :  80.7,
    "site_prior_success_rate"  :   0.2,
    "payload_utilisation"      :   0.4,
}

# Units for the continuous features (for the "1 SD = X units" annotation).
_FEATURE_UNITS = {
    "Launch Year"              : "years",
    "Price_cfg"                : "M$",
    "Liftoff Thrust (kN)_cfg"  : "kN",
    "Payload to LEO (kg)_cfg"  : "kg",
    "Payload to GTO (kg)_cfg"  : "kg",
    "Stages_cfg"               : "stages",
    "Rocket Height (m)_cfg"    : "m",
    "Fairing Diameter (m)_cfg" : "m",
    "total_payloads"           : "payloads",
    "total_mass_kg"            : "kg",
    "mission_count"            : "missions",
    "Launch Site Lat_loc"      : "°lat",
    "Launch Site Lon_loc"      : "°lon",
    "rocket_prior_launches"    : "launches",
    "rocket_prior_success_rate": "(rate 0–1)",
    "org_prior_launches"       : "launches",
    "org_prior_success_rate"   : "(rate 0–1)",
    "payload_per_thrust"       : "kg/kN",
    "fairing_slenderness"      : "(ratio)",
    "rocket_slenderness"       : "(ratio)",
    "site_prior_successes"     : "launches",
    "site_prior_success_rate"  : "(rate 0–1)",
    "payload_utilisation"      : "(ratio)",
}


def _feature_sentence(feature: str, coef: float, or_val: float,
                       or_lo: float, or_hi: float, pval: float,
                       is_binary: bool) -> str:
    """
    Build a plain-English sentence interpreting one coefficient.

    For continuous features (standardised):
        "+1 SD in <feature> (≈ X units) => the odds of a successful
         launch are multiplied by OR (±95% CI), i.e. a Y% change."
    For binary / dummy features (0/1):
        "When <feature> = 1 (vs. 0), the odds of a successful launch
         are multiplied by OR (±95% CI), i.e. a Y% change."
    """
    direction = "increase" if coef > 0 else "decrease"
    pct_change = abs(or_val - 1.0) * 100.0
    outcome = "raises" if coef > 0 else "lowers"
    ci_str = f"95% CI [{or_lo:.2f}, {or_hi:.2f}]"
    sig_str = f"p={'<0.001' if pval < 0.001 else f'{pval:.3f}'}"

    if is_binary:
        # Readable label: strip suffix codes and make it human-friendly
        label = (feature
                 .replace("_co", " (country dummy)")
                 .replace("_cfg", " (config flag)")
                 .replace("_missing", " (spec absent)")
                 .replace("_", " "))
        return (
            f"  • [{label}] — Binary flag (0/1).\n"
            f"    When this flag = 1 (vs. 0), the odds of a successful launch are "
            f"multiplied by {or_val:.2f} ({ci_str}), i.e. its presence {outcome} the odds "
            f"of success by ~{pct_change:.0f}%.\n"
            f"    Direction: {direction} in odds of SUCCESS. {sig_str}.\n"
        )
    else:
        sd = _FEATURE_SD.get(feature, None)
        unit = _FEATURE_UNITS.get(feature, "units")
        sd_str = f"≈ {sd:,.1f} {unit}" if sd is not None else f"1 SD {unit}"
        return (
            f"  • [{feature}] — Continuous feature (standardised: 1 SD {sd_str}).\n"
            f"    A +1 SD increase in this feature multiplies the odds of a "
            f"successful launch by {or_val:.2f} ({ci_str}), i.e. {outcome} the odds of "
            f"success by ~{pct_change:.0f}%.\n"
            f"    Direction: {direction} in odds of SUCCESS. {sig_str}.\n"
        )


def generate_logit_interpretation_report(
    or_df: pd.DataFrame,
    classes: dict,
    config_label: str,
    out_dir: Path,
    extra_notes: str = "",
) -> Path:
    """
    Write a plain-English interpretation report as a .txt file.

    For each significant feature the report explains:
      • What the feature is (continuous vs. binary)
      • The odds ratio and its 95% CI
      • The direction and magnitude of the effect on success probability
      • A concrete "+1 SD ≈ X units" anchor for continuous features

    Args:
        or_df        : output of compute_odds_ratios() — one row per feature
        classes      : output of classify_feature_cols()
        config_label : e.g. "LOGIT 1 - logit_C=0.01, logit_penalty=l1"
        out_dir      : directory where the .txt file is saved
        extra_notes  : optional additional paragraph appended at the end

    Returns:
        Path to the saved .txt file.
    """
    binset = set(classes["binary"])
    body   = or_df[or_df["feature"] != "intercept"].copy()
    # Sort: strongest effects first (by absolute coefficient)
    body   = body.iloc[body["coef"].abs().argsort()[::-1]]

    lines = []
    lines.append("=" * 70)
    lines.append(config_label)
    lines.append("=" * 70)
    lines.append(
        "\nThis report interprets the odds ratios of the LOGIT (Logistic Regression)\n"
        "model. The target variable is launch_success_binary (1 = Success, 0 = Failure).\n"
        "\nThe model was fitted on the PRE-2010 training set WITHOUT resampling\n"
        "(SMOTE is not applied here to preserve valid standard errors and p-values).\n"
        "All continuous features are standardised (z-score). Each coefficient\n"
        "represents the change in log-odds of SUCCESS per unit move in the feature,\n"
        "all other features held constant.\n"
        "\nOdds Ratio (OR) interpretation:\n"
        "  OR > 1  => the feature RAISES the odds of a successful launch.\n"
        "  OR < 1  => the feature LOWERS the odds of a successful launch.\n"
        "  OR = 1  => no effect (coefficient = 0).\n"
        "The percentage shown is |OR - 1| × 100, i.e. how much the odds change.\n"
        "Note: 'odds' and 'probability' are different; an OR of 2 does not mean\n"
        "the probability doubles.\n"
    )
    lines.append(f"Number of significant features (p < 0.05): {len(body)}\n")
    lines.append("-" * 70)
    lines.append("FEATURE INTERPRETATIONS (sorted by effect size, strongest first)\n")
    lines.append("-" * 70)

    for _, row in body.iterrows():
        is_bin = row["feature"] in binset
        lines.append(_feature_sentence(
            row["feature"], row["coef"], row["odds_ratio"],
            row["or_ci_lower"], row["or_ci_upper"], row["p_value"], is_bin,
        ))

    # Intercept
    icpt = or_df[or_df["feature"] == "intercept"].iloc[0]
    lines.append("-" * 70)
    lines.append(
        f"Intercept: coef = {icpt['coef']:.3f}  =>  baseline log-odds = {icpt['coef']:.3f}.\n"
        f"  This is the predicted log-odds when every feature equals its mean (all\n"
        f"  continuous features = 0, all binary flags = 0).\n"
    )

    if extra_notes:
        lines.append("-" * 70)
        lines.append("NOTES\n")
        lines.append(extra_notes)

    lines.append("=" * 70)
    lines.append(f"Report generated: {datetime.today().strftime('%Y-%m-%d %H:%M')}")
    lines.append("=" * 70)

    out_dir.mkdir(parents=True, exist_ok=True)
    safe_label = config_label.split(" - ")[0].strip().lower().replace(" ", "_")
    path = out_dir / f"interpretation_{safe_label}.txt"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Interpretation report => {path}")
    return path

def run_logit_analysis(
    df: pd.DataFrame = None,
    cut_year: int = 2010,
    pvalue_threshold: float = 0.05,
    vif_threshold: float = 10.0,
    min_binary_count: int = 10,
    save: bool = True,
    verbose: bool = True,
    config_label: str = None,
    out_dir: Path = None,
) -> dict:
    """
    Full inferential LOGIT analysis answering the research question
    "which features drive launch success, and by how much?".

    Pipeline:
        temporal split (no SMOTE) -> classify features -> impute(-1)->median
        -> standardise continuous -> sin/cos cyclical -> drop rare dummies
        -> VIF prune -> backward p-value selection -> Wald table + odds ratios.

    Args:
        df               : model_data DataFrame. If None, loads from disk.
        cut_year         : temporal split year (train < cut_year).
        pvalue_threshold : backward elimination stops when all p < threshold.
        vif_threshold    : VIF ceiling (10 = standard rule of thumb).
        min_binary_count : minimum minority-class count before a dummy is dropped.
        save             : write CSV report, odds-ratio forest plot, and .txt
                           interpretation file.
        verbose          : print progress.
        config_label     : one-line label for the report title, e.g.
                           "LOGIT 1 - logit_C=0.01, logit_penalty=l1".
                           Auto-generated when None.
        out_dir          : directory for the .txt report. Defaults to
                           figures/sensitivity/best_configs/logit/.

    Returns a dict with: classes, kept_features, wald_table, odds_ratios,
    artifacts, X_train, X_test, y_train, y_test (the prepared frames).
    """
    if df is None:
        df = pd.read_csv(C.DATA_PATH_INPUTS_CLEAN / C.MODEL_DATA_FILENAME)

    feature_cols = get_feature_cols(df)
    classes = classify_feature_cols(df, feature_cols)

    if verbose:
        print("=" * 60)
        print("LOGIT — inferential analysis (odds ratios)")
        print("=" * 60)
        print(f"  continuous : {len(classes['continuous'])}  "
              f"(of which sentinel -1 : {len(classes['sentinel'])})")
        print(f"  binary     : {len(classes['binary'])}")
        print(f"  cyclical   : {len(classes['cyclical'])} -> sin/cos pairs")

    # temporal split — keep raw rows, no resampling
    df_train = df[df["Launch Year"] < cut_year]
    df_test  = df[df["Launch Year"] >= cut_year]
    y_train = df_train[TARGET].values
    y_test  = df_test[TARGET].values

    X_train, artifacts = prepare_logit_frame(df_train[feature_cols], classes)
    X_test, _ = prepare_logit_frame(df_test[feature_cols], classes, artifacts)

    # candidate columns -> rare-binary guard -> VIF prune
    candidates = drop_rare_binaries(X_train, classes, min_count=min_binary_count)
    if verbose:
        print(f"\n  Candidate features        : {X_train.shape[1]}")
        print(f"  After rare-binary guard   : {len(candidates)}")
        print(f"\n  VIF pruning (threshold {vif_threshold:.0f}):")
    candidates = vif_prune(X_train, candidates, threshold=vif_threshold)
    if verbose:
        print(f"  After VIF pruning         : {len(candidates)}")
        print(f"\n  Backward p-value selection (alpha={pvalue_threshold}):")

    kept, table = select_features_by_pvalue(
        X_train, y_train, candidates, threshold=pvalue_threshold, verbose=verbose,
    )

    or_df = compute_odds_ratios(table)
    binset = set(classes["binary"])

    if verbose:
        print(f"\n  Significant features kept : {len(kept)}")
        print("\n  Odds-ratio interpretation (sorted by effect size):")
        for _, r in or_df[or_df["feature"] != "intercept"].iterrows():
            print("    " + interpret_odds_ratio(
                r["feature"], r["coef"], r["feature"] in binset))

    if save:
        ts = datetime.today().strftime("%Y%m%d_%Hh%M")
        rep_dir = C.DATA_PATH_OUTPUTS_EVL_REP
        rep_dir.mkdir(parents=True, exist_ok=True)
        rep_path = rep_dir / f"logit_odds_ratios_{ts}.csv"
        or_df.round(5).to_csv(rep_path, index=False)
        print(f"\n  Odds-ratio report saved => {rep_path}")
        plot_odds_ratios(or_df)

        # Plain-English interpretation text file
        if config_label is None:
            config_label = "LOGIT — inferential analysis"
        txt_dir = (out_dir if out_dir is not None
                   else C.DATA_PATH_OUTPUTS_FIG / "sensitivity" / "best_configs" / "logit")
        generate_logit_interpretation_report(
            or_df, classes, config_label, txt_dir,
        )

    return {
        "classes": classes,
        "kept_features": kept,
        "wald_table": table,
        "odds_ratios": or_df,
        "artifacts": artifacts,
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
    }


if __name__ == "__main__":
    run_logit_analysis()