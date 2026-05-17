# third party imports
import pandas as pd
import numpy as np

# local imports
from constants import Constant as C


# ─────────────────────────────────────────────────────────────
#  Feature Engineering
#  Goal: predict launch_success_binary (1=Success, 0=Failure) and interpret which features drive the outcome.
#
#  Features are grouped by theme:
#    A. Temporal
#    B. Rocket experience (point-in-time cumulative)
#    C. Organisation experience (point-in-time cumulative)
#    D. Physical specs ratios (rocket design complexity)
#    E. Launch site
#    F. Mission payload
#    G. Era flags
# ─────────────────────────────────────────────────────────────


def _require_sorted(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure rows are sorted by launch time before any cumcount/cumsum."""
    return df.sort_values('Launch Time').reset_index(drop=True)


# ─────────────────────────────────────────────────────────────
#  A. Temporal features
#  Rationale: early space age had far more failures; launch hour and month proxy for weather windows and political scheduling.
# ─────────────────────────────────────────────────────────────

def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    launch_dt = pd.to_datetime(df['Launch Time'], utc=True)

    df['launch_hour']         = launch_dt.dt.hour          # 0–23
    df['launch_month']        = launch_dt.dt.month         # 1–12
    df['launch_decade']       = (df['Launch Year'] // 10) * 10  # 1950, 1960, ...

    # Cold War era flag: pre-1990 launches had systematically higher failure rates
    df['era_cold_war']        = (df['Launch Year'] < 1990).astype(int)
    # Commercial era: post-Falcon 1 (2006)
    df['era_commercial']      = (df['Launch Year'] >= 2006).astype(int)

    print("  [A] Temporal features added: launch_hour, launch_month, launch_decade, era_*")
    return df


# ─────────────────────────────────────────────────────────────
#  B. Rocket experience — point-in-time cumulative
#
#  Count only launches of the SAME rocket that happened STRICTLY BEFORE it.
#  A rocket on its first flight has no track record;
#  and the 50th flight of a proven rocket is a very different risk profile.
# ─────────────────────────────────────────────────────────────

def add_rocket_experience_features(df: pd.DataFrame) -> pd.DataFrame:
    df = _require_sorted(df)

    # Per exact rocket name cumcount gives 0-indexed count of prior launches of same rocket
    df['rocket_prior_launches'] = (
        df.groupby('Rocket Name').cumcount()  # 0 = first flight ever
    )
    df['rocket_prior_successes'] = (
        df.groupby('Rocket Name')['launch_success_binary']
        .apply(lambda s: s.shift(1).fillna(0).cumsum())
        .reset_index(level=0, drop=True)
    )

    # Success rate of this rocket up to (not including) this launch
    # NaN on first flight (no prior data) => fill with global mean
    global_mean = df['launch_success_binary'].mean()
    df['rocket_prior_success_rate'] = np.where(
        df['rocket_prior_launches'] == 0,
        global_mean,  # first flight: use global base rate
        df['rocket_prior_successes'] / df['rocket_prior_launches']
    )

    df['rocket_is_maiden_flight'] = (df['rocket_prior_launches'] == 0).astype(int)

    print("  [B] Rocket experience features added: rocket_prior_launches, "
          "rocket_prior_successes, rocket_prior_success_rate, rocket_is_maiden_flight")
    return df


# ─────────────────────────────────────────────────────────────
#  C. Organisation experience — point-in-time cumulative
#
#  Same logic as B but at the organisation level.
#  A new company attempting its first launch (e.g. Iran,
#  North Korea) is very different from SpaceX on launch #100.
# ─────────────────────────────────────────────────────────────

def add_organisation_experience_features(df: pd.DataFrame) -> pd.DataFrame:
    df = _require_sorted(df)

    df['org_prior_launches'] = (
        df.groupby('Rocket Organisation').cumcount()
    )
    df['org_prior_successes'] = (
        df.groupby('Rocket Organisation')['launch_success_binary']
        .apply(lambda s: s.shift(1).fillna(0).cumsum())
        .reset_index(level=0, drop=True)
    )

    global_mean = df['launch_success_binary'].mean()
    df['org_prior_success_rate'] = np.where(
        df['org_prior_launches'] == 0,
        global_mean,
        df['org_prior_successes'] / df['org_prior_launches']
    )

    df['org_is_first_launch'] = (df['org_prior_launches'] == 0).astype(int)

    print("  [C] Organisation experience features added: org_prior_launches, "
          "org_prior_successes, org_prior_success_rate, org_is_first_launch")
    return df


# ─────────────────────────────────────────────────────────────
#  D. Physical design features (ratios & flags)
#
#  Raw measurements (height, thrust) are hard to interpret alone,
#  their ratios reveal design complexity and engineering margins.
#  All computed on _cfg columns which use -1 sentinel for missing;
#  ratio is set to -1 when either input is missing (-1).
# ─────────────────────────────────────────────────────────────

def add_physical_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    def _safe_ratio(num_col, den_col, result_col):
        """Compute ratio only when both columns have real (non-sentinel) values."""
        valid = (df[num_col] >= 0) & (df[den_col] > 0)
        df[result_col] = np.where(
            valid,
            df[num_col] / df[den_col],
            -1  # preserve sentinel for missing
        )

    # Payload efficiency: how much payload per unit of thrust
    # High ratio = efficient design; very low = high thrust needed (harder mission)
    _safe_ratio('Payload to LEO (kg)_cfg', 'Liftoff Thrust (kN)_cfg', 'payload_per_thrust')

    # Fairing slenderness: taller fairing relative to diameter = more drag risk
    _safe_ratio('Fairing Height (m)_cfg', 'Fairing Diameter (m)_cfg', 'fairing_slenderness')

    # Rocket slenderness: taller/thinner rockets are structurally more challenging
    _safe_ratio('Rocket Height (m)_cfg', 'Fairing Diameter (m)_cfg', 'rocket_slenderness')

    # Complexity proxy: more stages = more separation events = more failure points
    # Already numeric in Stages_cfg; flag multi-stage rockets explicitly
    df['is_multistage'] = (df['Stages_cfg'] > 1).astype(int)
    df['has_strap_ons'] = (df['Strap-ons_cfg'] > 0).astype(int)

    print("  [D] Physical features added: payload_per_thrust, fairing_slenderness, "
          "rocket_slenderness, is_multistage, has_strap_ons")
    return df


# ─────────────────────────────────────────────────────────────
#  E. Launch site experience — point-in-time cumulative
#
#  Some launch sites (Baikonur, Cape Canaveral) have decades of
#  experience; others are brand new.
# ─────────────────────────────────────────────────────────────

def add_launch_site_features(df: pd.DataFrame) -> pd.DataFrame:
    df = _require_sorted(df)

    df['site_prior_launches'] = (
        df.groupby('Location').cumcount()
    )
    df['site_prior_successes'] = (
        df.groupby('Location')['launch_success_binary']
        .apply(lambda s: s.shift(1).fillna(0).cumsum())
        .reset_index(level=0, drop=True)
    )

    global_mean = df['launch_success_binary'].mean()
    df['site_prior_success_rate'] = np.where(
        df['site_prior_launches'] == 0,
        global_mean,
        df['site_prior_successes'] / df['site_prior_launches']
    )

    print("  [E] Launch site features added: site_prior_launches, "
          "site_prior_successes, site_prior_success_rate")
    return df


# ─────────────────────────────────────────────────────────────
#  F. Mission payload features
#
#  Heavier/more complex payloads push rockets closer to their
#  performance envelope. Payload fraction (mass / capacity) is
#  a proxy for mission difficulty.
# ─────────────────────────────────────────────────────────────

def add_payload_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Payload utilisation: how much of LEO capacity is being used
    # Only meaningful when both total_mass_kg and Payload to LEO are real values
    valid = (df['total_mass_kg'] > 0) & (df['Payload to LEO (kg)_cfg'] > 0)
    df['payload_utilisation'] = np.where(
        valid,
        df['total_mass_kg'] / df['Payload to LEO (kg)_cfg'],
        -1
    )

    # Flag: mission carries multiple payloads (higher coordination complexity)
    df['is_multi_payload'] = (df['mission_count'] > 1).astype(int)

    print("  [F] Payload features added: payload_utilisation, is_multi_payload")
    return df


# ─────────────────────────────────────────────────────────────
#  G. Drop columns no longer needed after feature engineering
#
#  Launch Time (raw string) and Launch Year Mon are not directly
#  usable by sklearn — their information is captured in A–E.
#  Rocket Name / Rocket Organisation / Location are kept as
#  string labels for explainability groupby analysis, but must
#  be excluded from the feature matrix X when training.
# ─────────────────────────────────────────────────────────────

def drop_post_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    drop = ['Launch Time']
    dropped = [c for c in drop if c in df.columns]
    df = df.drop(columns=dropped)
    print(f"  [G] Dropped post-engineering columns: {dropped}")
    return df

def drop_high_correlation(
    df: pd.DataFrame,
    threshold: float = C.CORR_THRESHOLD,
    return_report: bool = True,
) -> pd.DataFrame:
    """
    Drop numeric features with absolute pairwise Pearson correlation > threshold.
 
    Args:
        df            : input DataFrame (post feature engineering)
        threshold     : correlation cutoff (default 0.85)
        return_report : if True, print a full report of dropped columns
 
    Returns:
        DataFrame with high-correlation columns removed.
    """
    df = df.copy()
 
    # 1. Apply forced drops first
    forced_present = [c for c in C._FORCED_DROPS if c in df.columns]
    df = df.drop(columns=forced_present)
    print(f"  [H] Forced drops ({len(forced_present)}): {forced_present}")
 
    # 2. Identify numeric columns to check
    #    Exclude: target, string reference cols, missingness flags
    #    (flags are structural correlates of their parent column — tree models
    #     use them independently; dropping them based on correlation loses signal)
    exclude_from_check = {'launch_success_binary'}
    flag_cols = {c for c in df.columns if c.endswith('_missing')}
    str_cols  = set(df.select_dtypes(include=['object', 'str']).columns)
    numeric_check_cols = [
        c for c in df.select_dtypes(include='number').columns
        if c not in exclude_from_check and c not in flag_cols
    ]
 
    # 3. Compute correlation matrix
    corr_matrix = df[numeric_check_cols].corr().abs()
 
    # 4. Greedy drop: iterate upper triangle, drop the less "central" column
    #    "centrality" = mean absolute correlation with all other features
    mean_corr = corr_matrix.mean(axis=1)
    dropped_auto = []
    cols_to_check = list(numeric_check_cols)
 
    for i in range(len(cols_to_check)):
        col_a = cols_to_check[i]
        if col_a not in df.columns:
            continue
        for j in range(i + 1, len(cols_to_check)):
            col_b = cols_to_check[j]
            if col_b not in df.columns:
                continue
            if col_a not in corr_matrix.index or col_b not in corr_matrix.index:
                continue
            r = corr_matrix.loc[col_a, col_b]
            if r > threshold:
                # Drop the one with lower mean correlation to others (less central)
                drop_col = col_b if mean_corr[col_a] >= mean_corr[col_b] else col_a
                if drop_col in df.columns:
                    if return_report:
                        keep_col = col_a if drop_col == col_b else col_b
                        print(f"    r={r:.3f}  DROP '{drop_col}'  (kept '{keep_col}')")
                    df = df.drop(columns=[drop_col])
                    dropped_auto.append(drop_col)
 
    print(f"  [H] Auto-dropped {len(dropped_auto)} high-correlation columns "
          f"(threshold={threshold})")
    print(f"  [H] Total dropped this step: {len(forced_present) + len(dropped_auto)}")
    print(f"  [H] Remaining columns: {df.shape[1]}")
    return df


# ─────────────────────────────────────────────────────────────
#  Master runner
# ─────────────────────────────────────────────────────────────

def run_feature_engineering(df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Load final_data.csv (or accept a DataFrame), apply all feature
    engineering steps, and save the result as model_data.csv.

    Args:
        df: optional — pass a DataFrame directly (e.g. from run_cleaning()).
            If None, loads from DATA_PATH_INPUTS_CLEAN / final_data.csv.

    Returns:
        pd.DataFrame: feature-engineered dataset ready for ML.
    """
    if df is None:
        path = C.DATA_PATH_INPUTS_CLEAN / "final_data.csv"
        print(f"Loading {path}")
        df = pd.read_csv(path)

    print("=" * 60)
    print("Starting feature engineering pipeline")
    print(f"Input shape: {df.shape}")
    print("=" * 60)

    df = add_temporal_features(df)
    df = add_rocket_experience_features(df)
    df = add_organisation_experience_features(df)
    df = add_physical_features(df)
    df = add_launch_site_features(df)
    df = add_payload_features(df)
    df = drop_post_engineering(df)
    df = drop_high_correlation(df)

    # Save
    C.DATA_PATH_INPUTS_CLEAN.mkdir(parents=True, exist_ok=True)
    out_path = C.DATA_PATH_INPUTS_CLEAN / "model_data.csv"
    df.to_csv(out_path, index=False)

    print()
    print("=" * 60)
    print(f"Feature engineering complete.")
    print(f"  Output shape : {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"  Saved        => {out_path}")
    print()

    # Summary of string columns still present (must be excluded from X)
    str_cols = df.select_dtypes(include=['object', 'str']).columns.tolist()
    if str_cols:
        print(f"  [INFO] {len(str_cols)} string columns kept for reference "
              f"(exclude from model feature matrix X):")
        for c in str_cols:
            print(f"    - {c}  ({df[c].nunique()} unique)")
    print("=" * 60)

    return df


if __name__ == "__main__":
    run_feature_engineering()