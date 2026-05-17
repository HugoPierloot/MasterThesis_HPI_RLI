# third party imports
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# local imports
from constants import Constant as C

# ─────────────────────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────────────────────

TARGET     = 'launch_success_binary'
CUT_YEAR   = 2010        # temporal split boundary (see analysis below)
RANDOM_STATE = 42

# String reference columns — informative for analysis but not fed to models
STR_REFS = ['Rocket Name', 'Rocket Organisation', 'Location', 'Country_loc']

# ─────────────────────────────────────────────────────────────
#  This dataset is a time series. Using random train/test split
#  would allow the model to train on 2015 data and test on 1980
#  data — impossible in production where we predict future launches.
#
#  Cut year 2010 was selected because:
#  - Train (1957–2009): 5,020 rows, 472 failures, 42 organisations
#  - Test  (2010–2021): 1,148 rows,  70 failures, 35 organisations
#  - Test set is the modern era (commercial launch boom).
#
#  SMOTE only on training data:
#  The test set must reflect the real-world class distribution
#  (91% success / 9% failure). Applying SMOTE to the test set
#  would measure performance on an artificial distribution.
#  SMOTE is applied ONLY to X_train / y_train after the split.
#
#  SMOTE requires only numeric features — string refs are excluded.
# ─────────────────────────────────────────────────────────────

def get_feature_cols(df: pd.DataFrame) -> list:
    """Return the numeric feature columns (excludes target + string refs)."""
    return [
        c for c in df.columns
        if c not in STR_REFS and c != TARGET
        and pd.api.types.is_numeric_dtype(df[c])
    ]


def make_split(
    df: pd.DataFrame = None,
    cut_year: int = CUT_YEAR,
    apply_smote: bool = True,
    smote_strategy: float = 0.3,
    random_state: int = RANDOM_STATE,
    verbose: bool = True,
) -> dict:
    """
    Perform a temporal train/test split and optionally apply SMOTE
    to the training set only.

    Args:
        df            : model_data DataFrame. If None, loads from disk.
        cut_year      : first year included in the test set (default 2010).
        apply_smote   : whether to apply SMOTE to X_train (default True).
        smote_strategy: target ratio minority/majority after SMOTE (default 0.3
                        = 30 failures per 100 successes, i.e. ~23% failure rate).
                        Set to 1.0 for fully balanced.
        random_state  : reproducibility seed.
        verbose       : print split summary.

    Returns:
        dict with keys:
            X_train, X_test, y_train, y_test  — numpy arrays
            X_train_raw, X_test_raw           — DataFrames (with string refs)
            feature_cols                       — list of feature column names
            df_train, df_test                 — full DataFrames before SMOTE
    """
    if df is None:
        path = C.DATA_PATH_INPUTS_CLEAN / "model_data.csv"
        df = pd.read_csv(path)

    feature_cols = get_feature_cols(df)

    # Temporal split
    df_train = df[df['Launch Year'] <  cut_year].copy()
    df_test  = df[df['Launch Year'] >= cut_year].copy()

    X_train_raw = df_train[feature_cols].values
    X_test_raw  = df_test[feature_cols].values
    y_train     = df_train[TARGET].values
    y_test      = df_test[TARGET].values

    if verbose:
        print("=" * 55)
        print("Train/Test Split Summary")
        print("=" * 55)
        print(f"  Cut year     : {cut_year} (train < {cut_year}, test >= {cut_year})")
        print(f"  Train size   : {len(df_train):,}  "
              f"({len(df_train)/len(df):.0%} of total)")
        print(f"  Test size    : {len(df_test):,}  "
              f"({len(df_test)/len(df):.0%} of total)")
        print()
        print(f"  Train failures : {(y_train==0).sum():,}  "
              f"({(y_train==0).mean():.1%})")
        print(f"  Train successes: {(y_train==1).sum():,}  "
              f"({(y_train==1).mean():.1%})")
        print(f"  Test  failures : {(y_test==0).sum():,}  "
              f"({(y_test==0).mean():.1%})")
        print(f"  Test  successes: {(y_test==1).sum():,}  "
              f"({(y_test==1).mean():.1%})")
        print()
        print(f"  Train orgs     : {df_train['Rocket Organisation'].nunique()}")
        print(f"  Test  orgs     : {df_test['Rocket Organisation'].nunique()}")
        print(f"  Features       : {len(feature_cols)}")

    # SMOTE on training set only
    X_train_final = X_train_raw
    if apply_smote:
        smote = SMOTE(
            sampling_strategy=smote_strategy,
            random_state=random_state,
            k_neighbors=5,
        )
        X_train_final, y_train = smote.fit_resample(X_train_raw, y_train)
        if verbose:
            print()
            print(f"  After SMOTE (strategy={smote_strategy}):")
            print(f"    Train failures : {(y_train==0).sum():,}  "
                  f"({(y_train==0).mean():.1%})")
            print(f"    Train successes: {(y_train==1).sum():,}  "
                  f"({(y_train==1).mean():.1%})")

    if verbose:
        print("=" * 55)

    return {
        "X_train"      : X_train_final,
        "X_test"       : X_test_raw,
        "y_train"      : y_train,
        "y_test"       : y_test,
        "X_train_raw"  : df_train[feature_cols],   # DataFrame, pre-SMOTE
        "X_test_raw"   : df_test[feature_cols],    # DataFrame
        "feature_cols" : feature_cols,
        "df_train"     : df_train,
        "df_test"      : df_test,
    }


if __name__ == "__main__":
    split = make_split()