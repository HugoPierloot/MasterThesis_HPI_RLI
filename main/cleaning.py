# third party imports
import pandas as pd
import numpy as np
from pathlib import Path

# local imports
from constants import Constant as C
from loaders import (
    load_raw_companies, load_raw_configs, load_raw_families,
    load_raw_launches, load_raw_locations, load_raw_missions
)


# =============================================
#  Defining base functions
# =============================================

def _strip_currency(series: pd.Series) -> pd.Series:
    """'$59.5 million' => 59.5  (unit: millions USD)"""
    return (
        series.astype(str)
        .str.replace(r"[$,\s]", "", regex=True)
        .str.replace("million", "", regex=False)
        .str.strip()
        .replace("nan", np.nan)
        .astype(float)
    )

def _strip_unit(series: pd.Series, unit: str) -> pd.Series:
    """'54.9 m' => 54.9  /  '4,940 kN' => 4940.0"""
    return (
        series.astype(str)
        .str.replace(unit, "", regex=False)
        .str.replace(",", "", regex=False)
        .str.strip()
        .replace("nan", np.nan)
        .astype(float)
    )

def _strip_percent(series: pd.Series) -> pd.Series:
    """'100%' => 1.0  /  '85.7%' => 0.857"""
    return (
        series.astype(str)
        .str.replace("%", "", regex=False)
        .str.strip()
        .replace("nan", np.nan)
        .astype(float)
        .div(100)
    )

def _one_hot(df: pd.DataFrame, col: str, prefix: str = None) -> pd.DataFrame:
    """Add OHE columns for a categorical column, drop original."""
    dummies = pd.get_dummies(df[col], prefix=prefix or col, dtype=int)
    return pd.concat([df, dummies], axis=1)


# =============================================
#  Per-table cleaners
# =============================================

#  Companies
#  Raw: Company Name | Company Country | Ownership
#  Steps: OHE Company Country + OHE Ownership
def clean_companies(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
 
    # OHE: Company Country (15 countries)
    df = _one_hot(df, C.COMPANY_COUNTRY_COL)
    # OHE: Ownership (State / Private)
    df = _one_hot(df, C.OWNERSHIP_COL)
    # Keep Company Name as plain string identifier
    return df

#  Configs
#  Raw:    Family Id | No | Config | Status | Price | Liftoff Thrust |
#          Payload to LEO | Payload to GTO | Stages | Strap-ons |
#          Rocket Height | Fairing Diameter | Fairing Height
#  Steps:
#   - Create Config Id = FamilyId_No
#   - Rename columns to include units in name
#   - Strip units from string columns => float
#   - OHE Status  (Active / Retired / Planned)
#   - Fill Strap-ons nulls with 0 (structural zero, not missing)
def clean_configs(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
 
    # Composite key
    df["Config Id"] = (
        df[C.FAMILY_ID_COL].astype(int).astype(str)
        + "_"
        + df[C.NO_COL].astype(int).astype(str)
    )
 
    # Rename columns to carry units
    rename_map = {
        C.LIFTOFF_THRUST_COL:   "Liftoff Thrust (kN)",
        C.CONF_PAYLOAD_LEO_COL: "Payload to LEO (kg)",
        C.CONF_PAYLOAD_GTO_COL: "Payload to GTO (kg)",
        C.ROCKET_HEIGHT_COL:    "Rocket Height (m)",
        C.FAIRING_DIAMETER_COL: "Fairing Diameter (m)",
        C.FAIRING_HEIGHT_COL:   "Fairing Height (m)",
    }

    df = df.rename(columns=rename_map)

    # Strip units => numeric
    df[C.PRICE_COL]              = _strip_currency(df[C.PRICE_COL])
    df["Liftoff Thrust (kN)"]    = _strip_unit(df["Liftoff Thrust (kN)"],   " kN")
    df["Payload to LEO (kg)"]    = _strip_unit(df["Payload to LEO (kg)"],   " kg")
    df["Payload to GTO (kg)"]    = _strip_unit(df["Payload to GTO (kg)"],   " kg")
    df["Rocket Height (m)"]      = _strip_unit(df["Rocket Height (m)"],     " m")
    df["Fairing Diameter (m)"]   = _strip_unit(df["Fairing Diameter (m)"],  " m")
    df["Fairing Height (m)"]     = _strip_unit(df["Fairing Height (m)"],    " m")
 
    # Strap-ons: structural zero (no strap-ons) not missing data
    df[C.STRAPS_ONS_COL] = df[C.STRAPS_ONS_COL].fillna(0).astype(int)
 
    # OHE Status
    df = _one_hot(df, C.STATUS_COL, prefix="status")
 
    # Reorder: Config Id first
    cols = ["Config Id"] + [c for c in df.columns if c != "Config Id"]
    df = df[cols]
 
    return df

#  Families
#  Raw:    Family Id | Family | Missions | Successes |
#          Partial Failures | Failures | Success Streak | Success Rate
#  Steps:
#   - Fill null counts with 0
#   - Parse Success Rate % => float [0, 1]
#   - Cast count columns to int
def clean_families(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
 
    count_cols = [
        C.MISSIONS_COL, C.SUCCESSES_COL,
        C.PARTIAL_FAILURES_COL, C.FAILURES_COL, C.SUCCESS_STREAK_COL,
    ]
    # Nulls = 0 launches recorded, not missing data
    for col in count_cols:
        df[col] = df[col].fillna(0).astype(int)
 
    df[C.SUCCESS_RATE_COL] = _strip_percent(df[C.SUCCESS_RATE_COL])
 
    return df

#  Launches
#  Raw:    Launch Id | Launch Time | Launch Status | Launch Suborbital |
#          Rocket Name | Rocket Organisation | Rocket Price |
#          Rocket Payload to LEO | Location | Launch Year |
#          Launch Year Mon | USD/kg to LEO | 2021 Mult |
#          USD/kg to LEO CPI Adjusted | Rocket Price CPI Adjusted | Dum
#  Notes:
#   - Launch Suborbital has only one value ("Orbital") in the data =>
#     dropped (zero-variance, useless for classification)
#   - Dum column appears to be a constant flag => inspect and drop if constant
#   - Launch Status is the TARGET variable => encode as binary + multiclass
#   - Rocket Price / USD cols have ~63% nulls => keep as-is, imputation
#     is an analytical decision, not a cleaning one
def clean_launches(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
 
    # Parse datetime
    df[C.LAUNCH_TIME_COL] = pd.to_datetime(df[C.LAUNCH_TIME_COL], utc=True, errors="coerce")
 
    # Drop zero-variance columns
    if df[C.LAUNCH_SUBORBITAL_COL].nunique() <= 1:
        print(f"  [INFO] Dropping '{C.LAUNCH_SUBORBITAL_COL}': only one unique value.")
        df = df.drop(columns=[C.LAUNCH_SUBORBITAL_COL])
 
    if df[C.DUM_COL].nunique() <= 1:
        print(f"  [INFO] Dropping '{C.DUM_COL}': constant column.")
        df = df.drop(columns=[C.DUM_COL])
 
    # TARGET — binary: Success=1 / any failure=0
    df["launch_success_binary"] = (df[C.LAUNCH_STATUS_COL] == "Success").astype(int)
 
    # TARGET — multiclass OHE (kept for reference / multi-output models)
    df = _one_hot(df, C.LAUNCH_STATUS_COL, prefix="status")
 
    # Categorical identifiers — keep as string (no OHE: too high cardinality)
    # Rocket Organisation and Location will be joined/encoded in the model pipeline
 
    return df

#  Locations
#  Raw:    Orig_Addr | Country | Country_Code | Lat | Lon |
#          Operator | Launch Site | Launch Site Lat | Launch Site Lon |
#          Comb Launch Site | Comb Launch Site Lat | Comb Launch Site Lon |
#          Operator Lat | Operator Lon
#  Notes:
#   - 2 nulls in Country / Country_Code => flagged
#   - All coordinate columns already numeric, no unit stripping needed
def clean_locations(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
 
    null_country = df[C.COUNTRY_COL].isna().sum()
    if null_country > 0:
        print(f"  [WARNING] {null_country} null(s) in '{C.COUNTRY_COL}' — review manually.")
 
    # Coordinates are already float in the raw file — nothing to strip
    return df

#  Missions
#  Raw:    Launch Id | No | Payloads | Mass
#  Notes:
#   - 659 null Launch Id rows => orphan rows, flag them
#   - Mass has 1481 nulls (payload mass unknown) => keep as NaN
def clean_missions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
 
    orphans = df[C.LAUNCH_ID_COL].isna().sum()
    if orphans > 0:
        print(f"  [WARNING] {orphans} rows with null '{C.LAUNCH_ID_COL}' in Missions — kept but flagged.")
        df["orphan_row"] = df[C.LAUNCH_ID_COL].isna().astype(int)
 
    # Cast IDs to nullable int where not null
    df[C.LAUNCH_ID_COL] = pd.to_numeric(df[C.LAUNCH_ID_COL], errors="coerce").astype("Int64")
    df[C.NO_COL]        = pd.to_numeric(df[C.NO_COL],        errors="coerce").astype("Int64")
 
    return df

# =============================================
#  Join dataframes in one set
# =============================================

def _add_suffix(df: pd.DataFrame, suffix: str, exclude: list) -> pd.DataFrame:
    """Rename all columns except `exclude` by appending `suffix`."""
    return df.rename(columns={
        c: f"{c}{suffix}" for c in df.columns if c not in exclude
    })

def build_final_data(cleaned: dict) -> pd.DataFrame:
    launches  = cleaned["Launches"].copy()
    configs   = cleaned["Configs"].copy()
    families  = cleaned["Families"].copy()
    companies = cleaned["Companies"].copy()
    missions  = cleaned["Missions"].copy()
    locations = cleaned["Locations"].copy()
 
    # Reset index on tables that had set_index in loaders
    # (Families uses Family Id as index, Launches uses Launch Id as index)
    if launches.index.name == C.LAUNCH_ID_COL:
        launches = launches.reset_index()
    if families.index.name == C.FAMILY_ID_COL:
        families = families.reset_index()
 
    # 1. launches LEFT JOIN configs ON Rocket Name = Config
    # Suffix non-key configs columns to avoid name collisions
    configs_join = _add_suffix(
        configs,
        suffix="_cfg",
        exclude=[C.CONFIG_COL, C.FAMILY_ID_COL],   # keep join keys unsuffixed
    )
    df = launches.merge(
        configs_join,
        how="left",
        left_on=C.ROCKET_NAME_COL,   # launches.Rocket Name
        right_on=C.CONFIG_COL,        # configs.Config
        suffixes=("", "_cfg"),
    )
    print(f"  After launches => configs:   {df.shape}")
 
    # 2. LEFT JOIN families ON configs.Family Id = families.Family Id
    families_join = _add_suffix(
        families,
        suffix="_fam",
        exclude=[C.FAMILY_ID_COL],
    )
    df = df.merge(
        families_join,
        how="left",
        on=C.FAMILY_ID_COL,
        suffixes=("", "_fam"),
    )
    print(f"  After missions => families:           {df.shape}")
 
    # 3. LEFT JOIN companies ON Rocket Organisation = Company Name
    companies_join = _add_suffix(
        companies,
        suffix="_co",
        exclude=[C.COMPANY_NAME_COL],
    )
    df = df.merge(
        companies_join,
        how="left",
        left_on=C.ROCKET_ORGANISATION_COL,   # launches.Rocket Organisation
        right_on=C.COMPANY_NAME_COL,          # companies.Company Name
        suffixes=("", "_co"),
    )
    print(f"  After families => companies:          {df.shape}")
 
    # 4. LEFT JOIN missions ON Launch Id = Launch Id
    # Missions is a one-to-many table (multiple payloads per launch).
    # We aggregate per Launch Id so the join stays one-to-one with launches.
    missions_agg = (
        missions
        .dropna(subset=[C.LAUNCH_ID_COL])           # drop orphan rows
        .groupby(C.LAUNCH_ID_COL, as_index=False)
        .agg(
            total_payloads=(C.PAYLOADS_COL, "sum"),
            total_mass_kg=(C.MASS_COL,     "sum"),
            mission_count=(C.NO_COL,       "count"),
        )
    )
    missions_agg[C.LAUNCH_ID_COL] = missions_agg[C.LAUNCH_ID_COL].astype(int)
 
    df = df.merge(
        missions_agg,
        how="left",
        on=C.LAUNCH_ID_COL,
        suffixes=("", "_mis"),
    )
    print(f"  After companies => missions (agg):     {df.shape}")
 
    # 5. LEFT JOIN locations ON Location = Orig_Addr
    locations_join = _add_suffix(
        locations,
        suffix="_loc",
        exclude=[C.ADRESS_COL],
    )
    df = df.merge(
        locations_join,
        how="left",
        left_on=C.LOCATION_COL,   # launches.Location
        right_on=C.ADRESS_COL,    # locations.Orig_Addr
        suffixes=("", "_loc"),
    )
    print(f"  After missions => locations:          {df.shape}")
 
    return df

# =============================================
#  Check and handle noisy ID / redundant / leaking columns
# =============================================
def drop_ids(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop columns that are pure identifiers or redundant join keys.
    They carry no predictive signal.
 
    Dropped:
      - Launch Id          : row identifier, not a feature
      - Family Id          : numeric key, replaced by family-level features
      - No_cfg             : config sequence number within a family, not meaningful alone
      - Config_cfg         : rocket config name string, already represented by OHE/numeric
                             features from configs table (kept Rocket Name for traceability)
      - Company Name       : 100% duplicate of Rocket Organisation (same join key)
      - Company Country_co : raw string, already OHE'd into Company Country_*_co columns
      - Ownership_co       : raw string, already OHE'd into Ownership_*_co columns
      - Status_cfg         : raw string, already OHE'd into status_Active/Retired/Planned_cfg
      - Family_fam         : family name string, represented by Family Id and family stats
      - Orig_Addr          : raw address string used as join key, Location col is cleaner
      - Comb Launch Site_loc / Launch Site_loc : string identifiers; lat/lon kept instead
      - Operator_loc       : launch pad operator name, represented by Country_loc
      - Country_Code_loc   : ISO code redundant with Country_loc
      - Operator Lat_loc / Operator Lon_loc : operator HQ coords, not launch pad coords
      - Comb Launch Site Lat_loc / Lon_loc  : near-duplicate of Launch Site Lat/Lon
 
    Kept as intentional string columns (for reference / later OHE in pipeline):
      - Rocket Name        : useful for groupby analysis and explainability
      - Rocket Organisation: useful for groupby / OHE in model pipeline
      - Location           : useful for groupby / OHE in model pipeline
      - Country_loc        : 20 unique values — low enough to OHE in pipeline
 
    NOTE: Missions_fam / Successes_fam / Success Rate_fam are dropped here too.
    These are END-OF-DATASET snapshots (not point-in-time) for example : Falcon 9 row
    from 2010 already shows 138 total missions from 2021. This is severe target
    leakage. Point-in-time cumulative stats are rebuilt in feature_engineering.py.
    """
    df = df.copy()
 
    drop_cols = [
        # Pure identifiers / join keys
        'Launch Id',
        'Family Id',
        'No_cfg',
        'Config_cfg',           # may appear depending on suffix logic
        'Config',               # raw string from configs, already covered
        'Company Name',         # duplicate of Rocket Organisation
 
        # Raw strings already OHE'd
        'Company Country_co',
        'Ownership_co',
        'Status_cfg',
        'Family_fam',
 
        # Location string identifiers (coords are kept)
        'Orig_Addr',
        'Launch Site_loc',
        'Comb Launch Site_loc',
        'Operator_loc',
        'Country_Code_loc',
 
        # Redundant / non-pad coordinates
        'Operator Lat_loc',
        'Operator Lon_loc',
        'Comb Launch Site Lat_loc',
        'Comb Launch Site Lon_loc',
 
        # LEAKAGE: end-of-dataset family stats (not point-in-time)
        'Missions_fam',
        'Successes_fam',
        'Partial Failures_fam',
        'Failures_fam',
        'Success Streak_fam',
        'Success Rate_fam',
    ]
 
    dropped = [c for c in drop_cols if c in df.columns]
    df = df.drop(columns=dropped)
    print(f"  Dropped {len(dropped)} ID / redundant / leaking columns")
 
    return df

# =============================================
#  Check and handle nulls
# =============================================
def handle_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """
    Null handling strategy — driven by data investigation results:
 
    GROUP 1 — DROP redundant duplicate columns
      Rocket Price, Rocket Payload to LEO, USD/kg to LEO, USD/kg to LEO CPI Adjusted,
      Rocket Price CPI Adjusted are 100% identical to their _cfg / derived counterparts.
      Keeping both would give the model duplicate signal and inflate feature importance.
      => Dropping the launches-side duplicates; keeping the _cfg versions.
 
    GROUP 2 — DROP columns that leak the target or are post-hoc
      status_Success, status_Failure, status_Partial Failure, status_Prelaunch Failure
      are OHE columns derived directly from Launch Status (the target).
      Launch Status itself (string) is also dropped for the same reason.
      => These must never be features in a classifier.
 
    GROUP 3 — DROP 2021 Mult
      Inflation multiplier used to compute CPI columns. It is not a feature of the
      rocket or launch — it is a data-processing artifact. Not predictive.
 
    GROUP 4 — DROP high-cardinality string identifiers
      Config Id_cfg, Config, Orig_Addr, Launch Year Mon — too many categories to OHE,
      and the information they carry is already covered by numeric/OHE features from
      the same table. Keeping them as raw strings would break sklearn pipelines.
 
    GROUP 5 — FILL missions nulls with 0
      611 launches (mostly pre-1970s) have no mission record. This is a data-coverage
      gap, not a true "no payload". But for the model, 0 is a better signal than NaN
      (the launch happened; we just don't have payload details). A flag column is added
      so the model can learn that "no record" is itself informative.
 
    GROUP 6 — FILL Stages_cfg (1 null: Atlas SLV-3)
      Median imputation — 1 row, well-known rocket, safe to impute.
 
    GROUP 7 — KEEP remaining nulls as-is, add missingness flags
      Price_cfg (63% null), physical specs (partial nulls), total_mass_kg (10% null).
      Imputing 63% of a column with mean/median would destroy its signal entirely.
      Instead: add a binary flag column (col_missing = 1/0) so tree-based models
      can learn from the pattern of missingness itself, then fill NaN with -1
      (a sentinel value outside the natural range — distinguishable from real zeros).
      NOTE: for linear models or neural networks, need proper imputation + scaling instead.
              The -1 flag is specifically appropriate for tree-based classifiers
              (Random Forest, XGBoost, LightGBM) which can split on it meaningfully.
 
    GROUP 8 — FILL location nulls (2 rows)
      Only 2 rows. Fill Country_loc / Country_Code_loc with 'Unknown'.
    """
    df = df.copy()
 
    # GROUP 1: Drop redundant duplicate columns
    drop_redundant = [
        'Rocket Price',                  # identical to Price_cfg
        'Rocket Payload to LEO',         # identical to Payload to LEO (kg)_cfg
        'USD/kg to LEO',                 # derived from Price (post-hoc cost metric)
        'USD/kg to LEO CPI Adjusted',    # same, CPI-adjusted
        'Rocket Price CPI Adjusted',     # identical to Price_cfg * 2021 Mult
    ]
    df = df.drop(columns=[c for c in drop_redundant if c in df.columns])
    print(f"  Dropped {len(drop_redundant)} redundant columns")
 
    # GROUP 2: Drop target-leaking columns
    drop_leaking = [
        'Launch Status',
        'status_Success',
        'status_Failure',
        'status_Partial Failure',
        'status_Prelaunch Failure',
    ]
    df = df.drop(columns=[c for c in drop_leaking if c in df.columns])
    print(f"  Dropped {len(drop_leaking)} target-leaking columns")
 
    # GROUP 3: Drop data-processing artifact
    df = df.drop(columns=['2021 Mult'], errors='ignore')
    print("  Dropped '2021 Mult' (inflation multiplier artifact)")
 
    # GROUP 4: Drop high-cardinality string identifiers
    drop_hc_strings = ['Config Id_cfg', 'Config', 'Orig_Addr', 'Launch Year Mon']
    df = df.drop(columns=[c for c in drop_hc_strings if c in df.columns])
    print(f"  Dropped {len(drop_hc_strings)} high-cardinality string columns")
 
    # GROUP 5: Missions nulls => 0 + missingness flag
    mission_cols = ['total_payloads', 'total_mass_kg', 'mission_count']
    df['mission_data_missing'] = df['mission_count'].isna().astype(int)
    for col in mission_cols:
        df[col] = df[col].fillna(0)
    print(f"  Filled {mission_cols} nulls with 0, added 'mission_data_missing' flag")
 
    # GROUP 6: Stages_cfg — 1 null => median imputation
    stages_median = df['Stages_cfg'].median()
    df['Stages_cfg'] = df['Stages_cfg'].fillna(stages_median)
    print(f"  Filled 1 null in 'Stages_cfg' with median ({stages_median})")
 
    # GROUP 7: High-null numeric columns => flag + sentinel "-1"
    sentinel_cols = [
        'Price_cfg',
        'Liftoff Thrust (kN)_cfg',
        'Payload to LEO (kg)_cfg',
        'Payload to GTO (kg)_cfg',
        'Rocket Height (m)_cfg',
        'Fairing Diameter (m)_cfg',
        'Fairing Height (m)_cfg',
        'total_mass_kg',
    ]
    flagged = 0
    for col in sentinel_cols:
        if col not in df.columns:
            continue
        null_count = df[col].isna().sum()
        if null_count > 0:
            flag_col = col.replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_') + '_missing'
            df[flag_col] = df[col].isna().astype(int)
            df[col] = df[col].fillna(-1)
            flagged += 1
    print(f"Added missingness flags + sentinel -1 fill for {flagged} numeric columns")
 
    # GROUP 8: Location nulls => 'Unknown'
    for col in ['Country_loc', 'Country_Code_loc']:
        if col in df.columns:
            df[col] = df[col].fillna('Unknown')
    print("Filled 2 location nulls with 'Unknown'")
 
    # Final null check
    remaining_nulls = df.isna().sum()
    remaining_nulls = remaining_nulls[remaining_nulls > 0]
    if len(remaining_nulls) == 0:
        print("No remaining nulls")
    else:
        print(f"[INFO] Remaining nulls after handling ({len(remaining_nulls)} cols):")
        print(remaining_nulls.to_string())
 
    return df

# =============================================
#  Save individual clean CSVs
# =============================================

def save_clean(df: pd.DataFrame):
    C.DATA_PATH_INPUTS_CLEAN.mkdir(parents=True, exist_ok=True)
    path = C.DATA_PATH_INPUTS_CLEAN / "final_data.csv"
    df.to_csv(path, index=False)
    print(f"\n  Saved => {path}")
    print(f"  Shape: {df.shape[0]} rows × {df.shape[1]} columns")


# =============================================
#  Master runner
# =============================================

def run_cleaning():
    print("=" * 60)
    print("Starting cleaning pipeline")
    print("=" * 60)
 
    steps = [
        ("Companies", load_raw_companies, clean_companies),
        ("Configs",   load_raw_configs,   clean_configs),
        ("Families",  load_raw_families,  clean_families),
        ("Launches",  load_raw_launches,  clean_launches),
        ("Locations", load_raw_locations, clean_locations),
        ("Missions",  load_raw_missions,  clean_missions),
    ]
 
    cleaned = {}
    for name, loader, cleaner in steps:
        print(f"\n[{name}]")
        df_raw   = loader()
        df_clean = cleaner(df_raw)
        cleaned[name] = df_clean
        print(f"  Shape: {df_raw.shape} => {df_clean.shape}")
 
    print("\n[Joining all tables]")
    df_final = build_final_data(cleaned)

    print("\n[Dropping ID / redundant / leaking columns]")
    df_final = drop_ids(df_final)

    print("\n[Handling nulls]")
    df_final = handle_nulls(df_final)
 
    print("\n[Saving]")
    save_clean(df_final)
 
    print("\n" + "=" * 60)
    print("Pipeline complete.")
    print(f"  Output: {C.DATA_PATH_INPUTS_CLEAN / 'final_data.csv'}")
    print("=" * 60)
    return cleaned


if __name__ == "__main__":
    run_cleaning()