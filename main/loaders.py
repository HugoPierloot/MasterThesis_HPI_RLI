# third parties imports
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# local imports
from constants import Constant as C

# Loaders imports:
def load_raw_companies():
    full_path = C.DATA_PATH_INPUTS_RAW / C.COMPANIES_FILENAME
    table = "Companies"
    file = f"'{table}' file"
    if not full_path.exists():
        print(f"[ERROR] {file} not found. Make sure to have '{C.COMPANIES_FILENAME}' at this path: {full_path}")
        return None
    try:
        df_loaded = pd.read_csv(full_path)
    except Exception as e:
        print(f"[ERROR] Failed to load or convert {file}: {e}")
        return None
    return df_loaded

def load_raw_configs():
    full_path = C.DATA_PATH_INPUTS_RAW / C.CONFIGS_FILENAME
    table = "Configs"
    file = f"'{table}' file"
    if not full_path.exists():
        print(f"[ERROR] {file} not found. Make sure to have '{C.CONFIGS_FILENAME}' at this path: {full_path}")
        return None
    try:
        df_loaded = pd.read_csv(full_path)
    except Exception as e:
        print(f"[ERROR] Failed to load or convert {file}: {e}")
        return None
    return df_loaded

def load_raw_families():
    full_path = C.DATA_PATH_INPUTS_RAW / C.FAMILIES_FILENAME
    table = "Families"
    file = f"'{table}' file"
    if not full_path.exists():
        print(f"[ERROR] {file} not found. Make sure to have '{C.FAMILIES_FILENAME}' at this path: {full_path}")
        return None
    try:
        df_loaded = pd.read_csv(full_path)
        df_loaded = df_loaded.set_index(C.FAMILY_ID_COL)
    except Exception as e:
        print(f"[ERROR] Failed to load or convert {file}: {e}")
        return None
    return df_loaded

def load_raw_launches():
    full_path = C.DATA_PATH_INPUTS_RAW / C.LAUNCHES_FILENAME
    table = "Launches"
    file = f"'{table}' file"
    if not full_path.exists():
        print(f"[ERROR] {file} not found. Make sure to have '{C.LAUNCHES_FILENAME}' at this path: {full_path}")
        return None
    try:
        df_loaded = pd.read_csv(full_path)
        df_loaded = df_loaded.set_index(C.LAUNCH_ID_COL)
    except Exception as e:
        print(f"[ERROR] Failed to load or convert {file}: {e}")
        return None
    return df_loaded

def load_raw_locations():
    full_path = C.DATA_PATH_INPUTS_RAW / C.LOCATIONS_FILENAME
    table = "Locations"
    file = f"'{table}' file"
    if not full_path.exists():
        print(f"[ERROR] {file} not found. Make sure to have '{C.LOCATIONS_FILENAME}' at this path: {full_path}")
        return None
    try:
        df_loaded = pd.read_csv(full_path)
    except Exception as e:
        print(f"[ERROR] Failed to load or convert {file}: {e}")
        return None
    return df_loaded

def load_raw_missions():
    full_path = C.DATA_PATH_INPUTS_RAW / C.MISSIONS_FILENAME
    table = "Missions"
    file = f"'{table}' file"
    if not full_path.exists():
        print(f"[ERROR] {file} not found. Make sure to have '{C.MISSIONS_FILENAME}' at this path: {full_path}")
        return None
    try:
        df_loaded = pd.read_csv(full_path)
    except Exception as e:
        print(f"[ERROR] Failed to load or convert {file}: {e}")
        return None
    return df_loaded

def load_clean_data():
    full_path = C.DATA_PATH_INPUTS_CLEAN / C.CLEAN_DATA_FILENAME
    table = "Cleaned data"
    file = f"'{table}' file"
    if not full_path.exists():
        print(f"[ERROR] {file} not found. Make sure to have '{C.CLEAN_DATA_FILENAME}' at this path: {full_path}")
        return None
    try:
        df_loaded = pd.read_csv(full_path)
    except Exception as e:
        print(f"[ERROR] Failed to load or convert {file}: {e}")
        return None
    return df_loaded

def load_model_data():
    full_path = C.DATA_PATH_INPUTS_CLEAN / C.MODEL_DATA_FILENAME
    table = "Model data"
    file = f"'{table}' file"
    if not full_path.exists():
        print(f"[ERROR] {file} not found. Make sure to have '{C.MODEL_DATA_FILENAME}' at this path: {full_path}")
        return None
    try:
        df_loaded = pd.read_csv(full_path)
    except Exception as e:
        print(f"[ERROR] Failed to load or convert {file}: {e}")
        return None
    return df_loaded

def plot_distributions(named_dfs: dict, stage: str):
    """
    Plot data distributions for all columns of all DataFrames.
    Each chart shows count labels on bars and a null-value count annotation.

    Args:
        named_dfs (dict): {"DataFrame name": df, ...}
        stage (str): "raw" or "clean" — used for folder and file naming
    """
    output_dir = C.DATA_PATH_OUTPUTS_FIG / stage
    output_dir.mkdir(parents=True, exist_ok=True)

    for df_name, df in named_dfs.items():
        n_cols = len(df.columns)
        fig, axes = plt.subplots(nrows=n_cols, ncols=1, figsize=(10, 4 * n_cols))

        if n_cols == 1:
            axes = [axes]

        fig.suptitle(f"{df_name} — {stage.capitalize()} distributions", fontsize=14, fontweight="bold", y=1.01)

        for ax, col in zip(axes, df.columns):
            series = df[col]
            null_count = series.isna().sum()
            null_label = f"Nulls: {null_count}"
            series_clean = series.dropna()

            if pd.api.types.is_numeric_dtype(series_clean):
                # --- Numerical: histogram
                counts, bin_edges, patches = ax.hist(series_clean, bins=30, edgecolor="white", color="steelblue")
                ax.set_title(f"{col}  [numerical]")
                ax.set_xlabel(col)
                ax.set_ylabel("Count")

                # Count labels on each bar
                for count, patch in zip(counts, patches):
                    if count > 0:
                        ax.text(
                            patch.get_x() + patch.get_width() / 2,
                            patch.get_height(),
                            f"{int(count)}",
                            ha="center", va="bottom",
                            fontsize=7, color="black"
                        )

            else:
                # --- Categorical / string: bar chart
                value_counts = series_clean.value_counts()
                cardinality = len(value_counts)

                if cardinality > C.CARDINALITY_THRESHOLD:
                    value_counts = value_counts.head(C.TOP_N)
                    ax.set_title(f"{col}  [string — top {C.TOP_N} of {cardinality}]")
                else:
                    ax.set_title(f"{col}  [categorical — {cardinality} unique]")

                bars = ax.bar(value_counts.index.astype(str), value_counts.values, color="steelblue", edgecolor="white")
                ax.set_xlabel(col)
                ax.set_ylabel("Count")
                ax.tick_params(axis="x", rotation=45)

                # Count labels on each bar
                for bar in bars:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height(),
                        f"{int(bar.get_height())}",
                        ha="center", va="bottom",
                        fontsize=8, color="black"
                    )

            # Null count annotation: always displayed, top-right corner
            null_color = "red" if null_count > 0 else "green"
            ax.annotate(
                null_label,
                xy=(1, 1), xycoords="axes fraction",
                xytext=(-8, -8), textcoords="offset points",
                ha="right", va="top",
                fontsize=9, fontweight="bold", color=null_color,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=null_color, alpha=0.8)
            )

            ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

        plt.tight_layout()
        filename = f"{df_name.lower()}_distributions_{stage}.png"
        fig.savefig(output_dir / filename, bbox_inches="tight", dpi=150)
        plt.show()
        plt.close(fig)
        print(f"Saved: {output_dir / filename}")

# Evaluation reports export:
# Must be adapted to the master thesis project (originally made in Recommender Systems Project).
def export_evaluation_report(df):
    """ Export the report to the evaluation folder.
    The name of the report is versioned using today's date
    Args:
        df (pd.DataFrame): the evaluation report dataframe.
    """
    try:
        today = datetime.today().strftime("%Y_%m_%d_%Hh%M")  # -- date format YYYY_MM_DD_HHhMM
        filename = f"{today}.csv" # -- name of the report
        full_path = C.EVALUATION_PATH / filename
        df.to_csv(full_path, index=True)

        print(f"\nEvaluation report exported to: {full_path}")
    except Exception as e:
        print(f"[ERROR] Failed to export evaluation report: {e}")
    pass

# Call the function to load the ratings dataset only if this script is run directly (test purpose)
if __name__ == "__main__":
    df_raw_companies = load_raw_companies()