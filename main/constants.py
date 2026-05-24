# third parties imports
from pathlib import Path


class Constant:
    ROOT_DIR = Path(__file__).resolve().parent.parent
    DATA_PATH = ROOT_DIR / 'data'
    DATA_PATH_INPUTS = DATA_PATH / 'inputs'
    DATA_PATH_INPUTS_RAW = DATA_PATH_INPUTS / 'raw'
    DATA_PATH_INPUTS_CLEAN = DATA_PATH_INPUTS / 'clean'
    DATA_PATH_OUTPUTS = DATA_PATH / 'outputs'
    DATA_PATH_OUTPUTS_EVL_REP = DATA_PATH_OUTPUTS / 'evaluation_reports'
    DATA_PATH_OUTPUTS_FIG = DATA_PATH_OUTPUTS / 'figures'
    DATA_PATH_OUTPUTS_SUM = DATA_PATH_OUTPUTS / 'summaries'
    DATA_PATH_OUTPUTS_TAB = DATA_PATH_OUTPUTS / 'tables'
    
    # Raw data inputs
    # - companies
    COMPANIES_FILENAME = 'Companies.csv'
    COMPANY_NAME_COL = 'Company Name'
    COMPANY_COUNTRY_COL = 'Company Country'
    OWNERSHIP_COL = 'Ownership'

    # - configs
    CONFIGS_FILENAME = 'Configs.csv'
    FAMILY_ID_COL = 'Family Id'
    NO_COL = 'No'
    CONFIG_COL = 'Config'
    STATUS_COL = 'Status'
    PRICE_COL = 'Price'
    LIFTOFF_THRUST_COL = 'Liftoff Thrust'
    CONF_PAYLOAD_LEO_COL = 'Payload to LEO'
    CONF_PAYLOAD_GTO_COL = 'Payload to GTO'
    STAGES_COL = 'Stages'
    STRAPS_ONS_COL = 'Strap-ons'
    ROCKET_HEIGHT_COL = 'Rocket Height'
    FAIRING_DIAMETER_COL = 'Fairing Diameter'
    FAIRING_HEIGHT_COL = 'Fairing Height'

    # - families
    FAMILIES_FILENAME = 'Families.csv'
    FAMILY_ID_COL = 'Family Id'
    FAMILY_NAME_COL = 'Family'
    MISSIONS_COL = 'Missions'
    SUCCESSES_COL = 'Successes'
    PARTIAL_FAILURES_COL = 'Partial Failures'
    FAILURES_COL = 'Failures'
    SUCCESS_STREAK_COL = 'Success Streak'
    SUCCESS_RATE_COL = 'Success Rate'

    # - launches
    LAUNCHES_FILENAME = 'Launches.csv'
    LAUNCH_ID_COL = 'Launch Id'
    LAUNCH_TIME_COL = 'Launch Time'
    LAUNCH_STATUS_COL = 'Launch Status'
    LAUNCH_SUBORBITAL_COL = 'Launch Suborbital'
    ROCKET_NAME_COL = 'Rocket Name'
    ROCKET_ORGANISATION_COL = 'Rocket Organisation'
    ROCKET_PRICE_COL = 'Rocket Price'
    PAYLOAD_LEO_COL = 'Rocket Payload to LEO'
    LOCATION_COL = 'Location'
    LAUNCH_YEAR_COL = 'Launch Year'
    LAUNCH_YEAR_MON_COL = 'Launch Year Mon'
    USD_KG_LEO_COL = 'USD/kg to LEO'
    MULT_2021_COL = '2021 Mult'
    USD_KG_LEO_CPI_COL = 'USD/kg to LEO CPI Adjusted'
    ROCKET_PRICE_CPI_COL = 'Rocket Price CPI Adjusted'
    DUM_COL = 'Dum'
    
     # - locations
    LOCATIONS_FILENAME = 'Locations.csv'
    ADRESS_COL = 'Orig_Addr'
    COUNTRY_COL = 'Country'
    COUNTRY_CODE_COL = 'Country_Code'
    LAT_COL = 'Lat'
    LON_COL = 'Lon'
    OPERATOR_COL = 'Operator'
    LAUNCH_SITE_COL = 'Launch Site'
    LAUNCH_SITE_LAT_COL = 'Launch Site Lat'
    LAUNCH_SITE_LON_COL = 'Launch Site Lon'
    COMB_LAUN_SITE_COL = 'Comb Launch Site'
    COMB_LAUN_SITE_LAT_COL = 'Comb Launch Site Lat'
    COMB_LAUN_SITE_LON_COL = 'Comb Launch Site Lon'
    OPERATOR_LAT_COL = 'Operator Lat'
    OPERATOR_LON_COL = 'Operator Lon'

    # - missions
    MISSIONS_FILENAME = 'Missions.csv'
    LAUNCH_ID_COL = 'Launch Id'
    NO_COL = 'No'
    PAYLOADS_COL = 'Payloads'
    MASS_COL = 'Mass'

    # Clean and model data inputs
    CLEAN_DATA_FILENAME = 'final_data.csv'
    MODEL_DATA_FILENAME = 'model_data.csv'

    # Rating scale 
    #TODO: Must be delete
    RATINGS_SCALE = (0.5,5)  # ratings scale as a tuple (min_value, max_value)

    # Top N for chart distribution ploting
    TOP_N = 15 # for high-cardinality string columns
    CARDINALITY_THRESHOLD = 20

    # Correlation analysis threshold to drop columns
    CORR_THRESHOLD = 0.85
    # Manually forced drops (one side of a perfect-inverse OHE pair, or exact duplicates)
    _FORCED_DROPS = [
        'Ownership_State_co',       # perfect inverse of Ownership_Private_co
        'status_Retired_cfg',       # perfect inverse of status_Active_cfg
        'status_Planned_cfg',       # near-zero variance
        'Lat_loc',                  # Duplicate of Launch Site Lat_loc
        'Lon_loc',                  # Duplicate of Launch Site Lon_loc
        'launch_decade',            # cor =0.987 with Launch Year; Launch Year is continuous
    ]