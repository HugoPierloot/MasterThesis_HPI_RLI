# ─────────────────────────────────────────────────────────────
#  Tree_models
#
#  Tree-based ensemble classifiers for rocket launch failure
#  prediction on imbalanced binary data.
#
#  Models:
#    1. Random Forest     — bagging + class_weight='balanced'
#    2. XGBoost           — gradient boosting + scale_pos_weight
#    3. AdaBoost          — adaptive boosting (base: Decision Tree)
#    4. RUSBoost          — Random Under-Sampling + AdaBoost
#
#  All models return sklearn-compatible estimators.
#  Hyperparameters are set to sensible defaults for imbalanced data.
# ─────────────────────────────────────────────────────────────

from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
)
from sklearn.tree import DecisionTreeClassifier
from imblearn.ensemble import RUSBoostClassifier
from xgboost import XGBClassifier


# ─────────────────────────────────────────────────────────────
#  1. Random Forest
#
#  class_weight='balanced' makes each sample's weight inversely
#  proportional to class frequency equivalent to multiplying
#  minority loss by ~10.4 (the imbalance ratio).
#  No SMOTE needed here since RF handles it internally.
# ─────────────────────────────────────────────────────────────

def build_random_forest(random_state: int = 42) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators      = 300,
        max_depth         = None,        # grow full trees
        min_samples_leaf  = 2,           # slight regularisation
        max_features      = "sqrt",      # standard for classification
        class_weight      = "balanced",  # handle imbalance
        n_jobs            = -1,
        random_state      = random_state,
    )


# ─────────────────────────────────────────────────────────────
#  2. XGBoost
#
#  scale_pos_weight = n_negative / n_positive (approx 10.4 for
#  raw data; set to 3.0 when training on SMOTE-resampled data
#  since SMOTE has already partially balanced the classes).
#  eval_metric='aucpr' optimises for area under precision-recall
#  curve more appropriate than logloss for imbalanced data.
# ─────────────────────────────────────────────────────────────

def build_xgboost(
    scale_pos_weight: float = 3.0,
    random_state: int = 42,
) -> XGBClassifier:
    return XGBClassifier(
        n_estimators      = 300,
        max_depth         = 6,
        learning_rate     = 0.05,
        subsample         = 0.8,
        colsample_bytree  = 0.8,
        scale_pos_weight  = scale_pos_weight,
        eval_metric       = "aucpr",
        use_label_encoder = False,
        random_state      = random_state,
        n_jobs            = -1,
        verbosity         = 0,
    )


# ─────────────────────────────────────────────────────────────
#  3. AdaBoost
#
#  Uses a shallow Decision Tree as base estimator (depth=2) 
#  weak learners are preferred in boosting to avoid overfitting.
#  class imbalance handled via sample_weight adjustment internally
#  (AdaBoost reweights misclassified samples, which naturally
#  focuses more on minority class failures).
# ─────────────────────────────────────────────────────────────

def build_adaboost(random_state: int = 42) -> AdaBoostClassifier:
    base = DecisionTreeClassifier(
        max_depth    = 2,
        class_weight = "balanced",
        random_state = random_state,
    )
    return AdaBoostClassifier(
        estimator         = base,
        n_estimators      = 200,
        learning_rate     = 0.1,
        #algorithm         = "SAMME",
        random_state      = random_state,
    )


# ─────────────────────────────────────────────────────────────
#  4. RUSBoost
#
#  Hybrid approach: Random Under-Sampling + AdaBoost.
#  At each boosting round, the majority class is randomly
#  undersampled before fitting the base learner. This directly
#  addresses imbalance without creating synthetic data.
#  Particularly relevant per your literature review (H. Chen & Ji,
#  2021) which recommends RUSBoost for IR > 60 (our IR ≈ 10.4,
#  also AdaBoost since IR < 70 per the flowchart in Appendix 3).
# ─────────────────────────────────────────────────────────────

def build_rusboost(random_state: int = 42) -> RUSBoostClassifier:
    return RUSBoostClassifier(
        n_estimators      = 200,
        learning_rate     = 0.1,
        sampling_strategy = "auto",   # undersample majority to match minority
        random_state      = random_state,
    )


# ─────────────────────────────────────────────────────────────
#  Registry used by train.py to iterate all models
# ─────────────────────────────────────────────────────────────

def get_all_models(random_state: int = 42) -> dict:
    """
    Returns a dict of {model_name: unfitted_estimator}.
    Import this in train.py and evaluation.py.
    """
    return {
        "RandomForest" : build_random_forest(random_state),
        "XGBoost"      : build_xgboost(random_state=random_state),
        "AdaBoost"     : build_adaboost(random_state),
        "RUSBoost"     : build_rusboost(random_state),
    }