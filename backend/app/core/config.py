"""Central configuration: paths, model registry, and the keyword list used
to auto-detect a valid "student performance" target column.

Ported verbatim (values unchanged) from the original Streamlit project's
utils/config.py so the migrated application keeps identical behaviour for
target detection and model dispatch.
"""

import os

# --- Paths -------------------------------------------------------------
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
UPLOADS_DIR = os.path.join(BACKEND_DIR, "uploads")
MODELS_DIR = os.path.join(BACKEND_DIR, "models")
OUTPUTS_DIR = os.path.join(BACKEND_DIR, "outputs")

for _d in (UPLOADS_DIR, MODELS_DIR, OUTPUTS_DIR):
    os.makedirs(_d, exist_ok=True)

# --- Model registry ----------------------------------------------------
# Single source of truth for model choices, keyed by problem type. Both the
# training API (dropdown options exposed to the frontend) and app.ml.training
# (dispatch logic) read from here so the two can never drift apart.
MODEL_OPTIONS = {
    "classification": ["Logistic Regression", "Decision Tree", "Random Forest"],
    "regression": ["Linear Regression", "Decision Tree Regressor", "Random Forest Regressor"],
}

# --- Target-column auto-detection --------------------------------------
# Ordered by priority (most specific / most common first). A column name is
# treated as a candidate target if, after stripping non-alphanumeric
# characters, it contains one of these keywords (or vice versa).
TARGET_COLUMN_KEYWORDS = [
    "exam_score", "examscore", "final_score", "test_score", "final_grade",
    "placement_status", "placed",
    "performance_index", "performance",
    "pass_fail", "passfail", "passed",
    "grade", "gpa", "cgpa",
    "result", "outcome",
    "target", "label",
]

# --- Charting ----------------------------------------------------------
COLOR_PALETTE = [
    "#16A34A",  # primary (green)
    "#22C55E",  # secondary (accent green)
    "#15803D",  # dark green
    "#F59E0B",  # warning
    "#EF4444",  # danger
    "#4ADE80",  # line (for trend)
]

CHART_TYPES = ["Histogram", "Box Plot"]

# --- Upload limits / security ------------------------------------------
ALLOWED_EXTENSIONS = {"csv", "xlsx"}
MAX_UPLOAD_BYTES = 200 * 1024 * 1024  # 200 MB

# --- Visualisation sampling --------------------------------------------
# Charts that need per-row coordinates (scatter, residuals) use a
# deterministic sample above this row count so the browser never receives
# 100k+ points. Training data is NEVER sampled — see app/ml/training.py.
SCATTER_SAMPLE_LIMIT = 5000
SCATTER_SAMPLE_SEED = 42
MAX_HISTOGRAM_BINS = 40
MAX_CATEGORY_BARS = 40
MAX_ROC_POINTS = 1000
MAX_HEATMAP_COLUMNS = 40

# --- In-memory cache sizes ---------------------------------------------
DATASET_CACHE_SIZE = 8
MODEL_CACHE_SIZE = 8

# --- Risk / confidence semantics (features 18, 19, 20, 27, 34, 37, 52) --
# Shared by app.services.dashboard_service (pass/fail dashboard split) and
# app.ml.risk (at-risk classification), so the two features can never
# disagree about which class label means "passing".
POSITIVE_LABELS = {"pass", "passed", "placed", "yes", "1", "true", "good"}
NEGATIVE_LABELS = {"fail", "failed", "not placed", "no", "0", "false", "poor"}

# A predicted failing outcome at or above this confidence is High risk;
# below it, Medium. A predicted passing outcome at or above this confidence
# is Low risk; below it, Medium. See app/ml/risk.py for the full rule.
RISK_HIGH_CONFIDENCE = 0.66

# Confidence tiers used for the Prediction Explorer's confidence filter, the
# Confidence Analysis chart, and the Low-Confidence Review Queue preset.
CONFIDENCE_HIGH = 0.75
CONFIDENCE_MEDIUM = 0.5
