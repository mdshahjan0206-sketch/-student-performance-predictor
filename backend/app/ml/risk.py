"""Risk and confidence derivation shared by every student-level feature:
At-Risk Detection (18), Individual Student Profile (19), Prediction Explorer
(20), Risk Heatmap (27), Student Comparison (34), Confidence Analysis (37)
and the Low-Confidence Review Queue (52).

Everything here reads an already-computed batch prediction result (the same
DataFrame app.ml.prediction.predict_batch produced: the uploaded columns
plus ``Prediction`` and, for classifiers that expose probabilities,
``Confidence`` / ``Predicted Class``). Nothing is retrained, re-scored or
invented — this module only labels rows that the existing model already
scored.

RISK RULE (kept deliberately simple and stated in plain language so the UI
can show exactly how a number was derived):

  * Classification, when the target's two classes are recognisably
    pass/fail-shaped (the same POSITIVE_LABELS / NEGATIVE_LABELS keyword
    sets app.services.dashboard_service uses for the dashboard's pass/fail
    split): a predicted failing outcome with confidence >= RISK_HIGH_CONFIDENCE
    is High risk; a failing prediction below that confidence, or a passing
    prediction below that confidence, is Medium risk; a confidently passing
    prediction is Low risk.
  * Regression, on a performance-style numeric target (this app only ever
    trains on score/grade/performance-shaped targets — see
    TARGET_COLUMN_KEYWORDS, where higher is always better): risk is the
    predicted value's percentile within this batch. Bottom quartile = High,
    next quartile = Medium, above the median = Low.
  * Anything else — multi-class classification, or two classes that are not
    recognisably pass/fail, or a classifier with no predict_proba — has no
    defensible "risk" without guessing which outcome is bad. Those cases
    report unsupported with a plain-language reason instead of a fabricated
    category.
"""

import re

import numpy as np
import pandas as pd

from app.core.config import CONFIDENCE_HIGH, CONFIDENCE_MEDIUM, NEGATIVE_LABELS, POSITIVE_LABELS, RISK_HIGH_CONFIDENCE

RISK_LEVELS = ("high", "medium", "low")

# Column-name keywords used only to group a student's own columns into
# display sections on the profile/comparison pages (Academic / Behavioral /
# Demographic). This never changes which columns exist or their values —
# it only decides which heading a column is shown under, and anything that
# matches nothing falls back to "Other".
_ACADEMIC_KEYWORDS = [
    "score", "grade", "gpa", "cgpa", "exam", "test", "study", "attendance",
    "assignment", "class", "course", "performance", "result", "mark",
    "credit", "semester", "failures", "tutoring", "homework",
]
_BEHAVIORAL_KEYWORDS = [
    "absence", "behavior", "behaviour", "discipline", "activity",
    "extracurricular", "participation", "screentime", "screen_time",
    "internet", "sleep", "motivation", "romantic", "freetime", "free_time",
    "goout", "go_out", "alcohol", "health",
]
_DEMOGRAPHIC_KEYWORDS = [
    "age", "gender", "sex", "address", "family", "parent", "guardian",
    "income", "school", "urban", "rural", "travel", "location", "famsize",
    "pstatus", "medu", "fedu", "mjob", "fjob", "reason", "guardian",
]
_ID_KEYWORDS = [
    "student_id", "studentid", "roll_no", "rollno", "registration",
    "reg_no", "regno", "admission_no", "admissionno", "enrollment",
    "enrollmentno", "student_name", "studentname", "name", "id",
]


def _normalize(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def find_identifier_column(columns) -> str | None:
    """Best-effort match for a human-readable student ID/name column, used
    only for DISPLAY — row lookups always use the row index (see
    app.services.student_service), so this never needs to be unique."""
    normalized = {c: _normalize(c) for c in columns}
    for keyword in _ID_KEYWORDS:
        for col, ncol in normalized.items():
            if keyword == ncol or keyword in ncol:
                return col
    return None


def categorize_column(name: str) -> str:
    n = _normalize(name)
    for kw in _ACADEMIC_KEYWORDS:
        if _normalize(kw) in n:
            return "academic"
    for kw in _BEHAVIORAL_KEYWORDS:
        if _normalize(kw) in n:
            return "behavioral"
    for kw in _DEMOGRAPHIC_KEYWORDS:
        if _normalize(kw) in n:
            return "demographic"
    return "other"


def _is_pass_fail(labels) -> bool:
    lowered = {str(v).strip().lower() for v in labels}
    return bool(lowered & POSITIVE_LABELS) and bool(lowered & NEGATIVE_LABELS)


def risk_capability(problem_type: str, prediction_df: pd.DataFrame, has_confidence: bool) -> dict:
    """Whether/why risk can be computed for this prediction run."""
    if "Prediction" not in prediction_df.columns:
        return {"supported": False, "reason": "No predictions are available yet.", "basis": None}

    if problem_type == "regression":
        values = pd.to_numeric(prediction_df["Prediction"], errors="coerce").dropna()
        if values.empty or values.nunique() < 4:
            return {
                "supported": False,
                "reason": "There are not enough distinct predicted values in this batch to form risk quartiles.",
                "basis": None,
            }
        return {
            "supported": True,
            "reason": None,
            "basis": (
                "Risk is the predicted value's percentile within this batch: the bottom 25% of "
                "predictions is High risk, the next 25% is Medium risk, and everything above the "
                "median is Low risk."
            ),
        }

    if problem_type == "classification":
        if not has_confidence:
            return {
                "supported": False,
                "reason": "This model does not produce prediction probabilities, so confidence-based risk cannot be computed.",
                "basis": None,
            }
        labels = prediction_df["Prediction"].dropna().astype(str).str.strip().unique().tolist()
        if len(labels) != 2 or not _is_pass_fail(labels):
            return {
                "supported": False,
                "reason": (
                    "The target's predicted classes are not recognisable as a pass/fail-style outcome, "
                    "so risk cannot be derived without guessing which class is unfavourable."
                ),
                "basis": None,
            }
        return {
            "supported": True,
            "reason": None,
            "basis": (
                f"A predicted failing outcome with confidence at or above {int(RISK_HIGH_CONFIDENCE * 100)}% is "
                f"High risk. A failing prediction below that confidence, or a passing prediction below "
                f"{int(RISK_HIGH_CONFIDENCE * 100)}% confidence, is Medium risk. A confidently passing "
                f"prediction is Low risk."
            ),
        }

    return {"supported": False, "reason": "Unrecognised problem type.", "basis": None}


def compute_risk_series(problem_type: str, prediction_df: pd.DataFrame) -> pd.Series:
    """Return a Series of 'high' / 'medium' / 'low' (or None) aligned to
    prediction_df's index. Call risk_capability() first and only call this
    when it reports supported=True."""
    if problem_type == "regression":
        values = pd.to_numeric(prediction_df["Prediction"], errors="coerce")
        q25, q50 = values.quantile(0.25), values.quantile(0.5)

        def bucket_reg(v):
            if pd.isna(v):
                return None
            if v <= q25:
                return "high"
            if v <= q50:
                return "medium"
            return "low"

        return values.apply(bucket_reg)

    lowered = prediction_df["Prediction"].astype(str).str.strip().str.lower()
    confidence = prediction_df.get("Confidence")

    def bucket_clf(pred, conf):
        if pred in NEGATIVE_LABELS:
            return "high" if (conf is not None and conf >= RISK_HIGH_CONFIDENCE) else "medium"
        if pred in POSITIVE_LABELS:
            return "low" if (conf is not None and conf >= RISK_HIGH_CONFIDENCE) else "medium"
        return None

    confs = confidence.tolist() if confidence is not None else [None] * len(lowered)
    out = []
    for pred, conf in zip(lowered.tolist(), confs):
        c = float(conf) if conf is not None and pd.notna(conf) else None
        out.append(bucket_clf(pred, c))
    return pd.Series(out, index=prediction_df.index)


def confidence_tier(confidence) -> str | None:
    if confidence is None or (isinstance(confidence, float) and np.isnan(confidence)):
        return None
    if confidence >= CONFIDENCE_HIGH:
        return "high"
    if confidence >= CONFIDENCE_MEDIUM:
        return "medium"
    return "low"
