"""Numerical equivalence between the original Streamlit project and this port.

The original utils/*.py modules import streamlit. A minimal stub stands in for
it (cache_data becomes a pass-through, spinner becomes a no-op context
manager), which lets the real original code run unchanged in this process. Both
implementations are then handed the same dataframes and the results are
compared exactly.

Run:  python -m pytest tests/test_equivalence.py -q
ORIGINAL_DIR must point at the unpacked original project.
"""

import contextlib
import os
import sys
import types

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ORIGINAL_DIR = os.environ.get("ORIGINAL_DIR", "/home/claude/src/student-performance-predictor-main")
DATA_DIR = os.environ.get("DATA_DIR", "/home/claude/testdata")

pytestmark = pytest.mark.skipif(
    not os.path.isdir(ORIGINAL_DIR),
    reason="original Streamlit project not available",
)


def _install_streamlit_stub():
    """Stand in for streamlit so the original modules import and run."""
    if "streamlit" in sys.modules:
        return

    stub = types.ModuleType("streamlit")

    def cache_data(*decorator_args, **decorator_kwargs):
        # Supports both @st.cache_data and @st.cache_data(show_spinner=False)
        if decorator_args and callable(decorator_args[0]):
            return decorator_args[0]

        def wrap(func):
            return func

        return wrap

    @contextlib.contextmanager
    def spinner(*_args, **_kwargs):
        yield

    stub.cache_data = cache_data
    stub.spinner = spinner
    stub.session_state = {}
    for name in ("error", "warning", "success", "info", "caption", "markdown", "write"):
        setattr(stub, name, lambda *args, **kwargs: None)

    sys.modules["streamlit"] = stub


_install_streamlit_stub()

# The ported package is imported FIRST. The original project has a top-level
# app.py, which would otherwise shadow this project's `app` package as soon as
# ORIGINAL_DIR joins sys.path.
from app.ml import helpers as ported_helpers  # noqa: E402
from app.ml import training as ported_trainer  # noqa: E402
from app.ml.prediction import predict_batch as ported_predict_batch  # noqa: E402

sys.path.append(ORIGINAL_DIR)

if os.path.isdir(ORIGINAL_DIR):
    from utils import helpers as original_helpers  # noqa: E402
    from utils import model_trainer as original_trainer  # noqa: E402
else:
    original_helpers = None  # type: ignore
    original_trainer = None  # type: ignore


@pytest.fixture(scope="module")
def classification_df():
    return pd.read_csv(os.path.join(DATA_DIR, "students_classification.csv"))


@pytest.fixture(scope="module")
def regression_df():
    return pd.read_csv(os.path.join(DATA_DIR, "students_regression.csv"))


# --- column introspection ---------------------------------------------

def test_column_helpers_agree(classification_df):
    assert ported_helpers.get_numeric_columns(classification_df) == original_helpers.get_numeric_columns(classification_df)
    assert ported_helpers.get_text_category_columns(classification_df) == original_helpers.get_text_category_columns(classification_df)
    assert ported_helpers.get_categorical_like_columns(classification_df) == original_helpers.get_categorical_like_columns(classification_df)


def test_target_detection_agrees(classification_df, regression_df):
    for frame in (classification_df, regression_df):
        assert ported_helpers.detect_target_candidates(frame) == original_helpers.detect_target_candidates(frame)

    # including the edge cases
    odd = pd.DataFrame({"a": [1], "Final Grade": [2], "student_name": ["x"]})
    assert ported_helpers.detect_target_candidates(odd) == original_helpers.detect_target_candidates(odd)


def test_schema_validation_agrees(classification_df):
    for target in ("pass_fail", "not_a_column"):
        assert ported_helpers.validate_schema(classification_df, target) == original_helpers.validate_schema(classification_df, target)


# --- problem type, recommendation, validation -------------------------

def test_problem_type_detection_agrees(classification_df, regression_df):
    assert ported_trainer.detect_problem_type(classification_df, "pass_fail") == original_trainer.detect_problem_type(classification_df, "pass_fail")
    assert ported_trainer.detect_problem_type(regression_df, "exam_score") == original_trainer.detect_problem_type(regression_df, "exam_score")
    # low-cardinality numeric column must be treated as classes by both
    assert ported_trainer.detect_problem_type(classification_df, "attendance") == original_trainer.detect_problem_type(classification_df, "attendance")


def test_model_recommendation_agrees(classification_df, regression_df):
    assert ported_trainer.recommend_model(classification_df, "pass_fail", "classification") == original_trainer.recommend_model(classification_df, "pass_fail", "classification")
    assert ported_trainer.recommend_model(regression_df, "exam_score", "regression") == original_trainer.recommend_model(regression_df, "exam_score", "regression")
    small = regression_df.head(50)
    assert ported_trainer.recommend_model(small, "exam_score", "regression") == original_trainer.recommend_model(small, "exam_score", "regression")
    assert ported_trainer.recommend_model(small, "exam_score", "classification") == original_trainer.recommend_model(small, "exam_score", "classification")


@pytest.mark.parametrize(
    "target,problem_type,model_choice",
    [
        ("pass_fail", "classification", "Random Forest"),
        ("pass_fail", "regression", "Linear Regression"),   # blocked: text target
        ("attendance", "classification", "Decision Tree"),
        ("previous_score", "classification", "Random Forest"),  # continuous-looking warning
    ],
)
def test_preflight_validation_agrees(classification_df, target, problem_type, model_choice):
    mine = ported_trainer.validate_before_training(classification_df, target, problem_type, model_choice)
    theirs = original_trainer.validate_before_training(classification_df, target, problem_type, model_choice)
    assert mine == theirs


def test_preflight_edge_cases_agree():
    tiny = pd.DataFrame({"grade": ["A"], "x": [1]})
    assert ported_trainer.validate_before_training(tiny, "grade", "classification", "Random Forest") == \
        original_trainer.validate_before_training(tiny, "grade", "classification", "Random Forest")

    one_class = pd.DataFrame({"grade": ["A"] * 6, "x": [1, 2, 3, 4, 5, 6]})
    assert ported_trainer.validate_before_training(one_class, "grade", "classification", "Random Forest") == \
        original_trainer.validate_before_training(one_class, "grade", "classification", "Random Forest")

    constant = pd.DataFrame({"score": [5.0] * 10, "x": range(10)})
    assert ported_trainer.validate_before_training(constant, "score", "regression", "Linear Regression") == \
        original_trainer.validate_before_training(constant, "score", "regression", "Linear Regression")


# --- preprocessing and training ---------------------------------------

def _compare_training(frame, target, model_choice, problem_type):
    mine = ported_trainer.train_model(frame, target, model_choice, problem_type)
    theirs = original_trainer.train_model(frame, target, model_choice, problem_type)

    assert mine["feature_columns"] == theirs["feature_columns"]
    assert mine["n_train"] == theirs["n_train"]
    assert mine["n_test"] == theirs["n_test"]
    assert list(mine["y_test"]) == list(theirs["y_test"])

    # identical predictions on the identical split
    assert np.array_equal(np.asarray(mine["y_pred"]), np.asarray(theirs["y_pred"]))

    # identical metrics
    assert set(mine["metrics"]) == set(theirs["metrics"])
    for key, value in mine["metrics"].items():
        assert value == pytest.approx(theirs["metrics"][key], rel=1e-12, abs=1e-12), key

    return mine, theirs


@pytest.mark.parametrize("model_choice", ["Logistic Regression", "Decision Tree", "Random Forest"])
def test_classification_training_is_identical(classification_df, model_choice):
    mine, theirs = _compare_training(classification_df, "pass_fail", model_choice, "classification")

    assert np.array_equal(mine["confusion_matrix"], theirs["confusion_matrix"])
    assert mine["confusion_matrix_labels"] == theirs["confusion_matrix_labels"]

    assert (mine["roc_curve"] is None) == (theirs["roc_curve"] is None)
    if mine["roc_curve"]:
        assert mine["roc_curve"]["auc"] == pytest.approx(theirs["roc_curve"]["auc"], rel=1e-12)
        assert np.allclose(mine["roc_curve"]["fpr"], theirs["roc_curve"]["fpr"])
        assert np.allclose(mine["roc_curve"]["tpr"], theirs["roc_curve"]["tpr"])
        assert mine["roc_curve"]["positive_class"] == theirs["roc_curve"]["positive_class"]

    assert (mine["feature_importance"] is None) == (theirs["feature_importance"] is None)
    if mine["feature_importance"] is not None:
        assert np.allclose(mine["feature_importance"], theirs["feature_importance"])
        assert mine["feature_names"] == theirs["feature_names"]

    assert "residuals" not in mine and "residuals" not in theirs


@pytest.mark.parametrize(
    "model_choice", ["Linear Regression", "Decision Tree Regressor", "Random Forest Regressor"]
)
def test_regression_training_is_identical(regression_df, model_choice):
    mine, theirs = _compare_training(regression_df, "exam_score", model_choice, "regression")

    assert np.allclose(np.asarray(mine["residuals"]), np.asarray(theirs["residuals"]))
    assert "confusion_matrix" not in mine and "confusion_matrix" not in theirs

    assert (mine["feature_importance"] is None) == (theirs["feature_importance"] is None)
    if mine["feature_importance"] is not None:
        assert np.allclose(mine["feature_importance"], theirs["feature_importance"])
        assert mine["feature_names"] == theirs["feature_names"]


def test_preprocessor_produces_identical_matrix(classification_df):
    from app.ml.preprocessing import create_preprocessor as mine
    from utils.model_trainer import create_preprocessor as theirs

    features = classification_df.drop(columns=["pass_fail"])
    numeric = ported_helpers.get_numeric_columns(features)
    categorical = ported_helpers.get_categorical_like_columns(features)

    a = mine(numeric, categorical).fit_transform(features)
    b = theirs(numeric, categorical).fit_transform(features)
    assert np.allclose(a, b)


# --- batch prediction --------------------------------------------------

def test_batch_validation_agrees(regression_df):
    features = [c for c in regression_df.columns if c != "exam_score"]
    batch = regression_df.head(40).drop(columns=["exam_score"])

    assert ported_helpers.validate_batch_prediction_data(batch, features, regression_df) == \
        original_helpers.validate_batch_prediction_data(batch, features, regression_df)

    missing = batch.drop(columns=["attendance"])
    assert ported_helpers.validate_batch_prediction_data(missing, features, regression_df) == \
        original_helpers.validate_batch_prediction_data(missing, features, regression_df)

    extra = batch.copy()
    extra["note"] = "x"
    assert ported_helpers.validate_batch_prediction_data(extra, features, regression_df) == \
        original_helpers.validate_batch_prediction_data(extra, features, regression_df)

    as_text = batch.copy()
    as_text["attendance"] = as_text["attendance"].astype(str)
    assert ported_helpers.validate_batch_prediction_data(as_text, features, regression_df) == \
        original_helpers.validate_batch_prediction_data(as_text, features, regression_df)


def test_batch_output_structure_is_identical(regression_df):
    """The ported batch path must produce the same columns and values as the
    original page's predict_batch()."""
    result = original_trainer.train_model(regression_df, "exam_score", "Random Forest Regressor", "regression")
    pipeline = result["pipeline"]
    features = result["feature_columns"]
    batch = regression_df.head(60).drop(columns=["exam_score"])

    # original page logic, inlined exactly as it was written there
    theirs = batch.copy()
    inputs = original_helpers.coerce_batch_dtypes(batch, features, regression_df)[features]
    theirs["Prediction"] = pipeline.predict(inputs)
    if hasattr(pipeline, "predict_proba"):
        probabilities = pipeline.predict_proba(inputs)
        theirs["Confidence"] = np.max(probabilities, axis=1)
        theirs["Predicted Class"] = pipeline.classes_[np.argmax(probabilities, axis=1)]

    mine = ported_predict_batch(pipeline, features, batch, regression_df)

    assert list(mine.columns) == list(theirs.columns)
    assert np.allclose(mine["Prediction"], theirs["Prediction"])


def test_classification_batch_confidence_is_identical(classification_df):
    result = original_trainer.train_model(classification_df, "pass_fail", "Random Forest", "classification")
    pipeline = result["pipeline"]
    features = result["feature_columns"]
    batch = classification_df.head(80).drop(columns=["pass_fail"])

    theirs = batch.copy()
    inputs = original_helpers.coerce_batch_dtypes(batch, features, classification_df)[features]
    theirs["Prediction"] = pipeline.predict(inputs)
    probabilities = pipeline.predict_proba(inputs)
    theirs["Confidence"] = np.max(probabilities, axis=1)
    theirs["Predicted Class"] = pipeline.classes_[np.argmax(probabilities, axis=1)]

    mine = ported_predict_batch(pipeline, features, batch, classification_df)

    assert list(mine.columns) == list(theirs.columns)
    assert list(mine["Prediction"]) == list(theirs["Prediction"])
    assert np.allclose(mine["Confidence"], theirs["Confidence"])
    assert list(mine["Predicted Class"]) == list(theirs["Predicted Class"])
