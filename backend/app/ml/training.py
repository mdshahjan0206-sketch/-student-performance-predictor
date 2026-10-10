"""Problem-type detection, model recommendation, pre-flight validation and
model training.

Ported from the original project's utils/model_trainer.py. Every rule,
threshold, message, split parameter and metric is unchanged. The only
difference is mechanical: the `st.spinner(...)` wrapper around `pipeline.fit`
is replaced by an optional `progress` callback so the background job runner
can report truthful stage transitions (training -> evaluating) instead of a
Streamlit widget.
"""

import logging
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from app.core.config import MODEL_OPTIONS
from app.ml.helpers import (
    get_categorical_like_columns,
    get_numeric_columns,
    validate_schema,
)
from app.ml.preprocessing import create_preprocessor

# Server-side logger for training failures. Real exception details (with
# traceback) go here for developers; end users only ever see the clean,
# templated messages built by validate_before_training() / the
# ModelTrainingError raised from train_model() below.
logger = logging.getLogger("edupredict.training")


class ModelTrainingError(Exception):
    """Raised when training fails for a reason that couldn't be caught by
    pre-flight validation (validate_before_training). Callers should catch
    this specifically and show a clean message instead of this exception's
    raw text, which is logged (not displayed) for debugging."""
    pass


def detect_problem_type(df, target_column):
    """Dataset-agnostic detection of whether target is regression or classification.

    Rules:
    - Text, categorical, or boolean columns -> classification
    - Binary targets (<= 2 distinct non-null values) -> classification
    - Numeric continuous targets:
      * Float columns with fractional/decimal values -> regression
      * Columns with high cardinality (>15 distinct values) -> regression
      * Discrete numeric with low cardinality (<= 15 unique integer values,
        like a 1-5 rating or 0-10 category code) -> classification
    """
    if len(df) == 0:
        return "classification"
    y = df[target_column]
    if pd.api.types.is_bool_dtype(y) or not pd.api.types.is_numeric_dtype(y):
        return 'classification'
    y_valid = y.dropna()
    if len(y_valid) == 0:
        return 'classification'
    nunique = int(y_valid.nunique())
    if nunique <= 2:
        return 'classification'

    # Float targets: check for continuous decimal values or cardinality
    if pd.api.types.is_float_dtype(y_valid):
        sample = y_valid.head(1000)
        has_decimals = bool(((sample % 1) != 0).any())
        if has_decimals or nunique > 5:
            return 'regression'
        return 'classification'

    # Integer targets: score/count with > 10 distinct values is regression
    if nunique > 10:
        return 'regression'

    # Discrete integer set (e.g. 0-5 or 1-10 rating scale)
    min_v = y_valid.min()
    max_v = y_valid.max()
    if min_v >= 0 and max_v <= 10 and nunique <= 10:
        return 'classification'

    return 'regression'


def recommend_model(df, target_column, problem_type):
    """Recommend the best-suited model for this dataset + problem type.

    Heuristic: on small datasets, simpler linear models (Logistic/Linear
    Regression) generalize better and are less prone to overfitting than tree
    ensembles; once there's enough data (>=200 rows) a Random Forest is
    recommended since it typically handles the mixed numeric/categorical
    features and non-linear relationships in student-performance data best.
    """
    n_rows = len(df)
    if problem_type == 'classification':
        return "Logistic Regression" if n_rows < 200 else "Random Forest"
    return "Linear Regression" if n_rows < 200 else "Random Forest Regressor"


def validate_before_training(df, target_column, problem_type, model_choice, is_refined: bool = False):
    """Intelligent pre-flight validation: inspects dataset characteristics
    (classification vs. regression shape, sample count, class count/balance,
    missing values, dimensionality) BEFORE pipeline.fit() is ever called, so
    an unsuitable model/dataset combination never reaches scikit-learn as a
    raw exception.

    Returns a dict:
      {
        "ok": bool,                  # False => BLOCKING, do not train
        "reason": str | None,        # human-readable reason when not ok
        "recommended_models": list,  # suggested MODEL_OPTIONS entries
        "warnings": list[str],       # non-blocking heads-up messages
      }
    """
    warnings = []

    # --- Basic schema check (target exists / isn't 100% null) ---
    schema_ok, schema_msg = validate_schema(df, target_column)
    if not schema_ok:
        return {"ok": False, "reason": schema_msg, "recommended_models": [], "warnings": []}

    n_rows = len(df)
    if n_rows < 2:
        return {
            "ok": False,
            "reason": f"The dataset only has {n_rows} row(s). At least 2 rows are required to train any model.",
            "recommended_models": [],
            "warnings": [],
        }

    y = df[target_column]
    # Same robust check as detect_problem_type — see the comment there.
    target_is_text = pd.api.types.is_bool_dtype(y) or not pd.api.types.is_numeric_dtype(y)
    nunique = y.nunique(dropna=True)

    # --- Target column missing values check ---
    target_missing_count = int(y.isna().sum())
    valid_target_rows = n_rows - target_missing_count
    if target_missing_count > 0:
        if valid_target_rows < 2:
            return {
                "ok": False,
                "reason": (
                    f"Target column '{target_column}' has {target_missing_count:,} missing values, leaving "
                    f"only {valid_target_rows} valid row(s). At least 2 valid rows are required to train."
                ),
                "recommended_models": [],
                "warnings": [],
            }
        warnings.append(
            f"{target_missing_count:,} rows have missing target values and will be excluded from supervised training."
        )

    # --- 100% missing and high missingness checks on features ---
    for c in df.columns:
        if c == target_column:
            continue
        missing_c = int(df[c].isna().sum())
        if missing_c == n_rows:
            warnings.append(
                f"Column '{c}' contains only missing values (100% missing) and cannot provide predictive "
                f"information. It will be excluded from model training."
            )
        elif missing_c > 0:
            rate = missing_c / n_rows
            if rate > 0.40:
                warnings.append(
                    f"Column '{c}' has very high missingness ({rate * 100:.1f}%)."
                )

    # --- Core compatibility check: does the chosen problem type actually fit
    #     this target column's data? ---
    y_valid = y.dropna()
    nunique = y_valid.nunique()

    if problem_type == "regression" and target_is_text:
        return {
            "ok": False,
            "reason": (
                f"'{model_choice}' cannot be used for a classification dataset. "
                f"Target column '{target_column}' contains text/category labels, not "
                f"continuous numbers, so a regression model has nothing numeric to predict."
            ),
            "recommended_models": MODEL_OPTIONS.get("classification", []),
            "warnings": [],
        }

    if problem_type == "regression" and not target_is_text and nunique < 2:
        return {
            "ok": False,
            "reason": f"Target column '{target_column}' has a single repeated value — there is nothing for a regression model to learn from.",
            "recommended_models": [],
            "warnings": [],
        }

    if problem_type == "classification" and nunique < 2:
        return {
            "ok": False,
            "reason": f"Target column '{target_column}' has only {nunique} distinct class. Classification needs at least 2.",
            "recommended_models": [],
            "warnings": [],
        }

    if problem_type == "classification" and not target_is_text and nunique > 20 and (nunique / valid_target_rows) > 0.5:
        # Not blocking — it will still technically fit — but this is almost
        # always a mistake (a continuous column being classified).
        warnings.append(
            f"Target column '{target_column}' looks continuous ({nunique} distinct values across "
            f"{valid_target_rows} rows). Classification will treat every unique value as its own class — "
            f"consider Regression instead for a more meaningful model."
        )

    # --- Non-blocking dataset-quality heads-up ---
    feature_count = df.shape[1] - 1
    if feature_count > valid_target_rows:
        warnings.append(
            f"This dataset is high-dimensional relative to its size ({feature_count} feature columns vs. "
            f"{valid_target_rows} rows). Models may overfit — consider collecting more rows or reducing columns."
        )

    if problem_type == "classification":
        class_counts = y_valid.value_counts()
        if len(class_counts) > 0 and (class_counts.min() / class_counts.max()) < 0.1:
            warnings.append(
                "Class imbalance detected — some classes are much rarer than others. "
                "Accuracy alone may be misleading; check precision/recall/F1 per class."
            )

    if valid_target_rows < 30:
        warnings.append(
            f"Only {valid_target_rows} valid rows available. With so little data, evaluation metrics may be unstable "
            f"— treat results as indicative rather than definitive."
        )

    return {"ok": True, "reason": None, "recommended_models": [], "warnings": warnings}


def _build_model(problem_type, model_choice):
    """Look up and instantiate the sklearn estimator for a given problem type
    + model choice, validated against the canonical MODEL_OPTIONS registry in
    app.core.config so this can never silently diverge from what the Train
    Model page actually offers."""
    valid_choices = MODEL_OPTIONS.get(problem_type, [])
    if model_choice not in valid_choices:
        raise ValueError(
            f"'{model_choice}' is not a supported {problem_type} model. "
            f"Choose one of: {', '.join(valid_choices)}"
        )

    if problem_type == 'classification':
        if model_choice == "Logistic Regression":
            return LogisticRegression(max_iter=1000, random_state=42)
        if model_choice == "Decision Tree":
            return DecisionTreeClassifier(random_state=42)
        if model_choice == "Random Forest":
            return RandomForestClassifier(random_state=42)

    else:  # regression
        if model_choice == "Linear Regression":
            return LinearRegression()
        if model_choice == "Decision Tree Regressor":
            return DecisionTreeRegressor(random_state=42)
        if model_choice == "Random Forest Regressor":
            return RandomForestRegressor(random_state=42)

    raise ValueError(f"Unsupported model: {model_choice}")


def train_model(df, target_column, model_choice, problem_type=None, progress=None):
    """Train model with preprocessing and return results.

    Returns a dict whose keys differ by problem type: classification results
    include 'confusion_matrix' and never 'residuals'; regression results
    include 'residuals' and never 'confusion_matrix'. Callers must branch on
    'problem_type' before reading either key.

    `progress` is an optional callable(stage: str) used by the background job
    runner to publish truthful status transitions. It replaces the original
    st.spinner() and has no effect on the model itself.

    NOTE: the full dataset is used here. Visualisation elsewhere in the app
    may sample rows, but training never does.
    """

    def _stage(name):
        if progress is not None:
            progress(name)

    if problem_type is None:
        problem_type = detect_problem_type(df, target_column)

    # 1. Target column protection: NEVER impute target automatically!
    # Exclude rows where target is missing from supervised training.
    target_isna = df[target_column].isna()
    target_missing_count = int(target_isna.sum())
    if target_missing_count > 0:
        logger.info("Excluding %d rows with missing target '%s'", target_missing_count, target_column)
        df_clean = df.loc[~target_isna].copy()
    else:
        df_clean = df.copy()

    if len(df_clean) < 2:
        raise ModelTrainingError(
            f"Training could not start because target column '{target_column}' contains "
            f"only {len(df_clean)} valid row(s)."
        )

    X_raw = df_clean.drop(columns=[target_column])
    y = df_clean[target_column]

    # 2. Exclude any 100% missing columns (unusable features with no information)
    unusable_cols = [c for c in X_raw.columns if X_raw[c].isna().all()]
    if unusable_cols:
        logger.warning("Excluding 100%% missing feature columns: %r", unusable_cols)
        X = X_raw.drop(columns=unusable_cols)
    else:
        X = X_raw

    if len(X.columns) == 0:
        raise ModelTrainingError("Training could not start because all feature columns contain only missing values.")

    numeric_cols = get_numeric_columns(X)
    categorical_cols = get_categorical_like_columns(X)

    preprocessor = create_preprocessor(numeric_cols, categorical_cols)
    model = _build_model(problem_type, model_choice)

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    # Stratify classification splits so rare classes are represented in both
    # train and test sets where possible.
    stratify = y if (problem_type == 'classification' and y.nunique() > 1 and y.value_counts().min() >= 2) else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify
    )

    start_time = time.time()
    try:
        _stage("training")
        pipeline.fit(X_train, y_train)
        training_time = time.time() - start_time
        _stage("evaluating")
        y_pred = pipeline.predict(X_test)
    except Exception as exc:
        # Anything unexpected that pre-flight validation didn't catch is
        # logged in full server-side for debugging, and surfaced to the user
        # only as a clean, generic message via ModelTrainingError.
        logger.exception(
            "Training failed for model=%r, target=%r, problem_type=%r",
            model_choice, target_column, problem_type,
        )
        raise ModelTrainingError(
            f"The selected model ('{model_choice}') could not be trained on the uploaded dataset."
        ) from exc

    def _feature_importance():
        """Feature importance for tree-based models, mapped back to
        human-readable names (numeric columns + one-hot-expanded categorical
        columns)."""
        if not hasattr(model, 'feature_importances_'):
            return None, None
        importance = model.feature_importances_
        cat_names = []
        if categorical_cols and 'cat' in pipeline.named_steps['preprocessor'].named_transformers_:
            cat_names = list(
                pipeline.named_steps['preprocessor']
                .named_transformers_['cat']
                .named_steps['onehot']
                .get_feature_names_out(categorical_cols)
            )
        names = list(numeric_cols) + cat_names
        return importance, names

    base_result = {
        'pipeline': pipeline,
        'problem_type': problem_type,
        'target_column': target_column,
        'feature_columns': list(X.columns),
        'model_choice': model_choice,
        'training_time': training_time,
        'X_test': X_test,
        'y_test': y_test,
        'y_pred': y_pred,
        'n_train': len(X_train),
        'n_test': len(X_test),
        'target_missing_excluded': target_missing_count,
        'unusable_columns_excluded': unusable_cols,
    }

    if problem_type == 'classification':
        feature_importance, feature_names = _feature_importance()

        # ROC curve: only well-defined for binary classification, and only
        # when the estimator can produce probability scores.
        roc_data = None
        classes = list(getattr(pipeline, 'classes_', []))
        if len(classes) == 2 and hasattr(pipeline, 'predict_proba'):
            positive_class = classes[1]
            probas = pipeline.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, probas, pos_label=positive_class)
            roc_data = {
                'fpr': fpr,
                'tpr': tpr,
                'auc': roc_auc_score(y_test == positive_class, probas),
                'positive_class': positive_class,
            }

        base_result.update({
            'metrics': {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
                'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
                'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0),
            },
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'confusion_matrix_labels': sorted(y.unique().tolist(), key=str),
            'feature_importance': feature_importance,
            'feature_names': feature_names,
            'roc_curve': roc_data,
        })
        return base_result

    # regression
    mse = mean_squared_error(y_test, y_pred)
    feature_importance, feature_names = _feature_importance()
    base_result.update({
        'metrics': {
            'mae': float(np.mean(np.abs(y_test - y_pred))),
            'mse': mse,
            'rmse': float(np.sqrt(mse)),
            'r2': r2_score(y_test, y_pred),
        },
        'residuals': y_test - y_pred,
        'feature_importance': feature_importance,
        'feature_names': feature_names,
    })
    return base_result
