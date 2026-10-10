"""The preprocessing pipeline.

Identical to the original project's create_preprocessor(): median imputation
plus standard scaling for numeric columns, constant imputation plus dense
one-hot encoding for categorical columns, joined by a ColumnTransformer.
"""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def create_preprocessor(numeric_cols, categorical_cols):
    """Create sklearn preprocessing pipeline for numeric and categorical features.
    
    Numeric features:
      - SimpleImputer(strategy='median') for outlier-robust imputation
      - StandardScaler() for feature scaling

    Categorical features:
      - SimpleImputer(strategy='most_frequent') for mode imputation
      - OneHotEncoder(handle_unknown='ignore', sparse_output=False) for encoding
    """
    transformers = []

    if numeric_cols:
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        transformers.append(('num', numeric_transformer, list(numeric_cols)))

    if categorical_cols:
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        transformers.append(('cat', categorical_transformer, list(categorical_cols)))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder='drop'
    )

    return preprocessor
