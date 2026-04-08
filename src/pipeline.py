# pipeline.py
# Data processing pipeline

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from src.transformers import (
    DropColumnsTransformer,
    UnknownToNaNTransformer,
    StringCleanerTransformer,
    DropHighMissingTransformer,
    SmartImputerTransformer,
    OutlierCapper,
    DropZeroVarianceTransformer
)


def build_preprocessing_pipeline(columns_to_drop=None):
    """Builds a preprocessing pipeline for the cosmetics dataset.

    The pipeline drops metadata columns, replaces unknown values with NaN,
    removes high-missing columns, imputes missing values, caps outliers,
    scales numeric features, and one-hot encodes categorical features.
    """
    default_drop = ['ID', 'img', 'name', 'description', 'shade_img', 'dupes', 'price_site']
    if columns_to_drop is None:
        columns_to_drop = default_drop
    else:
        columns_to_drop = default_drop + list(columns_to_drop)

    num_pipe = Pipeline([
        ('capper', OutlierCapper(apply_capping=True)),
        ('zero_variance', DropZeroVarianceTransformer()),
        ('scaler', StandardScaler())
    ])

    cat_pipe = Pipeline([
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_pipe, make_column_selector(dtype_include='number')),
            ('cat', cat_pipe, make_column_selector(dtype_exclude='number'))
        ],
        remainder='drop'
    )

    full_pipeline = Pipeline([
        ('drop_leaks', DropColumnsTransformer(columns_to_drop=columns_to_drop)),
        ('clean_strings', StringCleanerTransformer()),
        ('clean_unknowns', UnknownToNaNTransformer()),
        ('drop_high_nan', DropHighMissingTransformer(threshold=0.8)),
        ('smart_imputer', SmartImputerTransformer(low_threshold=0.10)),
        ('preprocessing', preprocessor)
    ])

    return full_pipeline


def run_pipeline(df, columns_to_drop=None):
    pipeline = build_preprocessing_pipeline(columns_to_drop=columns_to_drop)
    if isinstance(df, pd.DataFrame):
        return pipeline.fit_transform(df)
    raise ValueError('Input debe ser un pandas DataFrame')
