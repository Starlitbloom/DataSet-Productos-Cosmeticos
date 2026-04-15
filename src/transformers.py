"""
Custom Scikit-Learn Transformers.
Contains classes for structural cleaning, outlier capping, and smart imputation.
"""

import pandas as pd
import numpy as np
import re
from sklearn.base import BaseEstimator, TransformerMixin


class DropColumnsTransformer(BaseEstimator, TransformerMixin):
    """Drops specified columns from the DataFrame to prevent data leakage."""
    def __init__(self, columns_to_drop):
        self.columns_to_drop = columns_to_drop

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        cols = [col for col in self.columns_to_drop if col in X_copy.columns]
        return X_copy.drop(columns=cols)
    
    def set_output(self, transform=None):
        return self


class UnknownToNaNTransformer(BaseEstimator, TransformerMixin):
    """Converts 'unknown' string values into true numpy NaN values."""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        X_copy = X_copy.replace(r'^\s*unknown\s*$', np.nan, regex=True)
        return X_copy
    
    def set_output(self, transform=None):
        return self


class StringCleanerTransformer(BaseEstimator, TransformerMixin):
    """Strips whitespace and normalizes text values in string columns."""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        object_cols = X_copy.select_dtypes(include=['object', 'string']).columns
        for col in object_cols:
            X_copy[col] = (
                X_copy[col]
                .astype('string')
                .str.replace(r'\s+', ' ', regex=True)
                .str.strip()
                .replace('', pd.NA)
            )
        return X_copy
    
    def set_output(self, transform=None):
        return self


class DropHighMissingTransformer(BaseEstimator, TransformerMixin):
    """Drops columns that exceed a specific threshold of missing values."""
    def __init__(self, threshold=0.8):
        self.threshold = threshold
        self.cols_to_drop_ = []

    def fit(self, X, y=None):
        pct_nulos = X.isnull().mean()
        self.cols_to_drop_ = pct_nulos[pct_nulos > self.threshold].index.tolist()
        return self

    def transform(self, X):
        X_copy = X.copy()
        cols = [c for c in self.cols_to_drop_ if c in X_copy.columns]
        return X_copy.drop(columns=cols)
    
    def set_output(self, transform=None):
        return self


class OutlierCapper(BaseEstimator, TransformerMixin):
    """Caps numerical outliers using the IQR method. Can be bypassed using apply_capping=False."""
    def __init__(self, apply_capping=True):
        self.apply_capping = apply_capping
        self.bounds_ = {}

    def fit(self, X, y=None):
        if not self.apply_capping:
            return self

        for col in X.select_dtypes(include=['number']).columns:
            Q1 = X[col].quantile(0.25)
            Q3 = X[col].quantile(0.75)
            IQR = Q3 - Q1
            self.bounds_[col] = (Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)
        return self

    def transform(self, X):
        X_copy = X.copy()
        if not self.apply_capping:
            return X_copy

        for col, (lower, upper) in self.bounds_.items():
            if col in X_copy.columns:
                X_copy[col] = np.clip(X_copy[col], lower, upper)
        return X_copy

    def get_feature_names_out(self, input_features=None):
        return input_features
    
    def set_output(self, transform=None):
        return self


class DropZeroVarianceTransformer(BaseEstimator, TransformerMixin):
    """Drops numerical columns that have zero variance (constant values)."""
    def __init__(self):
        self.cols_to_drop_ = []

    def fit(self, X, y=None):
        num_cols = X.select_dtypes(include=['number']).columns
        zero_variance = [col for col in num_cols if X[col].std() == 0]
        # Evitamos dejar un conjunto numérico vacío en el pipeline.
        if len(num_cols) - len(zero_variance) == 0:
            self.cols_to_drop_ = []
        else:
            self.cols_to_drop_ = zero_variance
        return self

    def transform(self, X):
        X_copy = X.copy()
        cols = [c for c in self.cols_to_drop_ if c in X_copy.columns]
        return X_copy.drop(columns=cols)

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return None
        return np.array([f for f in input_features if f not in self.cols_to_drop_])

    def set_output(self, transform=None):
        return self

class SmartImputerTransformer(BaseEstimator, TransformerMixin):
    """
    Decides the imputation strategy based on the percentage of missing values:
    - < 10%: Simple imputation (Median/Mode)
    - 10% - 80%: Complex imputation (Placeholder for future KNN/Iterative)
    - > 80%: Ignored (Handled by previous transformers)
    """
    def __init__(self, low_threshold=0.10):
        self.low_threshold = low_threshold
        self.cols_simples_ = []
        self.cols_complejas_ = []

    def fit(self, X, y=None):
        porcentaje_nulos = X.isnull().mean()

        for col in X.columns:
            pct = porcentaje_nulos[col]
            if 0 < pct <= self.low_threshold:
                self.cols_simples_.append(col)
            elif pct > self.low_threshold:
                self.cols_complejas_.append(col)

        print(f"SmartImputer - Simples (<10%): {self.cols_simples_}")
        print(f"SmartImputer - Complejas (>10%): {self.cols_complejas_} (PENDIENTE)")
        return self

    def transform(self, X):
        X_copy = X.copy()

        for col in self.cols_simples_:
            if pd.api.types.is_numeric_dtype(X_copy[col]):
                X_copy[col] = X_copy[col].fillna(X_copy[col].median())
            else:
                X_copy[col] = X_copy[col].fillna(X_copy[col].mode()[0])

        if self.cols_complejas_:
            for col in self.cols_complejas_:
                if pd.api.types.is_numeric_dtype(X_copy[col]):
                    X_copy[col] = X_copy[col].fillna(X_copy[col].median())
                else:
                    X_copy[col] = X_copy[col].fillna(X_copy[col].mode()[0])

        return X_copy

    def get_feature_names_out(self, input_features=None):
        return input_features
    
    def set_output(self, transform=None):
        return self
