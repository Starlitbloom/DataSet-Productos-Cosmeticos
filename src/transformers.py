"""
Custom Scikit-Learn transformers for cosmetics data.
Includes handling for data leakage prevention, imputation, and string cleaning.
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class DropColumnsTransformer(BaseEstimator, TransformerMixin):
    """Drops specified columns to prevent data leakage."""
    def __init__(self, columns_to_drop=None):
        self.columns_to_drop = columns_to_drop if columns_to_drop else []

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        # Elimina las columnas no deseadas (URLs, imágenes, IDs)
        cols = [c for c in self.columns_to_drop if c in X_copy.columns]
        return X_copy.drop(columns=cols)

    def set_output(self, transform=None):
        """
        Protocol requirement for scikit-learn >= 1.2 to support 
        the 'set_output' API and ensure pandas compatibility.
        """
        return self

class UnknownToNaNTransformer(BaseEstimator, TransformerMixin):
    """Converts 'unknown' string values into true numpy NaN values."""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        # Reemplaza la palabra 'unknown' por un nulo matemático real
        return X_copy.replace(r'^\s*unknown\s*$', np.nan, regex=True)

    def set_output(self, transform=None):
        """
        Protocol requirement for scikit-learn >= 1.2 to support 
        the 'set_output' API and ensure pandas compatibility.
        """
        return self


class StringCleanerTransformer(BaseEstimator, TransformerMixin):
    """Strips whitespace and normalizes text values in string columns."""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        # Limpia los espacios en blanco sobrantes de las variables de texto
        for col in X_copy.select_dtypes(include=['object', 'string']).columns:
            X_copy[col] = X_copy[col].astype(str).str.strip()
        return X_copy

    def set_output(self, transform=None):
        """
        Protocol requirement for scikit-learn >= 1.2 to support 
        the 'set_output' API and ensure pandas compatibility.
        """
        return self

class DropHighMissingTransformer(BaseEstimator, TransformerMixin):
    """Drops columns exceeding a defined missing values threshold."""
    def __init__(self, threshold=0.8):
        self.threshold = threshold
        self.cols_to_drop_ = []

    def fit(self, X, y=None):
        missing_pct = X.isnull().mean()
        self.cols_to_drop_ = missing_pct[missing_pct > self.threshold].index.tolist()
        return self

    def transform(self, X):
        return X.drop(columns=self.cols_to_drop_, errors='ignore')

    def set_output(self, transform=None):
        """
        Protocol requirement for scikit-learn >= 1.2 to support 
        the 'set_output' API and ensure pandas compatibility.
        """
        return self


class SmartImputerTransformer(BaseEstimator, TransformerMixin):
    """Leakage-free imputer using median for numeric and mode for categorical features."""
    def __init__(self, low_threshold=0.10):
        self.low_threshold = low_threshold
        self.impute_values_ = {}

    def fit(self, X, y=None):
        # PREVENCIÓN DATA LEAKAGE: Aprendemos solo de los datos de entrenamiento
        for col in X.columns:
            if pd.api.types.is_numeric_dtype(X[col]):
                self.impute_values_[col] = X[col].median()
            else:
                mode_val = X[col].mode()
                self.impute_values_[col] = mode_val[0] if not mode_val.empty else 'Unknown'
        return self

    def transform(self, X):
        # Aplicamos los valores seguros sobre los datos nuevos
        return X.fillna(self.impute_values_)

    def set_output(self, transform=None):
        """
        Protocol requirement for scikit-learn >= 1.2 to support 
        the 'set_output' API and ensure pandas compatibility.
        """
        return self

class OutlierCapper(BaseEstimator, TransformerMixin):
    """Caps numeric outliers using the Interquartile Range (IQR) method."""
    def __init__(self, apply_capping=True, factor=1.5):
        self.apply_capping = apply_capping
        self.factor = factor
        self.bounds_ = {}

    def fit(self, X, y=None):
        if not self.apply_capping: return self
        for col in X.select_dtypes(include='number').columns:
            q1 = X[col].quantile(0.25)
            q3 = X[col].quantile(0.75)
            iqr = q3 - q1
            self.bounds_[col] = (q1 - self.factor * iqr, q3 + self.factor * iqr)
        return self

    def transform(self, X):
        X_copy = X.copy()
        if not self.apply_capping: return X_copy
        # Recorte de valores extremos para estabilizar la varianza
        for col, (lower, upper) in self.bounds_.items():
            if col in X_copy.columns:
                X_copy[col] = np.clip(X_copy[col], lower, upper)
        return X_copy

    def set_output(self, transform=None):
        """
        Protocol requirement for scikit-learn >= 1.2 to support 
        the 'set_output' API and ensure pandas compatibility.
        """
        return self