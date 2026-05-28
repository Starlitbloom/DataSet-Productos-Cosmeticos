"""src.data_preprocessing

Transformadores personalizados de *preprocesamiento* para datos tipo pandas.

Este módulo está pensado para centralizar únicamente las clases "core" utilizadas
en el pipeline final, evitando duplicar lógica dentro de notebooks.

Todas las clases siguen el contrato de scikit-learn:
- Heredan de :class:`sklearn.base.BaseEstimator` y :class:`sklearn.base.TransformerMixin`.
- Implementan :meth:`fit` y :meth:`transform`.
- Mantienen compatibilidad con el API `set_output` (sklearn >= 1.2).

Notas:
- Se asume que las entradas/salidas son :class:`pandas.DataFrame`.
- La prevención de *data leakage* se implementa aprendiendo parámetros en `fit`
  y aplicándolos en `transform`.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple


import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class DropColumnsTransformer(BaseEstimator, TransformerMixin):
    """Drop de columnas específicas.

    Se utiliza para eliminar variables que no deben entrar al entrenamiento
    (por ejemplo, IDs o columnas generadas por joins que no aportan señal).

    Parameters
    ----------
    columns_to_drop:
        Lista de nombres de columnas a eliminar. Las que no existan en el DataFrame
        se ignoran.
    """

    def __init__(self, columns_to_drop: Optional[Sequence[str]] = None):
        self.columns_to_drop = list(columns_to_drop) if columns_to_drop else []

    def fit(self, X: pd.DataFrame, y: Any = None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_copy = X.copy()
        cols = [c for c in self.columns_to_drop if c in X_copy.columns]
        return X_copy.drop(columns=cols)

    def set_output(self, transform: str | None = None):
        return self


class UnknownToNaNTransformer(BaseEstimator, TransformerMixin):
    """Convierte valores tipo string como 'unknown' a NaN.

    Suele ser útil cuando el dataset trae placeholders en texto.

    Convierten a :data:`numpy.nan` los valores que coinciden (ignorando espacios)
    con: ``unknown``.
    """

    def fit(self, X: pd.DataFrame, y: Any = None):
        def fit(self, X: pd.DataFrame, y=None):
            """
            Aprende los parámetros necesarios desde los datos de entrenamiento.

            En este caso no se ajustan parámetros, pero se mantiene
            compatibilidad con el pipeline de sklearn.
            """
        return self


    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_copy = X.copy()
        # reemplaza 'unknown' (con variaciones de espacios) por NaN real
        return X_copy.replace(r"^\s*unknown\s*$", np.nan, regex=True)

    def set_output(self, transform: str | None = None):
        return self


class SmartImputerTransformer(BaseEstimator, TransformerMixin):
    """Imputación *leakage-free* (mediana para numéricas, moda para categóricas).

    Este imputador aprende valores en `fit` únicamente usando el dataset de entrenamiento,
    y luego aplica esos valores en `transform`.

    Parameters
    ----------
    low_threshold:
        Umbral (porcentaje) para dividir columnas en "más simples" vs "más complejas".

        - En esta versión, se mantiene la misma estrategia general (mediana/mode).
        - El parámetro queda para extender reglas adicionales si quieres.

    Attributes
    ----------
    impute_values_:
        Diccionario {columna: valor_imputación} aprendido en `fit`.
    """

    def __init__(self, low_threshold: float = 0.10):
        self.low_threshold = low_threshold
        self.impute_values_: Dict[str, Any] = {}

    def fit(self, X: pd.DataFrame, y: Any = None):
        self.impute_values_ = {}

        # Mediana para variables numéricas
        # Moda para variables categóricas
        for col in X.columns:
            if pd.api.types.is_numeric_dtype(X[col]):
                self.impute_values_[col] = X[col].median()
            else:
                mode_val = X[col].mode()
                self.impute_values_[col] = mode_val[0] if not mode_val.empty else "Unknown"

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Elimina columnas especificadas del dataset.

        Parameters
        ----------
        X : DataFrame de entrada

        Returns
        -------
        DataFrame sin las columnas eliminadas
        """
        return X.fillna(self.impute_values_)

    def set_output(self, transform: str | None = None):
        return self


class DropHighMissingTransformer(BaseEstimator, TransformerMixin):
    """Elimina columnas con demasiados valores faltantes.

    Parameters
    ----------
    threshold:
        Umbral de proporción de nulos. Si ``missing_pct[col] > threshold`` la columna
        se elimina.
    """

    def __init__(self, threshold: float = 0.8):
        self.threshold = threshold
        self.cols_to_drop_: List[str] = []

    def fit(self, X: pd.DataFrame, y: Any = None):
        missing_pct = X.isnull().mean()
        self.cols_to_drop_ = missing_pct[missing_pct > self.threshold].index.tolist()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.drop(columns=self.cols_to_drop_, errors="ignore")

    def set_output(self, transform: str | None = None):
        return self


class OutlierCapper(BaseEstimator, TransformerMixin):
    """Caps de outliers con método IQR.

    Aplica recorte: ``[Q1 - factor*IQR, Q3 + factor*IQR]`` sobre columnas numéricas.

    Parameters
    ----------
    apply_capping:
        Activa/desactiva el capping.
    factor:
        Factor multiplicador del IQR.

    Attributes
    ----------
    bounds_:
        Diccionario {columna: (lower, upper)} aprendido en `fit`.
    """

    def __init__(self, apply_capping: bool = True, factor: float = 1.5):
        self.apply_capping = apply_capping
        self.factor = factor
        self.bounds_: Dict[str, Tuple[float, float]] = {}

    def fit(self, X: pd.DataFrame, y: Any = None):
        if not self.apply_capping:
            return self

        self.bounds_ = {}
        for col in X.select_dtypes(include="number").columns:
            # Se calculan límites usando IQR para detectar outliers
            q1 = X[col].quantile(0.25)
            q3 = X[col].quantile(0.75)
            iqr = q3 - q1
            self.bounds_[col] = (q1 - self.factor * iqr, q3 + self.factor * iqr)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_copy = X.copy()
        if not self.apply_capping:
            return X_copy

        for col, (lower, upper) in self.bounds_.items():
            if col in X_copy.columns:
                X_copy[col] = np.clip(X_copy[col], lower, upper)

        return X_copy

    def set_output(self, transform: str | None = None):
        return self
    
