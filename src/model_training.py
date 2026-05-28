"""src.model_training

Entrenamiento de modelos base para clasificación binaria.

Este módulo define utilidades pequeñas y reutilizables para instanciar y ajustar
modelos entrenables evaluados en el EDA. Se mantiene un estilo compatible con
el resto del proyecto:

- Docstrings estilo NumPy.
- Tipado estricto.
- Reproducibilidad mediante ``random_state`` por defecto.

Notas
-----
Las funciones asumen que ``X_train`` es un :class:`pandas.DataFrame` y que ``y_train``
corresponde a la etiqueta binaria ``is_recommended`` (por ejemplo, 0/1).

Las funciones NO incluyen el preprocesamiento: se espera que el usuario
conecte el modelo dentro de un :class:`sklearn.pipeline.Pipeline` con el
pipeline de preprocesamiento definido en :mod:`src.pipeline`.
"""

from __future__ import annotations

from typing import Any, Dict

import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def train_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    **kwargs: Any,
) -> LogisticRegression:
    """Instancia y entrena un modelo de Logistic Regression.

    Parameters
    ----------
    X_train : pd.DataFrame
        Matriz de características de entrenamiento.
    y_train : pd.Series
        Etiquetas de entrenamiento (clasificación binaria).
    **kwargs : Any
        Hiperparámetros adicionales para :class:`sklearn.linear_model.LogisticRegression`.

        - ``random_state``: si no se especifica, se usará ``42``.

    Returns
    -------
    LogisticRegression
        Modelo entrenado (ajustado mediante ``fit``).
    """

    params: Dict[str, Any] = dict(kwargs)
    params.setdefault("random_state", 42)

    model = LogisticRegression(**params)
    model.fit(X_train, y_train)
    return model


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    **kwargs: Any,
) -> RandomForestClassifier:
    """Instancia y entrena un modelo Random Forest para clasificación.

    Parameters
    ----------
    X_train : pd.DataFrame
        Matriz de características de entrenamiento.
    y_train : pd.Series
        Etiquetas de entrenamiento (clasificación binaria).
    **kwargs : Any
        Hiperparámetros adicionales para :class:`sklearn.ensemble.RandomForestClassifier`.

        - ``random_state``: si no se especifica, se usará ``42``.

    Returns
    -------
    RandomForestClassifier
        Modelo entrenado (ajustado mediante ``fit``).
    """

    params: Dict[str, Any] = dict(kwargs)
    params.setdefault("random_state", 42)

    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    return model


def train_gradient_boosting(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    **kwargs: Any,
) -> GradientBoostingClassifier:
    """Instancia y entrena un modelo Gradient Boosting para clasificación.

    Parameters
    ----------
    X_train : pd.DataFrame
        Matriz de características de entrenamiento.
    y_train : pd.Series
        Etiquetas de entrenamiento (clasificación binaria).
    **kwargs : Any
        Hiperparámetros adicionales para
        :class:`sklearn.ensemble.GradientBoostingClassifier`.

        - ``random_state``: si no se especifica, se usará ``42``.

    Returns
    -------
    GradientBoostingClassifier
        Modelo entrenado (ajustado mediante ``fit``).
    """

    params: Dict[str, Any] = dict(kwargs)
    params.setdefault("random_state", 42)

    model = GradientBoostingClassifier(**params)
    model.fit(X_train, y_train)
    return model

