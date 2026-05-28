"""src.hyperparameter_tuning

Módulo de ajuste (tuning) de hiperparámetros para modelos de clasificación.

Este módulo proporciona utilidades enfocadas en el entrenamiento y selección de
modelos usando ``GridSearchCV`` con métricas de clasificación.

Notas
-----
- La función principal expone un flujo simple y reproducible para
  ``GradientBoostingClassifier``.
- Se asume que ``X_train`` y ``y_train`` ya vienen preprocesados.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV


def optimize_gradient_boosting(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    param_grid: Optional[Dict[str, Any]] = None,
    cv: int = 3,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Optimiza hiperparámetros de GradientBoostingClassifier usando GridSearchCV.

    Returns
    -------
    dict
        Diccionario con:
        - best_model: mejor estimador entrenado
        - best_params: mejores hiperparámetros encontrados
        - best_score: mejor F1 en validación cruzada
        - cv_results: resultados completos del GridSearch
    """

    # Malla por defecto
    if param_grid is None:
        param_grid = {
            "n_estimators": [100, 200],
            "learning_rate": [0.05, 0.1, 0.2],
            "max_depth": [2, 3],
        }

    # Modelo base
    estimator = GradientBoostingClassifier(random_state=random_state)

    # GridSearchCV
    grid = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring="f1",
        cv=cv,
        n_jobs=-1,
        refit=True,
    )

    grid.fit(X_train, y_train)

    # Retorno estructurado
    return {
        "best_model": grid.best_estimator_,
        "best_params": grid.best_params_,
        "best_score": grid.best_score_,
        "cv_results": grid.cv_results_,
    }