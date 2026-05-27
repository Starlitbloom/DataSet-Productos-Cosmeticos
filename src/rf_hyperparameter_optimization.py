"""Optimización de Random Forest con GridSearchCV y RandomizedSearchCV.

Este módulo está pensado para integrarse con tu pipeline de preprocesamiento
existente (src/pipeline.py). Permite justificar en tu informe:

- Por qué se eligieron rangos de hiperparámetros (sesgo/varianza, estabilidad,
  trade-off computacional).
- Cómo impacta el rendimiento final (comparación por CV).

Notas importantes:
- RandomForest es robusto pero sensible a: número de árboles, profundidad,
  y tamaño mínimo de hoja.
- Para clasificación, usamos RandomForestClassifier con métricas tipo accuracy/F1/ROC-AUC.
- Para regresión, usarías RandomForestRegressor con r2/MAE/RMSE.

El código asume que
- X es pandas.DataFrame
- y es pandas.Series (target)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline

from sklearn.model_selection import KFold, StratifiedKFold

from sklearn.metrics import make_scorer, accuracy_score, f1_score, roc_auc_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from src.pipeline import build_preprocessing_pipeline


@dataclass
class SearchResult:
    search_name: str
    best_params: Dict[str, Any]
    best_score: float
    cv_results_: pd.DataFrame
    best_estimator_: BaseEstimator


def infer_problem_type(y: pd.Series) -> str:
    """Inferir 'classification' vs 'regression' con heurística simple."""
    if pd.api.types.is_numeric_dtype(y):
        y_non_na = y.dropna()
        unique_vals = y_non_na.unique()
        # Si parece discreto (pocos valores enteros) -> clasificación.
        if len(unique_vals) <= 20 and np.all(np.equal(np.mod(unique_vals.astype(float), 1), 0)):
            return "classification"
        return "regression"
    return "classification"


def get_cv(problem_type: str, n_splits: int, random_state: int):
    if problem_type == "classification":
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return KFold(n_splits=n_splits, shuffle=True, random_state=random_state)


def build_rf_model(problem_type: str, random_state: int):
    if problem_type == "classification":
        return RandomForestClassifier(
            random_state=random_state,
            n_jobs=-1,
            class_weight="balanced",  # útil si hay desbalance
        )
    return RandomForestRegressor(
        random_state=random_state,
        n_jobs=-1,
    )


def default_scoring(problem_type: str):
    """Scoring principal para elegir el mejor modelo.

    En informe es clave explicar que:
    - accuracy/f1 evalúan desempeño general.
    - roc_auc es más informativa si hay desbalance.
    - Para regresión usamos r2/MAE/RMSE.

    Usamos una sola métrica de "refit" para escoger best_estimator_.
    Luego puedes evaluar métricas adicionales con cross_validate si lo deseas.
    """
    if problem_type == "classification":
        # Si multiclase, roc_auc_score requiere multi_class. Usamos OVR.
        roc_auc = make_scorer(roc_auc_score, needs_proba=True, multi_class="ovr")
        # En problemas binarios roc_auc_score funciona igual.
        return roc_auc

    # regresión
    # r2 suele ser estándar para selección; MAE/RMSE pueden ir a evaluación adicional.
    return make_scorer(r2_score)


def rf_param_distributions(problem_type: str) -> Dict[str, Any]:
    """Rangos para GridSearchCV/RandomizedSearchCV.

    Justificación técnica (resumen para tu informe):
    - n_estimators:
      Más árboles reducen varianza; se acota para controlar costo.
    - max_depth:
      Controla complejidad. Profundidades bajas (p.ej. 5-20) evitan sobreajuste,
      profundidades altas se acercan a bosque “casi sin podar”.
    - min_samples_split / min_samples_leaf:
      Controlan el tamaño mínimo de divisiones/hojas.
      Valores más grandes regularizan, reducen overfitting y estabilizan.
    - max_features:
      Para Random Forest afecta diversidad de árboles. Valores típicos:
      - sqrt para clasificación (regla empírica)
      - log2 o porcentajes para ajustar diversidad.
    - bootstrap:
      Permite muestreo con reemplazo; en RF clásico suele funcionar bien.

    RandomizedSearchCV explora mejor espacios amplios; GridSearchCV refina en zonas
    más prometedoras (menos combinaciones).
    """

    # Distribuciones comunes
    n_estimators = [200, 400, 700, 1000]

    if problem_type == "classification":
        return {
            "model__n_estimators": n_estimators,
            "model__max_depth": [None, 5, 10, 20, 30, 40],
            "model__min_samples_split": [2, 5, 10, 20],
            "model__min_samples_leaf": [1, 2, 4, 8, 16],
            "model__max_features": ["sqrt", "log2", 0.3, 0.5, 0.7, None],
            "model__bootstrap": [True, False],
        }

    return {
        "model__n_estimators": n_estimators,
        "model__max_depth": [None, 5, 10, 20, 30, 40],
        "model__min_samples_split": [2, 5, 10, 20],
        "model__min_samples_leaf": [1, 2, 4, 8, 16],
        "model__max_features": ["sqrt", "log2", 0.3, 0.5, 0.7, None],
        "model__bootstrap": [True, False],
    }


def build_search_pipelines(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    columns_to_drop: Optional[List[str]] = None,
    preprocessing_pipeline=None,
    random_state: int = 42,
):
    """Crea el Pipeline preprocess + modelo RF."""
    problem_type = infer_problem_type(y)

    preprocess = preprocessing_pipeline or build_preprocessing_pipeline(columns_to_drop=columns_to_drop)

    rf_model = build_rf_model(problem_type=problem_type, random_state=random_state)

    pipe = Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("model", rf_model),
        ]
    )

    return problem_type, pipe


def run_grid_search(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    n_splits: int = 5,
    random_state: int = 42,
    columns_to_drop: Optional[List[str]] = None,
    preprocessing_pipeline=None,
    scoring=None,
    refit: bool = True,
    n_jobs: int = -1,
    verbose: int = 1,
    param_grid: Optional[Dict[str, Any]] = None,
) -> SearchResult:
    problem_type, pipe = build_search_pipelines(
        X, y,
        columns_to_drop=columns_to_drop,
        preprocessing_pipeline=preprocessing_pipeline,
        random_state=random_state,
    )

    cv = get_cv(problem_type, n_splits=n_splits, random_state=random_state)

    if scoring is None:
        scoring = default_scoring(problem_type)

    # Grid refina un subconjunto (menos combinaciones) respecto a Randomized.
    # Si no se pasa param_grid, usamos uno compacto.
    if param_grid is None:
        param_grid = {
            # tamaño de bosque (varianza) - no se pone demasiado alto en Grid para costo
            "model__n_estimators": [300, 600, 900],
            # control de complejidad
            "model__max_depth": [None, 10, 20, 30],
            # regularización vía hojas
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 4],
            # diversidad entre árboles
            "model__max_features": ["sqrt", "log2", 0.5],
            "model__bootstrap": [True],
        }

    search = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring=scoring,
        refit=refit,
        cv=cv,
        n_jobs=n_jobs,
        verbose=verbose,
        return_train_score=False,
    )

    search.fit(X, y)

    cv_results_df = pd.DataFrame(search.cv_results_)

    return SearchResult(
        search_name="GridSearchCV (Random Forest)",
        best_params=search.best_params_,
        best_score=float(search.best_score_),
        cv_results_=cv_results_df,
        best_estimator_=search.best_estimator_,
    )


def run_randomized_search(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    n_splits: int = 5,
    random_state: int = 42,
    columns_to_drop: Optional[List[str]] = None,
    preprocessing_pipeline=None,
    scoring=None,
    refit: bool = True,
    n_jobs: int = -1,
    verbose: int = 1,
    n_iter: int = 40,
    param_distributions: Optional[Dict[str, Any]] = None,
) -> SearchResult:
    problem_type, pipe = build_search_pipelines(
        X, y,
        columns_to_drop=columns_to_drop,
        preprocessing_pipeline=preprocessing_pipeline,
        random_state=random_state,
    )

    cv = get_cv(problem_type, n_splits=n_splits, random_state=random_state)

    if scoring is None:
        scoring = default_scoring(problem_type)

    if param_distributions is None:
        param_distributions = rf_param_distributions(problem_type)

    # RandomizedSearchCV explora muchas combinaciones sin el explosivo crecimiento de combinaciones.
    # n_iter controla el presupuesto computacional.
    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring=scoring,
        refit=refit,
        cv=cv,
        n_jobs=n_jobs,
        verbose=verbose,
        random_state=random_state,
        return_train_score=False,
    )

    search.fit(X, y)

    cv_results_df = pd.DataFrame(search.cv_results_)

    return SearchResult(
        search_name="RandomizedSearchCV (Random Forest)",
        best_params=search.best_params_,
        best_score=float(search.best_score_),
        cv_results_=cv_results_df,
        best_estimator_=search.best_estimator_,
    )


def summarize_search_results(results: List[SearchResult]) -> pd.DataFrame:
    rows = []
    for r in results:
        rows.append({
            "search": r.search_name,
            "best_score": r.best_score,
            **{f"param_{k}": v for k, v in r.best_params.items()},
        })
    return pd.DataFrame(rows)

