"""Modelos supervisados con preprocesamiento existente + validación cruzada.

Este módulo inicializa al menos 3 modelos de scikit-learn (clasificación/regresión)
conectados a tu pipeline de preprocesamiento definido en `src/pipeline.py`.

- Implementa CV robusta (KFold/StratifiedKFold según el tipo de problema).
- Calcula múltiples métricas.
- Mantiene la estructura modular para que puedas reutilizarla en notebooks o main.

Suposiciones:
- Las entradas (X) son un pandas.DataFrame con las columnas del dataset crudo.
- Los objetivos (y) vienen como Series/array-like.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    make_scorer,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from src.pipeline import build_preprocessing_pipeline


TargetType = Union[str, int, float]


@dataclass
class CVResult:
    model_name: str
    metrics_summary: Dict[str, Dict[str, float]]
    raw: Dict[str, Any]


def _infer_problem_type(y: pd.Series) -> str:
    """Inferir problema: 'classification' o 'regression'.

    Regla práctica:
    - Si y es categórico (dtype 'object'/'category') o tiene pocas clases y es entero => clasificación.
    - Si y es numérico continuo => regresión.
    """
    if pd.api.types.is_numeric_dtype(y):
        # si son pocos valores enteros -> clasificación
        y_non_na = y.dropna()
        unique_vals = y_non_na.unique()
        if len(unique_vals) <= 20 and np.all(np.equal(np.mod(unique_vals.astype(float), 1), 0)):
            return "classification"
        return "regression"

    return "classification"


def build_models_for_problem(problem_type: str, random_state: int = 42) -> Dict[str, BaseEstimator]:
    """Devuelve diccionario de modelos (mínimo 3) para classification o regression."""

    if problem_type == "classification":
        # 3 modelos diferentes
        return {
            "logistic_regression": LogisticRegression(
                max_iter=2000,
                n_jobs=None,
                class_weight="balanced",
                random_state=random_state,
            ),
            "random_forest": RandomForestClassifier(
                n_estimators=400,
                random_state=random_state,
                n_jobs=-1,
                class_weight="balanced",
            ),
            "gradient_boosting": GradientBoostingClassifier(random_state=random_state),
        }

    # regression
    return {
        "ridge_regression": Ridge(random_state=random_state),
        "random_forest_regressor": __import__("sklearn.ensemble").ensemble.RandomForestRegressor(
            n_estimators=400,
            random_state=random_state,
            n_jobs=-1,
        ),
        "gradient_boosting_regressor": __import__("sklearn.ensemble").ensemble.GradientBoostingRegressor(
            random_state=random_state
        ),
    }


def get_cv_strategy(problem_type: str, n_splits: int = 5, random_state: int = 42):
    if problem_type == "classification":
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return KFold(n_splits=n_splits, shuffle=True, random_state=random_state)


def build_scorers(problem_type: str):
    """Scorers múltiples para comparar modelos."""

    if problem_type == "classification":
        # AUC requiere probabilidades; cross_validate llamará predict_proba si existe.
        # Para ROC-AUC, sklearn usa scorer que internamente llama predict_proba.
        return {
            "accuracy": make_scorer(accuracy_score),
            "f1_weighted": make_scorer(f1_score, average="weighted"),
            # Maneja binaria/multiclase: roc_auc_score requiere especificar multi_class.
            "roc_auc_ovr": make_scorer(roc_auc_score, needs_proba=True, multi_class="ovr"),
        }

    return {
        "r2": make_scorer(r2_score),
        "rmse": make_scorer(lambda yt, yp: mean_squared_error(yt, yp, squared=False), greater_is_better=False),
        "mae": make_scorer(mean_absolute_error, greater_is_better=False),
    }


def evaluate_models_cv(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    target_column_name: Optional[str] = None,
    n_splits: int = 5,
    random_state: int = 42,
    scoring_override: Optional[Dict[str, Any]] = None,
    columns_to_drop: Optional[List[str]] = None,
    preprocessing_pipeline=None,
    verbose: int = 0,
) -> List[CVResult]:
    """Evalúa múltiples modelos con CV usando tu pipeline de preprocesamiento."""

    problem_type = _infer_problem_type(y)
    scorers = scoring_override or build_scorers(problem_type)

    cv = get_cv_strategy(problem_type, n_splits=n_splits, random_state=random_state)

    preprocessing = preprocessing_pipeline or build_preprocessing_pipeline(columns_to_drop=columns_to_drop)

    models = build_models_for_problem(problem_type, random_state=random_state)

    results: List[CVResult] = []

    for model_name, model in models.items():
        # pipeline end-to-end: preprocesa + model
        clf_or_reg = Pipeline(
            steps=[
                ("preprocess", preprocessing),
                ("model", model),
            ]
        )

        cv_out = cross_validate(
            clf_or_reg,
            X,
            y,
            cv=cv,
            scoring=scorers,
            return_train_score=False,
            n_jobs=None,
            verbose=verbose,
        )

        # Resumen: media/std de cada métrica
        metrics_summary: Dict[str, Dict[str, float]] = {}
        for metric_key, values in cv_out.items():
            if not metric_key.startswith("test_"):
                continue
            short_key = metric_key.replace("test_", "")
            arr = np.asarray(values, dtype=float)
            metrics_summary[short_key] = {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr, ddof=0)),
            }

        results.append(
            CVResult(
                model_name=model_name,
                metrics_summary=metrics_summary,
                raw=cv_out,
            )
        )

    return results


def pretty_print_results(results: List[CVResult]) -> pd.DataFrame:
    """Devuelve un DataFrame con ranking/overview de métricas."""
    rows = []
    for r in results:
        row = {"model": r.model_name}
        for metric, stats in r.metrics_summary.items():
            row[f"{metric}_mean"] = stats["mean"]
            row[f"{metric}_std"] = stats["std"]
        rows.append(row)
    return pd.DataFrame(rows).set_index("model")


# Ejemplo de uso (para notebook / main):
#
# df = pd.read_csv("data/raw/...csv")
# y = df["rating"]  # o la columna objetivo que decidas
# X = df.drop(columns=["rating"])
#
# results = evaluate_models_cv(X, y, n_splits=5)
# summary_df = pretty_print_results(results)
# print(summary_df)

