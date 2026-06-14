"""src.hyperparameter_tuning

Optimización de hiperparámetros para modelos de clasificación usando
GridSearchCV y RandomizedSearchCV de scikit-learn.

Estrategia
----------
- ``optimize_gradient_boosting``: búsqueda exhaustiva (GridSearchCV) sobre
  un grid adaptativo según el tamaño del dataset.
- ``randomized_search_gradient_boosting``: búsqueda aleatoria
  (RandomizedSearchCV) sobre una distribución de parámetros más amplia;
  útil como exploración rápida antes del grid fino o cuando el espacio
  de búsqueda es demasiado grande para un grid exhaustivo.
- ``optimize_random_forest``: GridSearchCV para RandomForestClassifier.
- ``randomized_search_random_forest``: RandomizedSearchCV para
  RandomForestClassifier.

Notas de implementación
-----------------------
- Submuestreo controlado cuando n_samples > ``MAX_TUNE_SAMPLES`` para
  evitar tiempos de cómputo excesivos con GradientBoosting (no paralelo
  por árbol en sklearn).
- Todos los estimadores reciben ``random_state`` para reproducibilidad.
- Las funciones devuelven un dict uniforme con ``best_model``,
  ``best_params``, ``best_score`` y ``cv_results`` para facilitar la
  comparación posterior.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

MAX_TUNE_SAMPLES = 120_000  # Submuestreo máximo para GradientBoosting


# ---------------------------------------------------------------------------
# Helpers internos
# ---------------------------------------------------------------------------

def _subsample(X_train, y_train, max_samples: int, random_state: int):
    """Devuelve una sub-muestra estratificada si n > max_samples."""
    n_samples = int(getattr(X_train, "shape", [len(y_train)])[0])
    if n_samples <= max_samples:
        return X_train, y_train

    rng = np.random.default_rng(random_state)
    idx = rng.choice(n_samples, size=max_samples, replace=False)
    return X_train[idx], y_train[idx]


def _grid_for_gb(n_samples: int) -> Dict[str, list]:
    """Grid adaptativo para GradientBoosting según tamaño del dataset.

    Reduce combinaciones en datasets grandes para evitar tiempos
    de entrenamiento excesivos.

    Parameters
    ----------
    n_samples:
        Número de muestras del conjunto de entrenamiento (antes del
        posible submuestreo).

    Returns
    -------
    Dict[str, list]
        Diccionario de hiperparámetros para GridSearchCV.
    """
    if n_samples > 200_000:
        return {
            "n_estimators": [150, 250],
            "learning_rate": [0.03, 0.05],
            "max_depth": [2, 3],
            "subsample": [0.8, 1.0],
        }
    if n_samples > 50_000:
        return {
            "n_estimators": [150, 250],
            "learning_rate": [0.03, 0.05],
            "max_depth": [2, 3],
        }
    return {
        "n_estimators": [150, 250, 350],
        "learning_rate": [0.02, 0.05, 0.08],
        "max_depth": [2, 3, 4],
        "subsample": [0.8, 1.0],
    }


# ---------------------------------------------------------------------------
# GradientBoosting — GridSearchCV
# ---------------------------------------------------------------------------

def optimize_gradient_boosting(
    X_train,
    y_train,
    *,
    cv: int = 3,
    random_state: int = 42,
) -> Dict[str, Any]:
    """GridSearchCV exhaustivo sobre GradientBoostingClassifier.

    Realiza una búsqueda exhaustiva sobre un grid adaptativo al tamaño
    del dataset. Para datasets con más de ``MAX_TUNE_SAMPLES`` muestras
    aplica un submuestreo aleatorio previo para reducir el tiempo de
    cómputo.

    Parameters
    ----------
    X_train:
        Features de entrenamiento (array-like o ndarray).
    y_train:
        Etiquetas de entrenamiento (array-like o Series).
    cv:
        Número de folds para la validación cruzada. Por defecto 3.
    random_state:
        Semilla para reproducibilidad. Por defecto 42.

    Returns
    -------
    Dict con claves:
        - ``best_model``: GradientBoostingClassifier reentrenado con los
          mejores hiperparámetros.
        - ``best_params``: diccionario con la combinación óptima.
        - ``best_score``: f1_weighted promedio en CV del mejor estimador.
        - ``cv_results``: dict completo de resultados del GridSearchCV.
        - ``search_method``: ``"GridSearchCV"`` (para comparación).
    """
    n_samples = int(getattr(X_train, "shape", [len(y_train)])[0])
    grid = _grid_for_gb(n_samples)

    X_tune, y_tune = _subsample(X_train, y_train, MAX_TUNE_SAMPLES, random_state)

    base = GradientBoostingClassifier(random_state=random_state)

    search = GridSearchCV(
        estimator=base,
        param_grid=grid,
        scoring="f1_weighted",
        cv=cv,
        n_jobs=1,       # GradientBoosting ya no es thread-safe con n_jobs > 1
        verbose=2,
        pre_dispatch=1,
    )

    search.fit(X_tune, y_tune)

    return {
        "best_model": search.best_estimator_,
        "best_params": search.best_params_,
        "best_score": float(search.best_score_),
        "cv_results": search.cv_results_,
        "search_method": "GridSearchCV",
    }


# ---------------------------------------------------------------------------
# GradientBoosting — RandomizedSearchCV
# ---------------------------------------------------------------------------

def randomized_search_gradient_boosting(
    X_train,
    y_train,
    *,
    n_iter: int = 20,
    cv: int = 3,
    random_state: int = 42,
) -> Dict[str, Any]:
    """RandomizedSearchCV sobre GradientBoostingClassifier.

    Realiza una búsqueda aleatoria sobre un espacio de distribuciones más
    amplio que el grid exhaustivo. Es más rápido cuando el espacio de
    hiperparámetros es grande y permite explorar combinaciones que un
    grid fijo no cubriría.

    Cuándo preferir RandomizedSearch sobre GridSearch
    -------------------------------------------------
    - Espacio de búsqueda grande (muchos parámetros o rangos continuos).
    - Presupuesto computacional limitado (controlas el costo con n_iter).
    - Fase exploratoria previa a un GridSearch fino.

    Parameters
    ----------
    X_train:
        Features de entrenamiento.
    y_train:
        Etiquetas de entrenamiento.
    n_iter:
        Número de combinaciones aleatorias a evaluar. Por defecto 20.
    cv:
        Número de folds para validación cruzada. Por defecto 3.
    random_state:
        Semilla para reproducibilidad. Por defecto 42.

    Returns
    -------
    Dict con claves:
        - ``best_model``: GradientBoostingClassifier con los mejores params.
        - ``best_params``: combinación óptima encontrada.
        - ``best_score``: f1_weighted promedio en CV.
        - ``cv_results``: dict completo de resultados.
        - ``search_method``: ``"RandomizedSearchCV"`` (para comparación).
    """
    # Distribución de parámetros más amplia que el grid
    param_distributions: Dict[str, Any] = {
        "n_estimators": [100, 150, 200, 250, 300, 400],
        "learning_rate": [0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15],
        "max_depth": [2, 3, 4, 5],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    }

    X_tune, y_tune = _subsample(X_train, y_train, MAX_TUNE_SAMPLES, random_state)

    base = GradientBoostingClassifier(random_state=random_state)

    search = RandomizedSearchCV(
        estimator=base,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring="f1_weighted",
        cv=cv,
        n_jobs=1,
        verbose=2,
        random_state=random_state,
        pre_dispatch=1,
    )

    search.fit(X_tune, y_tune)

    return {
        "best_model": search.best_estimator_,
        "best_params": search.best_params_,
        "best_score": float(search.best_score_),
        "cv_results": search.cv_results_,
        "search_method": "RandomizedSearchCV",
    }


# ---------------------------------------------------------------------------
# RandomForest — GridSearchCV
# ---------------------------------------------------------------------------

def optimize_random_forest(
    X_train,
    y_train,
    *,
    cv: int = 3,
    random_state: int = 42,
) -> Dict[str, Any]:
    """GridSearchCV sobre RandomForestClassifier.

    Busca exhaustivamente la mejor combinación de hiperparámetros para
    RandomForest. A diferencia de GradientBoosting, RandomForest admite
    n_jobs=-1 durante el fit, lo que reduce significativamente el tiempo
    de entrenamiento en datasets grandes.

    Parameters
    ----------
    X_train:
        Features de entrenamiento.
    y_train:
        Etiquetas de entrenamiento.
    cv:
        Número de folds para validación cruzada. Por defecto 3.
    random_state:
        Semilla para reproducibilidad. Por defecto 42.

    Returns
    -------
    Dict con claves:
        - ``best_model``: RandomForestClassifier con los mejores params.
        - ``best_params``: combinación óptima.
        - ``best_score``: f1_weighted promedio en CV.
        - ``cv_results``: dict completo de resultados.
        - ``search_method``: ``"GridSearchCV"``.
    """
    param_grid: Dict[str, list] = {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
        "class_weight": [None, "balanced"],
    }

    base = RandomForestClassifier(random_state=random_state, n_jobs=-1)

    search = GridSearchCV(
        estimator=base,
        param_grid=param_grid,
        scoring="f1_weighted",
        cv=cv,
        n_jobs=-1,
        verbose=2,
    )

    search.fit(X_train, y_train)

    return {
        "best_model": search.best_estimator_,
        "best_params": search.best_params_,
        "best_score": float(search.best_score_),
        "cv_results": search.cv_results_,
        "search_method": "GridSearchCV",
    }


# ---------------------------------------------------------------------------
# RandomForest — RandomizedSearchCV
# ---------------------------------------------------------------------------

def randomized_search_random_forest(
    X_train,
    y_train,
    *,
    n_iter: int = 20,
    cv: int = 3,
    random_state: int = 42,
) -> Dict[str, Any]:
    """RandomizedSearchCV sobre RandomForestClassifier.

    Explora un espacio más amplio de hiperparámetros para RandomForest
    de forma aleatoria y eficiente. Útil como complemento o alternativa
    al GridSearchCV cuando se quiere cubrir más combinaciones con el
    mismo presupuesto computacional.

    Parameters
    ----------
    X_train:
        Features de entrenamiento.
    y_train:
        Etiquetas de entrenamiento.
    n_iter:
        Número de combinaciones a evaluar. Por defecto 20.
    cv:
        Número de folds. Por defecto 3.
    random_state:
        Semilla. Por defecto 42.

    Returns
    -------
    Dict con claves:
        - ``best_model``: RandomForestClassifier con los mejores params.
        - ``best_params``: combinación óptima encontrada.
        - ``best_score``: f1_weighted promedio en CV.
        - ``cv_results``: dict completo de resultados.
        - ``search_method``: ``"RandomizedSearchCV"``.
    """
    param_distributions: Dict[str, Any] = {
        "n_estimators": [50, 100, 150, 200, 300, 400, 500],
        "max_depth": [None, 5, 10, 15, 20, 30],
        "min_samples_split": [2, 5, 10, 15],
        "min_samples_leaf": [1, 2, 4, 8],
        "max_features": ["sqrt", "log2", None],
        "class_weight": [None, "balanced"],
    }

    base = RandomForestClassifier(random_state=random_state, n_jobs=-1)

    search = RandomizedSearchCV(
        estimator=base,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring="f1_weighted",
        cv=cv,
        n_jobs=-1,
        verbose=2,
        random_state=random_state,
    )

    search.fit(X_train, y_train)

    return {
        "best_model": search.best_estimator_,
        "best_params": search.best_params_,
        "best_score": float(search.best_score_),
        "cv_results": search.cv_results_,
        "search_method": "RandomizedSearchCV",
    }


# ---------------------------------------------------------------------------
# Utilidad: comparar resultados de búsqueda
# ---------------------------------------------------------------------------

def compare_search_results(*results: Dict[str, Any]) -> None:
    """Imprime una tabla comparativa de los resultados de búsqueda.

    Permite contrastar rápidamente los mejores scores obtenidos por
    GridSearchCV vs RandomizedSearchCV (u otras combinaciones).

    Parameters
    ----------
    *results:
        Dicts devueltos por las funciones de optimización de este módulo.
        Cada uno debe tener las claves ``search_method``, ``best_score``
        y ``best_params``.

    Example
    -------
    >>> grid_result = optimize_gradient_boosting(X_train, y_train)
    >>> rand_result = randomized_search_gradient_boosting(X_train, y_train)
    >>> compare_search_results(grid_result, rand_result)
    """
    header = f"{'Método':<22} {'Best F1 (CV)':>14}   Mejores parámetros"
    print(header)
    print("-" * 80)
    for r in results:
        method = r.get("search_method", "?")
        score = r.get("best_score", float("nan"))
        params = r.get("best_params", {})
        print(f"{method:<22} {score:>14.4f}   {params}")
