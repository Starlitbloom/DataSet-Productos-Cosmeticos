"""src.model_training

Definición y entrenamiento de modelos de clasificación para el dataset
Sephora Cosmetics Reviews (variable objetivo: ``is_recommended``).

Modelos implementados
---------------------
- **LogisticRegression**: modelo lineal, sirve como baseline interpretable.
  Apropiado para clasificación binaria cuando se desea un modelo rápido y
  transparente. Sus coeficientes permiten analizar directamente el peso de
  cada variable.
- **RandomForestClassifier**: ensamble de árboles por bagging. Robusto ante
  outliers, no requiere escala de features (aunque el pipeline ya la aplica),
  y proporciona importancia de variables nativa.
- **GradientBoostingClassifier**: ensamble secuencial boosting. Generalmente
  más preciso que RF en clasificación tabular, a costa de mayor tiempo de
  entrenamiento y mayor sensibilidad a hiperparámetros.

Justificación de la selección
------------------------------
Los tres modelos cubren el espectro lineal → no-lineal → boosting, lo que
permite comparar trade-offs entre interpretabilidad, velocidad y rendimiento
predictivo. LogisticRegression actúa como baseline; RF y GB compiten en
rendimiento y se optimizan en ``hyperparameter_tuning.py``.

Notas de implementación
-----------------------
- Todos los modelos reciben ``random_state`` para reproducibilidad.
- El guardado con ``joblib`` usa ``compress=3`` para reducir tamaño en disco.
- Las funciones son stateless: reciben datos, devuelven el modelo entrenado.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import joblib
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.paths import TRAINED_MODELS_DIR


# ---------------------------------------------------------------------------
# Helper interno
# ---------------------------------------------------------------------------

def _save_model(model: Any, output_path: Optional[str]) -> None:
    """Serializa el modelo en ``output_path`` si se especifica.

    Parameters
    ----------
    model:
        Estimador scikit-learn ya entrenado.
    output_path:
        Ruta de destino. Si es ``None`` no se hace nada.
    """
    if output_path is None:
        return
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out, compress=3)


# ---------------------------------------------------------------------------
# Logistic Regression  (baseline lineal)
# ---------------------------------------------------------------------------

def train_logistic_regression(
    X_train,
    y_train,
    *,
    model_params: Optional[Dict[str, Any]] = None,
    model_output_path: Optional[str] = None,
) -> LogisticRegression:
    """Entrena un LogisticRegression y opcionalmente lo persiste con joblib.

    Uso como baseline
    -----------------
    La regresión logística es el punto de partida natural para clasificación
    binaria. Si los modelos de ensamble no superan significativamente a este
    baseline, puede ser señal de que el problema es esencialmente lineal o
    que el preprocesamiento necesita mejoras.

    Parameters
    ----------
    X_train:
        Features de entrenamiento (array-like o ndarray preprocesado).
    y_train:
        Etiquetas de entrenamiento.
    model_params:
        Diccionario opcional con hiperparámetros. Claves reconocidas:
        ``C`` (regularización inversa, default 1.0),
        ``max_iter`` (default 1000),
        ``class_weight`` (default None),
        ``solver`` (default ``"lbfgs"``),
        ``random_state`` (default 42).
    model_output_path:
        Ruta donde guardar el modelo serializado. Si es ``None``, no guarda.

    Returns
    -------
    LogisticRegression
        Modelo entrenado.
    """
    model_params = model_params or {}

    model = LogisticRegression(
        C=model_params.get("C", 1.0),
        max_iter=model_params.get("max_iter", 1000),
        class_weight=model_params.get("class_weight", None),
        solver=model_params.get("solver", "lbfgs"),
        random_state=model_params.get("random_state", 42),
        n_jobs=model_params.get("n_jobs", -1),
    )

    model.fit(X_train, y_train)
    _save_model(model, model_output_path)
    return model


# ---------------------------------------------------------------------------
# Random Forest
# ---------------------------------------------------------------------------

def train_random_forest(
    X_train,
    y_train,
    *,
    model_params: Optional[Dict[str, Any]] = None,
    model_output_path: Optional[str] = None,
) -> RandomForestClassifier:
    """Entrena un RandomForestClassifier y opcionalmente lo persiste con joblib.

    RandomForest es un ensamble por bagging: entrena múltiples árboles de
    decisión en subconjuntos aleatorios de datos y features, y agrega sus
    predicciones por votación. Es robusto ante overfitting, no requiere
    escala de features y expone ``feature_importances_`` para interpretabilidad.

    Parameters
    ----------
    X_train:
        Features de entrenamiento.
    y_train:
        Etiquetas de entrenamiento.
    model_params:
        Diccionario opcional. Claves reconocidas:
        ``n_estimators`` (default 300),
        ``max_depth`` (default None),
        ``class_weight`` (default None),
        ``n_jobs`` (default -1),
        ``random_state`` (default 42).
    model_output_path:
        Ruta de guardado. Si es ``None``, no guarda.

    Returns
    -------
    RandomForestClassifier
        Modelo entrenado.
    """
    model_params = model_params or {}

    model = RandomForestClassifier(
        n_estimators=model_params.get("n_estimators", 300),
        max_depth=model_params.get("max_depth", None),
        random_state=model_params.get("random_state", 42),
        n_jobs=model_params.get("n_jobs", -1),
        class_weight=model_params.get("class_weight", None),
    )

    model.fit(X_train, y_train)
    _save_model(model, model_output_path)
    return model


# ---------------------------------------------------------------------------
# Gradient Boosting
# ---------------------------------------------------------------------------

def train_gradient_boosting(
    X_train,
    y_train,
    *,
    model_params: Optional[Dict[str, Any]] = None,
    model_output_path: Optional[str] = None,
) -> GradientBoostingClassifier:
    """Entrena un GradientBoostingClassifier y opcionalmente lo persiste con joblib.

    GradientBoosting construye árboles secuencialmente, donde cada árbol
    corrige los errores del anterior minimizando una función de pérdida
    (log-loss en clasificación). Suele superar a RandomForest en rendimiento
    tabular, pero es más sensible a hiperparámetros y más lento de entrenar.

    Parameters
    ----------
    X_train:
        Features de entrenamiento.
    y_train:
        Etiquetas de entrenamiento.
    model_params:
        Diccionario opcional. Claves reconocidas:
        ``n_estimators`` (default 300),
        ``learning_rate`` (default 0.05),
        ``max_depth`` (default 3),
        ``subsample`` (default 1.0),
        ``random_state`` (default 42).
    model_output_path:
        Ruta de guardado. Si es ``None``, no guarda.

    Returns
    -------
    GradientBoostingClassifier
        Modelo entrenado.
    """
    model_params = model_params or {}

    model = GradientBoostingClassifier(
        n_estimators=model_params.get("n_estimators", 300),
        learning_rate=model_params.get("learning_rate", 0.05),
        max_depth=model_params.get("max_depth", 3),
        random_state=model_params.get("random_state", 42),
        subsample=model_params.get("subsample", 1.0),
    )

    model.fit(X_train, y_train)
    _save_model(model, model_output_path)
    return model
