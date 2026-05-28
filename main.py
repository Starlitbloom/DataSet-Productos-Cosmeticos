"""
main.py

Orquestador end-to-end para entrenamiento, optimización y evaluación de un
modelo Gradient Boosting sobre el dataset Sephora.

Requisitos del pipeline
------------------------
1) Carga y split usando `load_and_split_data()`
2) Entrenamiento del modelo base con `train_gradient_boosting()`
3) Optimización con `optimize_gradient_boosting()`
4) Evaluación + guardado de métricas y gráficas en `results/`
5) Serialización del mejor modelo en:
   `models/trained_models/mejor_gradient_boosting.pkl`

Convenciones
------------
- Docstrings estilo NumPy
- Type hints
- Logs claros con separadores visuales
- Ejecución protegida con `if __name__ == "__main__":`
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import joblib

from src.utils import load_and_split_data
from src.model_training import train_gradient_boosting
from src.hyperparameter_tuning import optimize_gradient_boosting
from src.model_evaluation import (
    evaluate_and_save_metrics,
    save_classification_report,
    plot_and_save_confusion_matrix,
    plot_and_save_roc_curve,
)


def _setup_logger() -> logging.Logger:
    """Configura un logger estándar para ejecución de consola.

    Returns
    -------
    logging.Logger
        Logger configurado para el módulo principal.
    """
    logger = logging.getLogger("cosmetics_main")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def _phase(title: str, logger: logging.Logger) -> None:
    """Imprime un separador visual para delimitar fases.

    Parameters
    ----------
    title : str
        Texto del título de la fase.
    logger : logging.Logger
        Logger para emitir mensajes.
    """
    sep = "=" * 50
    logger.info(sep)
    logger.info(title)
    logger.info(sep)


def _get_project_root() -> Path:
    """Obtiene la raíz del proyecto (directorio conteniendo este main.py)."""
    return Path(__file__).resolve().parent


def run_pipeline() -> None:
    """Ejecuta el pipeline end-to-end solicitado."""
    logger = _setup_logger()

    root = _get_project_root()
    results_dir = root / "results"
    models_dir = root / "models" / "trained_models"
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    model_name = "mejor_gradient_boosting"
    model_output_path = models_dir / f"{model_name}.pkl"

    # =========================
    # Fase 1: Carga y Split
    # =========================
    _phase("Fase 1: Carga y Split de Datos", logger)

    csv_path = root / "data" / "processed" / "sephora_limpio.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"No se encontró el dataset: {csv_path}")

    X_train, X_test, y_train, y_test = load_and_split_data(
        filepath=str(csv_path),
        target_col="is_recommended",
        test_size=0.2,
        random_state=42,
    )

    logger.info("X_train: %s | X_test: %s", getattr(X_train, "shape", None), getattr(X_test, "shape", None))

    # =========================
    # Fase 2: Modelo Base
    # =========================
    _phase("Fase 2: Entrenamiento del Modelo Base", logger)

    base_model = train_gradient_boosting(X_train, y_train)
    logger.info("Modelo base entrenado: %s", type(base_model).__name__)

    # ==========================================
    # Fase 3: Optimización de Hiperparámetros
    # ==========================================
    _phase("Fase 3: Optimización de Hiperparámetros", logger)

    tuning_payload: Dict[str, Any] = optimize_gradient_boosting(X_train, y_train)
    best_model = tuning_payload["best_model"]
    best_params = tuning_payload.get("best_params")
    best_score = tuning_payload.get("best_score")

    logger.info("Mejores params: %s", best_params)
    logger.info("Mejor score (cv, f1): %s", best_score)

    # =========================
    # Fase 4: Evaluación y Guardado
    # =========================
    _phase("Fase 4: Evaluación y Guardado de Reportes/Gráficas", logger)

    # Métricas (texto + consola)
    metrics: Dict[str, float] = evaluate_and_save_metrics(
        model=best_model,
        X_test=X_test,
        y_test=y_test,
        model_name=model_name,
    )

    logger.info("Métricas principales: %s", metrics)

    # Classification report
    _ = save_classification_report(
        model=best_model,
        X_test=X_test,
        y_test=y_test,
        model_name=model_name,
    )
    logger.info("Classification report guardado en results/ (según implementación).")

    # Matriz de confusión
    _ = plot_and_save_confusion_matrix(
        model=best_model,
        X_test=X_test,
        y_test=y_test,
        model_name=model_name,
    )
    logger.info("Matriz de confusión guardada en results/plots/ (PNG).")

    # Curva ROC
    _ = plot_and_save_roc_curve(
        model=best_model,
        X_test=X_test,
        y_test=y_test,
        model_name=model_name,
    )
    logger.info("Curva ROC guardada en results/plots/ (PNG).")

    # =========================
    # Fase 5: Serialización
    # =========================
    _phase("Fase 5: Serialización del Mejor Modelo", logger)

    joblib.dump(best_model, str(model_output_path), compress=3)
    logger.info("Modelo guardado en: %s", model_output_path)

    logger.info("%s\nPIPELINE COMPLETADO EXITOSAMENTE\n%s", "=" * 50, "=" * 50)


if __name__ == "__main__":
    run_pipeline()
