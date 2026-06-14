"""
main.py

Orquestador end-to-end para entrenamiento, optimización y evaluación de un
modelo Gradient Boosting sobre el dataset Sephora.

Pipeline actualizado:
- Usa preprocesamiento modular (sin CSV intermedio)
- Usa Gradient Boosting + tuning
- Evalúa y guarda resultados automáticamente
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

import joblib

# =========================
# PATHS CENTRALIZADOS
# =========================
from src.paths import PROJECT_ROOT, RESULTS_DIR, TRAINED_MODELS_DIR


# =========================
# IMPORTS PIPELINE
# =========================
from src.data_preprocessing import run_preprocessing_pipeline
from src.hyperparameter_tuning import optimize_gradient_boosting
from src.model_evaluation import (
    evaluate_and_save_metrics,
    save_classification_report,
    plot_and_save_confusion_matrix,
    plot_and_save_roc_curve,
)

# =========================================================
# LOGGING
# =========================================================
def _setup_logger() -> logging.Logger:
    logger = logging.getLogger("cosmetics_main")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def _phase(title: str, logger: logging.Logger) -> None:
    sep = "=" * 50
    logger.info(sep)
    logger.info(title)
    logger.info(sep)


# =========================================================
# PIPELINE
# =========================================================
def run_pipeline() -> None:
    logger = _setup_logger()

    root = PROJECT_ROOT

    # =========================
    # MODELOS OUTPUT
    # =========================
    models_dir = TRAINED_MODELS_DIR
    models_dir.mkdir(parents=True, exist_ok=True)

    # =========================
    # SALIDAS (ESTABILIDAD)
    # =========================
    # Creamos explícitamente estructura de results desde el inicio para evitar
    # confusiones si el proceso se detiene antes de evaluación.
    (RESULTS_DIR / "metrics").mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "plots").mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "reports").mkdir(parents=True, exist_ok=True)



    model_name = "mejor_gradient_boosting"
    model_output_path = models_dir / f"{model_name}.pkl"


    # =====================================================
    # FASE 1: PREPROCESAMIENTO
    # =====================================================
    _phase("Fase 1: Preprocesamiento de datos", logger)

    artifacts = run_preprocessing_pipeline(
        raw_dir=str(root / "data" / "raw"),
        processed_dir=str(root / "data" / "processed"),
        target_col="is_recommended",
    )

    X_train = joblib.load(artifacts["X_train_processed"])
    X_test = joblib.load(artifacts["X_test_processed"])
    y_train = joblib.load(artifacts["y_train"])
    y_test = joblib.load(artifacts["y_test"])

    logger.info("Datos listos:")
    logger.info("X_train: %s | X_test: %s", X_train.shape, X_test.shape)

    # =====================================================
    # FASE 2: OPTIMIZACIÓN
    # =====================================================
    _phase("Fase 2: Optimización Gradient Boosting", logger)

    tuning_payload: Dict[str, Any] = optimize_gradient_boosting(X_train, y_train)

    best_model = tuning_payload["best_model"]

    logger.info("Mejores parámetros: %s", tuning_payload["best_params"])
    logger.info("Mejor score CV: %s", tuning_payload["best_score"])

    # =====================================================
    # FASE 3: EVALUACIÓN
    # =====================================================
    _phase("Fase 3: Evaluación del modelo", logger)

    logger.info("RESULTS DIR: %s", RESULTS_DIR)

    metrics = evaluate_and_save_metrics(
        model=best_model,
        X_test=X_test,
        y_test=y_test,
        model_name=model_name
    )

    logger.info("Métricas finales: %s", metrics)

    save_classification_report(best_model, X_test, y_test, model_name)
    plot_and_save_confusion_matrix(best_model, X_test, y_test, model_name)
    plot_and_save_roc_curve(best_model, X_test, y_test, model_name)

    # =====================================================
    # FASE 4: GUARDADO MODELO
    # =====================================================
    _phase("Fase 4: Guardado del modelo", logger)

    joblib.dump(best_model, model_output_path, compress=3)
    logger.info("Modelo guardado en: %s", model_output_path)

    logger.info("\nPIPELINE COMPLETADO EXITOSAMENTE\n")


# =========================================================
if __name__ == "__main__":
    run_pipeline()