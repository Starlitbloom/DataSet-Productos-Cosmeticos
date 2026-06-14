"""src.model_evaluation

Funciones de evaluación y comparación de modelos de clasificación para el
dataset Sephora Cosmetics Reviews (variable objetivo: ``is_recommended``).

Responsabilidades del módulo
-----------------------------
- Calcular y persistir métricas estándar de clasificación (accuracy, F1,
  precision, recall, ROC-AUC) en formato JSON y TXT.
- Generar y guardar el ``classification_report`` de scikit-learn.
- Producir visualizaciones de matriz de confusión y curva ROC, guardadas
  como imágenes PNG de alta resolución.

Estructura de salidas
---------------------
Todos los artefactos se escriben en las rutas definidas en ``src.paths``:

- ``results/metrics/<model_name>_metrics.json`` — métricas en JSON.
- ``results/metrics/<model_name>_metrics.txt``  — métricas en texto plano.
- ``results/reports/<model_name>_classification_report.txt`` — reporte completo.
- ``results/plots/<model_name>_confusion_matrix.png`` — matriz de confusión.
- ``results/plots/<model_name>_roc_curve.png`` — curva ROC.

Notas de implementación
-----------------------
- ``matplotlib.use("Agg")`` garantiza que los gráficos se generen sin
  necesidad de un display gráfico (compatible con ejecución headless).
- Las funciones de plot cierran explícitamente la figura tras guardarla
  (``plt.close(fig)``) para evitar acumulación de memoria en ejecuciones
  largas.
- El cálculo de ROC-AUC soporta clasificación binaria y multiclase (OvR).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import joblib

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from sklearn.preprocessing import label_binarize

from src.paths import METRICS_DIR, PLOTS_DIR, REPORTS_DIR


# ---------------------------------------------------------------------------
# Helpers internos
# ---------------------------------------------------------------------------

def _ensure_results_dirs() -> None:
    """Crea los directorios de resultados si no existen.

    Garantiza que ``METRICS_DIR``, ``PLOTS_DIR`` y ``REPORTS_DIR`` estén
    disponibles antes de intentar escribir cualquier artefacto. Es seguro
    llamar esta función múltiples veces (``exist_ok=True``).
    """
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def _compute_roc_auc(model: Any, X_test, y_test) -> float | None:
    """Calcula el ROC-AUC del modelo sobre el conjunto de test.

    Soporta clasificación binaria y multiclase. En el caso binario usa
    directamente la probabilidad de la clase positiva (índice 1). En el
    caso multiclase aplica la estrategia One-vs-Rest (OvR) con promedio
    ponderado.

    Parameters
    ----------
    model:
        Estimador scikit-learn ya entrenado. Debe implementar
        ``predict_proba``; si no lo implementa, retorna ``None``.
    X_test:
        Features del conjunto de prueba.
    y_test:
        Etiquetas reales del conjunto de prueba.

    Returns
    -------
    float | None
        Valor de ROC-AUC, o ``None`` si el modelo no soporta probabilidades
        o si ocurre un error durante el cálculo multiclase.
    """
    if not hasattr(model, "predict_proba"):
        return None

    proba = model.predict_proba(X_test)

    # Clasificación binaria: clase positiva en columna 1
    if proba.ndim == 2 and proba.shape[1] == 2:
        return float(roc_auc_score(y_test, proba[:, 1]))

    # Clasificación multiclase: One-vs-Rest ponderado
    try:
        return float(roc_auc_score(y_test, proba, multi_class="ovr", average="weighted"))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Funciones públicas
# ---------------------------------------------------------------------------

def evaluate_and_save_metrics(
    *,
    model: Any,
    X_test,
    y_test,
    model_name: str,
) -> Dict[str, Any]:
    """Calcula métricas de clasificación y las persiste en disco.

    Genera las métricas estándar (accuracy, F1, precision, recall y,
    opcionalmente, ROC-AUC) sobre el conjunto de test, las guarda en
    formato JSON y TXT, y las retorna como diccionario para uso programático.

    Parameters
    ----------
    model:
        Estimador scikit-learn ya entrenado con el conjunto de entrenamiento.
    X_test:
        Features del conjunto de prueba (array-like o ndarray preprocesado).
    y_test:
        Etiquetas reales del conjunto de prueba.
    model_name:
        Identificador del modelo. Se usa como prefijo en los nombres de
        archivo de salida (p.ej. ``"gradient_boosting"``).

    Returns
    -------
    Dict[str, Any]
        Diccionario con las claves: ``model_name``, ``accuracy``, ``f1``,
        ``precision``, ``recall`` y, si aplica, ``roc_auc``.

    Side effects
    ------------
    Escribe dos archivos en ``results/metrics/``:
    - ``<model_name>_metrics.json``
    - ``<model_name>_metrics.txt``
    """
    _ensure_results_dirs()

    y_pred = model.predict(X_test)

    metrics: Dict[str, Any] = {
        "model_name": model_name,
        "accuracy":   float(accuracy_score(y_test, y_pred)),
        "f1":         float(f1_score(y_test, y_pred, average="weighted")),
        "precision":  float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
        "recall":     float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
    }

    roc_auc = _compute_roc_auc(model, X_test, y_test)
    if roc_auc is not None:
        metrics["roc_auc"] = roc_auc

    # Guardado JSON
    metrics_path = METRICS_DIR / f"{model_name}_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # Guardado TXT (formato legible)
    metrics_txt_path = METRICS_DIR / f"{model_name}_metrics.txt"
    with open(metrics_txt_path, "w", encoding="utf-8") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")

    return metrics


def save_classification_report(
    model: Any,
    X_test,
    y_test,
    model_name: str,
) -> str:
    """Genera y guarda el classification_report de scikit-learn en disco.

    El reporte incluye precision, recall, F1-score y support por clase,
    con 4 dígitos de precisión decimal. Es útil para analizar el desempeño
    del modelo clase por clase, especialmente en datasets desbalanceados.

    Parameters
    ----------
    model:
        Estimador scikit-learn ya entrenado.
    X_test:
        Features del conjunto de prueba.
    y_test:
        Etiquetas reales del conjunto de prueba.
    model_name:
        Identificador del modelo. Define el nombre del archivo de salida.

    Returns
    -------
    str
        Ruta absoluta del archivo TXT generado.

    Side effects
    ------------
    Escribe ``results/reports/<model_name>_classification_report.txt``.
    """
    _ensure_results_dirs()

    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, digits=4)

    report_path = REPORTS_DIR / f"{model_name}_classification_report.txt"
    report_path.write_text(report, encoding="utf-8")
    return str(report_path)


def plot_and_save_confusion_matrix(
    model: Any,
    X_test,
    y_test,
    model_name: str,
) -> str:
    """Genera, visualiza y guarda la matriz de confusión como imagen PNG.

    La matriz de confusión permite analizar los errores del modelo por tipo:
    verdaderos positivos, verdaderos negativos, falsos positivos y falsos
    negativos. El color de las celdas facilita la lectura visual, y los
    valores numéricos se anotan directamente sobre cada celda.

    Parameters
    ----------
    model:
        Estimador scikit-learn ya entrenado.
    X_test:
        Features del conjunto de prueba.
    y_test:
        Etiquetas reales del conjunto de prueba.
    model_name:
        Identificador del modelo. Define el nombre del archivo de salida.

    Returns
    -------
    str
        Ruta absoluta de la imagen PNG generada.

    Side effects
    ------------
    Escribe ``results/plots/<model_name>_confusion_matrix.png`` a 200 DPI.
    Cierra la figura tras guardarla para liberar memoria.
    """
    _ensure_results_dirs()

    y_pred = model.predict(X_test)
    labels = np.unique(y_test)
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion Matrix",
    )

    # Anotación numérica en cada celda con contraste automático
    thresh = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i,
                format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()
    out_path = PLOTS_DIR / f"{model_name}_confusion_matrix.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return str(out_path)


def plot_and_save_roc_curve(
    model: Any,
    X_test,
    y_test,
    model_name: str,
) -> str | None:
    """Genera, visualiza y guarda la curva ROC como imagen PNG.

    La curva ROC representa el trade-off entre la tasa de verdaderos
    positivos (TPR) y la tasa de falsos positivos (FPR) a lo largo de
    todos los umbrales de clasificación posibles. El área bajo la curva
    (AUC) resume la capacidad discriminatoria del modelo en un único valor:
    AUC = 1.0 es discriminación perfecta; AUC = 0.5 equivale a un
    clasificador aleatorio.

    Para clasificación multiclase se aplica la estrategia One-vs-Rest
    (OvR), trazando una curva por clase.

    Parameters
    ----------
    model:
        Estimador scikit-learn ya entrenado. Si no implementa
        ``predict_proba``, retorna ``None`` sin generar imagen.
    X_test:
        Features del conjunto de prueba.
    y_test:
        Etiquetas reales del conjunto de prueba.
    model_name:
        Identificador del modelo. Define el nombre del archivo de salida.

    Returns
    -------
    str | None
        Ruta absoluta de la imagen PNG generada, o ``None`` si el modelo
        no soporta probabilidades.

    Side effects
    ------------
    Escribe ``results/plots/<model_name>_roc_curve.png`` a 200 DPI.
    Cierra la figura tras guardarla para liberar memoria.
    """
    _ensure_results_dirs()

    if not hasattr(model, "predict_proba"):
        return None

    proba = model.predict_proba(X_test)
    labels = np.unique(y_test)

    fig, ax = plt.subplots(figsize=(7, 6))

    if proba.ndim == 2 and proba.shape[1] == 2:
        # Clasificación binaria
        pos_label = labels[1] if len(labels) > 1 else 1
        fpr, tpr, _ = roc_curve(y_test, proba[:, 1], pos_label=pos_label)
        ax.plot(fpr, tpr, label="ROC")
        ax.plot([0, 1], [0, 1], "--", color="gray")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve (binary)")
        ax.legend(loc="lower right")
    else:
        # Clasificación multiclase: One-vs-Rest
        y_bin = label_binarize(y_test, classes=labels)
        for i, cls in enumerate(labels):
            fpr, tpr, _ = roc_curve(y_bin[:, i], proba[:, i])
            ax.plot(fpr, tpr, label=f"Class {cls}")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve (multiclass OvR)")
        ax.legend(loc="lower right")

    fig.tight_layout()
    out_path = PLOTS_DIR / f"{model_name}_roc_curve.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return str(out_path)
