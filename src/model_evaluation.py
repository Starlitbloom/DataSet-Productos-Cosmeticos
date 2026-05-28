"""src.model_evaluation

Módulo de utilidades para evaluar modelos en clasificación binaria.

Incluye funciones reutilizables para:
- Métricas en conjunto de test: accuracy, f1-score, precision, recall, roc_auc.
- classification report.
- Matriz de confusión (con heatmap opcional).
- Curva ROC (fpr, tpr y AUC).
- Comparación de múltiples modelos ordenados por F1-score.
- Guardado automático de métricas/plots/reports en estructura de carpetas.

Este módulo está pensado para integrarse en proyectos académicos y flujos
basados en scikit-learn, incluyendo pipelines con preprocesamiento.

Notas
-----
- Se asume clasificación binaria para ROC-AUC y para la inferencia de la
  etiqueta positiva.
- La salida se guarda en rutas relativas a este módulo, creando
  automáticamente las carpetas necesarias.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.base import BaseEstimator
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


ArrayLike1D = Union[Sequence[Any], np.ndarray, pd.Series]


def _to_1d_array(y: ArrayLike1D) -> np.ndarray:
    """Convierte una entrada a un array 1D de numpy.

    Parameters
    ----------
    y : ArrayLike1D
        Etiquetas o valores numéricos.

    Returns
    -------
    np.ndarray
        Array 1D.
    """
    if isinstance(y, pd.Series):
        return y.to_numpy()
    arr = np.asarray(y)
    if arr.ndim != 1:
        return arr.reshape(-1)
    return arr


def _get_binary_positive_label(y_true: np.ndarray) -> Any:
    """Infiere la etiqueta positiva para clasificación binaria.

    Regla por defecto:
    - Si existen exactamente 2 clases, la etiqueta positiva será la máxima.
    - Si no puede inferirse, se usa 1.

    Parameters
    ----------
    y_true : np.ndarray
        Etiquetas reales.

    Returns
    -------
    Any
        Etiqueta positiva.
    """
    uniq = np.unique(y_true)
    if uniq.shape[0] == 2:
        try:
            return uniq.max()
        except TypeError:
            return 1
    return 1


def _predict_scores(model: BaseEstimator, X: Any) -> np.ndarray:
    """Obtiene un score continuo para curva ROC.

    Preferencias:
    - ``predict_proba`` usando probabilidad de la clase positiva.
    - ``decision_function`` si existe.

    Parameters
    ----------
    model : BaseEstimator
        Modelo entrenado de scikit-learn.
    X : Any
        Features de entrada.

    Returns
    -------
    np.ndarray
        Scores continuos 1D (mayor => clase positiva).

    Raises
    ------
    AttributeError
        Si el modelo no soporta ``predict_proba`` ni ``decision_function``.
    """
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        proba_arr = np.asarray(proba)
        if proba_arr.ndim != 2:
            raise ValueError("predict_proba debe devolver una matriz 2D.")

        # Asume clase positiva en columna 1 (convención habitual).
        if proba_arr.shape[1] < 2:
            raise ValueError("predict_proba debe tener al menos 2 columnas para binaria.")
        return proba_arr[:, 1]

    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        scores_arr = np.asarray(scores)
        if scores_arr.ndim != 1:
            scores_arr = scores_arr.reshape(-1)
        return scores_arr

    raise AttributeError(
        "El modelo no implementa predict_proba ni decision_function; no es posible calcular ROC-AUC/ROC curve."
    )


def evaluate_binary_classification(
    model: BaseEstimator,
    X_test: Any,
    y_test: ArrayLike1D,
    *,
    target_names: Optional[Sequence[str]] = None,
    return_confusion_df: bool = False,
) -> Dict[str, Any]:
    """Evalúa un modelo binario en conjunto de test.

    Parameters
    ----------
    model : BaseEstimator
        Modelo entrenado que implementa ``predict`` y preferiblemente
        ``predict_proba``.
    X_test : Any
        Features del conjunto de test.
    y_test : ArrayLike1D
        Etiquetas reales (binarias).
    target_names : Optional[Sequence[str]], default=None
        Nombres para las clases en el ``classification_report``.
    return_confusion_df : bool, default=False
        Si True, incluye la matriz de confusión como :class:`pandas.DataFrame`.

    Returns
    -------
    Dict[str, Any]
        Diccionario con:
        - ``accuracy``
        - ``f1_score``
        - ``precision``
        - ``recall``
        - ``roc_auc``
        - ``classification_report`` (str)
        - ``confusion_matrix`` (np.ndarray)
        - ``confusion_matrix_df`` (opcional, pd.DataFrame)
        - ``roc_curve``: dict con ``fpr``, ``tpr`` y ``auc``
        - ``y_pred`` (np.ndarray)
        - ``y_scores`` (np.ndarray)

    Raises
    ------
    ValueError
        Si no se puede calcular roc_auc por falta de scores.
    """
    y_true = _to_1d_array(y_test)

    y_pred = np.asarray(model.predict(X_test))
    y_scores = _predict_scores(model, X_test)

    accuracy = float(accuracy_score(y_true, y_pred))
    pos_label = _get_binary_positive_label(y_true)

    f1 = float(f1_score(y_true, y_pred, pos_label=pos_label))
    precision = float(precision_score(y_true, y_pred, pos_label=pos_label))
    recall = float(recall_score(y_true, y_pred, pos_label=pos_label))

    # ROC-AUC
    try:
        roc_auc = float(roc_auc_score(y_true, y_scores))
    except Exception as e:
        raise ValueError(f"No se pudo calcular ROC-AUC: {e}")


    report = classification_report(
        y_true,
        y_pred,
        target_names=target_names,
        digits=4,
    )

    cm = confusion_matrix(y_true, y_pred)

    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_payload = {"fpr": fpr, "tpr": tpr, "auc": roc_auc}

    result: Dict[str, Any] = {
        "accuracy": accuracy,
        "f1_score": f1,
        "precision": precision,
        "recall": recall,
        "roc_auc": roc_auc,
        "classification_report": report,
        "confusion_matrix": cm,
        "roc_curve": roc_payload,
        "y_pred": y_pred,
        "y_scores": y_scores,
    }

    if return_confusion_df:
        # Si y contiene 0/1, asigna índices 0/1; si no, usa etiquetas reales.
        classes = np.unique(y_true)
        if len(classes) == cm.shape[0] == cm.shape[1]:
            cm_df = pd.DataFrame(cm, index=classes, columns=classes)
        else:
            cm_df = pd.DataFrame(cm)
        result["confusion_matrix_df"] = cm_df

    return result


def compare_multiple_models_by_f1(
    models: Mapping[str, BaseEstimator],
    X_test: Any,
    y_test: ArrayLike1D,
    *,
    target_names: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Compara múltiples modelos en test y devuelve un ranking por F1-score.

    Parameters
    ----------
    models : Mapping[str, BaseEstimator]
        Diccionario ``{nombre: modelo}`` con modelos entrenados.
    X_test : Any
        Features del conjunto de test.
    y_test : ArrayLike1D
        Etiquetas reales (binarias).
    target_names : Optional[Sequence[str]], default=None
        Nombres para clases (solo para el classification report; no se retorna
        en la tabla).

    Returns
    -------
    pd.DataFrame
        Tabla con columnas:
        - model
        - accuracy
        - f1_score
        - precision
        - recall
        - roc_auc

        Ordenada descendentemente por ``f1_score``.
    """
    rows = []
    for name, model in models.items():
        metrics = evaluate_binary_classification(
            model=model,
            X_test=X_test,
            y_test=y_test,
            target_names=target_names,
            return_confusion_df=False,
        )

        rows.append(
            {
                "model": name,
                "accuracy": metrics["accuracy"],
                "f1_score": metrics["f1_score"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "roc_auc": metrics["roc_auc"],
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(by="f1_score", ascending=False).reset_index(drop=True)
    return df


def evaluate_and_save_metrics(
    model: BaseEstimator,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str,
) -> Dict[str, float]:
    """Evalúa métricas y las guarda en un resumen de texto.

    Parameters
    ----------
    model : BaseEstimator
        Modelo entrenado.
    X_test : pd.DataFrame
        Features de prueba.
    y_test : pd.Series
        Etiqueta binaria real.
    model_name : str
        Nombre del modelo (usado como prefijo para archivos).

    Returns
    -------
    Dict[str, float]
        Diccionario con métricas calculadas.

    Notes
    -----
    Guarda el archivo en:
    - ``../results/metrics/{model_name}_summary.txt``
    """
    results_dir = os.path.join("..", "results")
    metrics_dir = os.path.join(results_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)

    try:
        metrics_payload = evaluate_binary_classification(
            model=model,
            X_test=X_test,
            y_test=y_test,
            target_names=None,
            return_confusion_df=False,
        )

        metrics = {
            "Accuracy": float(metrics_payload["accuracy"]),
            "F1-Score": float(metrics_payload["f1_score"]),
            "Precision": float(metrics_payload["precision"]),
            "Recall": float(metrics_payload["recall"]),
            "ROC-AUC": float(metrics_payload["roc_auc"]),
        }

        # Impresión en consola
        print(f"\n==== Métricas - {model_name} ====")
        for k, v in metrics.items():
            print(f"{k}: {v:.6f}")

        # Guardado en texto
        out_path = os.path.join(metrics_dir, f"{model_name}_summary.txt")
        lines = [f"Modelo: {model_name}", "", "Métricas:"]
        for k, v in metrics.items():
            lines.append(f"- {k}: {v:.6f}")

        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

        return metrics

    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Error evaluando y guardando métricas para {model_name}: {exc}") from exc


def save_classification_report(
    model: BaseEstimator,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str,
) -> str:
    """Genera y guarda el classification report completo.

    Parameters
    ----------
    model : BaseEstimator
        Modelo entrenado.
    X_test : pd.DataFrame
        Features de prueba.
    y_test : pd.Series
        Etiqueta binaria real.
    model_name : str
        Nombre del modelo (usado como prefijo para archivos).

    Returns
    -------
    str
        Reporte en texto generado por scikit-learn.

    Notes
    -----
    Guarda el archivo en:
    - ``../results/reports/{model_name}_classification_report.txt``
    """
    reports_dir = os.path.join("..", "results", "reports")
    os.makedirs(reports_dir, exist_ok=True)

    try:
        y_true = _to_1d_array(y_test)
        y_pred = np.asarray(model.predict(X_test))

        report = classification_report(
            y_true,
            y_pred,
            digits=4,
        )

        out_path = os.path.join(reports_dir, f"{model_name}_classification_report.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(f"Classification Report - {model_name}\n\n")
            f.write(report)
            f.write("\n")

        return report

    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"Error generando/guardando classification report para {model_name}: {exc}"
        ) from exc


def plot_and_save_confusion_matrix(
    model: BaseEstimator,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str,
    *,
    cmap: str = "PuRd",
) -> str:
    """Genera y guarda la matriz de confusión en formato imagen.

    Parameters
    ----------
    model : BaseEstimator
        Modelo entrenado.
    X_test : pd.DataFrame
        Features de prueba.
    y_test : pd.Series
        Etiqueta binaria real.
    model_name : str
        Nombre del modelo (usado como prefijo para archivos).
    cmap : str, default="PuRd"
        Paleta de colores para el heatmap.

    Returns
    -------
    str
        Ruta del archivo PNG generado.

    Notes
    -----
    Guarda la imagen en:
    - ``../results/plots/{model_name}_confusion_matrix.png``
    """
    plots_dir = os.path.join("..", "results", "plots")
    os.makedirs(plots_dir, exist_ok=True)

    try:
        y_true = _to_1d_array(y_test)
        y_pred = np.asarray(model.predict(X_test))

        cm = confusion_matrix(y_true, y_pred)
        labels = np.unique(y_true)

        plt.figure(figsize=(6.5, 5.5))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap=cmap,
            cbar=True,
            linewidths=0.5,
            linecolor="white",
            xticklabels=labels,
            yticklabels=labels,
        )
        plt.title(f"Confusion Matrix - {model_name}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()

        out_path = os.path.join(plots_dir, f"{model_name}_confusion_matrix.png")
        plt.savefig(out_path, dpi=200)
        plt.close()

        return out_path

    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"Error generando/guardando confusion matrix para {model_name}: {exc}"
        ) from exc


def plot_and_save_roc_curve(
    model: BaseEstimator,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str,
) -> str:
    """Genera y guarda la curva ROC con AUC.

    Parameters
    ----------
    model : BaseEstimator
        Modelo entrenado.
    X_test : pd.DataFrame
        Features de prueba.
    y_test : pd.Series
        Etiqueta binaria real.
    model_name : str
        Nombre del modelo (usado como prefijo para archivos).

    Returns
    -------
    str
        Ruta del archivo PNG generado.

    Notes
    -----
    Guarda la imagen en:
    - ``../results/plots/{model_name}_roc_curve.png``

    Raises
    ------
    ValueError
        Si no es posible calcular ROC-AUC/ROC curve por falta de scores.
    """
    plots_dir = os.path.join("..", "results", "plots")
    os.makedirs(plots_dir, exist_ok=True)

    try:
        y_true = _to_1d_array(y_test)
        y_scores = _predict_scores(model, X_test)

        fpr, tpr, _ = roc_curve(y_true, y_scores)
        auc_value = float(roc_auc_score(y_true, y_scores))

        plt.figure(figsize=(7, 5.5))
        plt.plot(fpr, tpr, color="#2c7fb8", lw=2, label=f"ROC curve (AUC = {auc_value:.4f})")
        plt.plot([0, 1], [0, 1], color="gray", lw=1.5, linestyle="--", label="Random")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve - {model_name}")
        plt.legend(loc="lower right")
        plt.grid(alpha=0.25)
        plt.tight_layout()

        out_path = os.path.join(plots_dir, f"{model_name}_roc_curve.png")
        plt.savefig(out_path, dpi=200)
        plt.close()

        return out_path

    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Error generando/guardando ROC curve para {model_name}: {exc}") from exc

