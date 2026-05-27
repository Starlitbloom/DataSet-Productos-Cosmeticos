"""Módulo de utilidades para el análisis de productos cosméticos.

Incluye utilidades de ETL del dataset y (a partir de la fase de modelado)
funciones para exportar el mejor modelo entrenado, sus métricas y gráficos.
"""

from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib

# Asegura backend no-interactivo (útil al correr desde script/notebook headless)
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import joblib

from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)


def _ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _safe_json(obj: Any) -> Any:
    """Convierte objetos no-JSON (numpy types) a tipos nativos."""
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    return obj


def _infer_classification_from_estimator(estimator: Any) -> bool:
    # Si tiene predict_proba asumimos clasificación
    return hasattr(estimator, "predict_proba")


# ---------------------------
# ETL utilities (existentes)
# ---------------------------

def load_all_data(folder_path: str) -> pd.DataFrame:
    """Busca todos los archivos CSV en la carpeta y los une."""
    all_files = glob.glob(os.path.join(folder_path, "*.csv"))

    if not all_files:
        raise FileNotFoundError(f"No se encontraron archivos CSV en {folder_path}")

    df_list = []
    for filename in all_files:
        try:
            df = pd.read_csv(filename)
            df_list.append(df)
            print(f"Cargado: {filename} - Filas: {len(df)}")
        except Exception as e:
            print(f"Error al cargar {filename}: {e}")

    if not df_list:
        return pd.DataFrame()

    full_df = pd.concat(df_list, ignore_index=True)
    print(f"Dataset completo cargado. Total filas: {len(full_df)}")
    return full_df


def clean_product_type(product_type: pd.Series) -> pd.Series:
    """Limpia la columna product_type."""
    return (
        product_type.fillna("Unknown")
        .astype(str)
        .str.replace("\n", "")
        .str.strip()
    )


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega columnas derivadas al DataFrame."""
    df = df.copy()
    # Usamos .get() y verificamos existencia para evitar errores si la columna no existe
    if "description" in df.columns:
        df["description_length"] = df["description"].str.len()
    if "img" in df.columns:
        df["has_image"] = df["img"].notna()
    if "shade_img" in df.columns:
        df["has_shade_image"] = df["shade_img"].notna()
    return df


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Realiza la limpieza completa del dataset."""
    df_clean = df.copy()

    # 1. Normalizar nombres de columnas a minúsculas
    df_clean.columns = [col.strip().lower() for col in df_clean.columns]

    # 2. Aplicar limpieza de tipo de producto
    if "product_type" in df_clean.columns:
        df_clean["product_type"] = clean_product_type(df_clean["product_type"])

    # 3. Agregar columnas derivadas
    df_clean = add_derived_columns(df_clean)

    # 4. Eliminar columnas no útiles si existen
    columns_to_drop = ["rating", "dupes", "price_site"]
    df_clean = df_clean.drop(columns=[col for col in columns_to_drop if col in df_clean.columns])

    return df_clean


def validate_dataset(df: pd.DataFrame) -> dict:
    """Valida la integridad del dataset."""
    # Como normalizamos a minúsculas, buscamos 'id' en lugar de 'ID'
    id_col = "id" if "id" in df.columns else ("ID" if "ID" in df.columns else None)

    validation = {
        "shape": df.shape,
        "duplicates": df.duplicated().sum(),
        "unique_ids": df[id_col].nunique() if id_col else "No ID column found",
        "null_counts": df.isnull().sum().to_dict(),
    }
    return validation


# -------------------------------------------------
# Export utilities for best model + artifacts
# -------------------------------------------------

@dataclass
class ExportConfig:
    model_output_path: str = "models/trained_models/best_model.joblib"
    metrics_output_path: str = "results/metrics/best_model_metrics.txt"
    plots_output_dir: str = "results/plots"
    artifacts_meta_output_path: str = "results/metrics/best_model_artifacts_meta.json"


def compute_classification_metrics(
    estimator: Any,
    X: Any,
    y: Any,
) -> Dict[str, Any]:
    y_pred = estimator.predict(X)

    metrics: Dict[str, Any] = {}
    metrics["accuracy"] = float(accuracy_score(y, y_pred))
    metrics["f1_weighted"] = float(f1_score(y, y_pred, average="weighted"))

    # ROC-AUC: si hay predict_proba
    if hasattr(estimator, "predict_proba"):
        proba = estimator.predict_proba(X)

        # Soporta binario y multiclase
        labels = getattr(estimator, "classes_", None)
        if labels is None and hasattr(estimator, "steps"):
            # Pipeline: el clasificador real es estimator.steps[-1][1]
            try:
                labels = estimator.named_steps["model"].classes_
            except Exception:
                labels = None

        n_classes = proba.shape[1] if proba.ndim == 2 else 1
        if n_classes == 2:
            # Usa probas de la clase positiva (la clase 1 si existe)
            # sklearn suele ordenar clases
            auc = roc_auc_score(y, proba[:, 1])
            metrics["roc_auc_ovr"] = float(auc)
        else:
            metrics["roc_auc_ovr"] = float(
                roc_auc_score(y, proba, multi_class="ovr", average="weighted")
            )

    # Report como texto
    metrics["classification_report"] = classification_report(y, y_pred, digits=4)

    return metrics


def save_confusion_matrix_plot(
    estimator: Any,
    X: Any,
    y: Any,
    output_path: str,
    normalize: Optional[str] = None,
) -> Dict[str, Any]:
    y_pred = estimator.predict(X)
    cm = confusion_matrix(y, y_pred, normalize=normalize)

    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y))
    disp.plot(ax=ax, cmap="Blues", values_format=".2f" if normalize else "d", colorbar=False)
    ax.set_title("Matriz de confusión" + (" (normalizada)" if normalize else ""))
    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return {"path": output_path, "normalize": normalize}


def save_roc_curve_plot(
    estimator: Any,
    X: Any,
    y: Any,
    output_path: str,
) -> Dict[str, Any]:
    if not hasattr(estimator, "predict_proba"):
        return {"skipped": True, "reason": "Estimator no soporta predict_proba"}

    proba = estimator.predict_proba(X)

    # binario
    if proba.ndim == 2 and proba.shape[1] == 2:
        fpr, tpr, _ = roc_curve(y, proba[:, 1])
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot(fpr, tpr, label="ROC")
        ax.plot([0, 1], [0, 1], "--", color="gray")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("Curva ROC (binaria)")
        ax.legend(loc="lower right")
        plt.tight_layout()
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return {"path": output_path, "mode": "binary"}

    # multiclase: generamos ROC por clase (micro/promediadas) con RocCurveDisplay
    fig, ax = plt.subplots(figsize=(7, 6))
    labels = getattr(estimator, "classes_", None)
    if labels is None and hasattr(estimator, "steps"):
        try:
            labels = estimator.named_steps["model"].classes_
        except Exception:
            labels = None

    try:
        RocCurveDisplay.from_predictions(
            y_true=y,
            y_pred=proba,
            name="ROC OvR",
            ax=ax,
            plot_chance_level=True,
            response_method="predict_proba",
        )
    except Exception:
        # fallback manual OvR
        classes = np.unique(y) if labels is None else labels
        for i, cls in enumerate(classes):
            y_bin = (y == cls).astype(int)
            fpr, tpr, _ = roc_curve(y_bin, proba[:, i])
            ax.plot(fpr, tpr, label=f"Clase {cls}")
        ax.plot([0, 1], [0, 1], "--", color="gray")
        ax.legend(loc="lower right")

    ax.set_title("Curva ROC (multiclase)")
    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return {"path": output_path, "mode": "multiclass"}


def export_best_model_and_reports(
    *,
    best_estimator: Any,
    X_test: Any,
    y_test: Any,
    export_config: Optional[ExportConfig] = None,
    joblib_compress: int = 3,
) -> Dict[str, Any]:
    """Serializa el mejor modelo y exporta métricas + gráficas.

    Salidas:
      - models/trained_models/best_model.joblib
      - results/metrics/best_model_metrics.txt
      - results/plots/*.png (confusion matrix + ROC si aplica)
    """

    cfg = export_config or ExportConfig()

    model_dir = _ensure_dir(Path(cfg.model_output_path).parent)
    metrics_dir = _ensure_dir(Path(cfg.metrics_output_path).parent)
    plots_dir = _ensure_dir(cfg.plots_output_dir)

    # 1) Guardar modelo
    joblib.dump(best_estimator, cfg.model_output_path, compress=joblib_compress)

    # 2) Métricas
    metrics: Dict[str, Any] = {
        "model_path": cfg.model_output_path,
    }

    metrics_payload = compute_classification_metrics(best_estimator, X_test, y_test)
    metrics.update(metrics_payload)

    # Guardar métricas en texto
    # Incluimos JSON para machine-readability + reporte humano.
    metrics_txt = []
    metrics_txt.append("BEST MODEL METRICS\n")
    metrics_txt.append("=" * 60 + "\n")
    metrics_txt.append(f"Model path: {cfg.model_output_path}\n")
    metrics_txt.append("\nResumen numérico:\n")

    for k in ["accuracy", "f1_weighted", "roc_auc_ovr"]:
        if k in metrics:
            metrics_txt.append(f"- {k}: {metrics[k]}\n")

    metrics_txt.append("\nClassification report:\n")
    metrics_txt.append(str(metrics_payload.get("classification_report", "")))
    metrics_txt.append("\n")

    metrics_dir.mkdir(parents=True, exist_ok=True)
    with open(cfg.metrics_output_path, "w", encoding="utf-8") as f:
        f.write("".join(metrics_txt))

    # 3) Meta info JSON adicional
    meta = {
        "model_output_path": cfg.model_output_path,
        "metrics_output_path": cfg.metrics_output_path,
        "plots_output_dir": cfg.plots_output_dir,
        "best_params": getattr(best_estimator, "best_params_", None),
        "n_features_in_": getattr(best_estimator, "n_features_in_", None),
    }
    meta = {k: _safe_json(v) for k, v in meta.items()}
    with open(cfg.artifacts_meta_output_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # 4) Gráficas
    # 4.1 Matriz de confusión
    cm_path = str(Path(cfg.plots_output_dir) / "confusion_matrix.png")
    save_confusion_matrix_plot(best_estimator, X_test, y_test, cm_path, normalize=None)

    # 4.2 ROC (si aplica)
    roc_path = str(Path(cfg.plots_output_dir) / "roc_curve.png")
    roc_info = save_roc_curve_plot(best_estimator, X_test, y_test, roc_path)

    return {
        "model_path": cfg.model_output_path,
        "metrics_path": cfg.metrics_output_path,
        "plots": {
            "confusion_matrix": cm_path,
            "roc_curve": roc_path,
            "roc_info": roc_info,
        },
    }


if __name__ == "__main__":
    # Ruta a los archivos cargados con Git LFS
    data_folder = "data/raw/"

    try:
        # 1. Cargar
        df_raw = load_all_data(data_folder)

        if not df_raw.empty:
            # 2. Limpiar
            df_final = clean_dataset(df_raw)

            # 3. Validar
            resumen = validate_dataset(df_final)

            print("\n--- Resumen de Validación ---")
            print(f"Filas y Columnas: {resumen['shape']}")
            print(f"Duplicados: {resumen['duplicates']}")
            print(f"IDs únicos: {resumen['unique_ids']}")
            # print("Nulos por columna:", resumen['null_counts'])

    except Exception as e:
        print(f"Error crítico: {e}")

