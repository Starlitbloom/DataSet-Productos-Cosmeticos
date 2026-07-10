"""api.routers.predict
=======================
Endpoint de predicción en tiempo real usando el modelo GradientBoosting
entrenado en EP2.

El modelo serializado (``models/trained_models/gb_optimized.pkl``) y el
preprocessor (``data/processed/preprocessor.pkl``) se cargan una sola vez
al primer request y se reutilizan en cada llamada posterior (patrón singleton),
evitando el costo de deserialización repetida.

Endpoints
---------
- ``POST /predict``       — Predicción individual.
- ``POST /predict/batch`` — Predicción en batch (hasta 100 registros).

Cómo probarlo en Swagger
-------------------------
1. Abre http://localhost:8000/docs
2. Ve a la sección "predict"
3. Usa el body de ejemplo con rating=4.5 y price_usd=35.0
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/predict", tags=["predict"])

# ---------------------------------------------------------------------------
# Rutas de los artefactos — se resuelven desde la raíz del proyecto
# ---------------------------------------------------------------------------

# En Docker, WORKDIR=/app y los volúmenes montan models/ y data/ ahí mismo.
# Localmente, la raíz del proyecto está 2 niveles arriba de este archivo.
_BASE = Path(os.getenv("PROJECT_ROOT", Path(__file__).resolve().parents[2]))

MODEL_PATH        = _BASE / "models" / "trained_models" / "gb_optimized.pkl"
PREPROCESSOR_PATH = _BASE / "data" / "processed" / "preprocessor.pkl"

# Singletons — se populan en la primera llamada
_model        = None
_preprocessor = None


def _load_artifacts():
    """Carga el modelo y preprocessor la primera vez que se necesitan.

    Returns
    -------
    tuple
        (model, preprocessor). preprocessor puede ser None si el archivo
        no existe — en ese caso se usa imputación básica con medianas.

    Raises
    ------
    HTTPException
        503 si el modelo serializado no existe todavía.
    """
    global _model, _preprocessor

    if _model is None:
        if not MODEL_PATH.exists():
            raise HTTPException(
                status_code=503,
                detail=(
                    f"Modelo no encontrado en '{MODEL_PATH}'. "
                    "Asegúrate de que el volumen 'models/' esté montado "
                    "en el contenedor y que el pipeline de EP2 haya sido "
                    "ejecutado para entrenar y serializar el modelo."
                ),
            )
        logger.info("Cargando modelo desde %s", MODEL_PATH)
        _model = joblib.load(MODEL_PATH)
        logger.info("Modelo listo: %s", _model.__class__.__name__)

    if _preprocessor is None and PREPROCESSOR_PATH.exists():
        logger.info("Cargando preprocessor desde %s", PREPROCESSOR_PATH)
        _preprocessor = joblib.load(PREPROCESSOR_PATH)
        logger.info("Preprocessor listo.")

    return _model, _preprocessor


# ---------------------------------------------------------------------------
# Modelos Pydantic
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    """Features de entrada para predecir si un producto será recomendado.

    Todos los campos son opcionales: el preprocessor aplica imputación
    de mediana (numéricas) y moda (categóricas), igual que durante el
    entrenamiento. Proveer más campos mejora la calidad de la predicción.

    Las features más influyentes según el modelo son:
    ``rating``, ``price_usd`` y ``helpfulness``.
    """

    rating: Optional[float] = Field(
        None, ge=1.0, le=5.0,
        description="Calificación del producto (1.0 – 5.0).",
        json_schema_extra={"example": 4.5},
    )
    price_usd: Optional[float] = Field(
        None, ge=0.0,
        description="Precio del producto en USD.",
        json_schema_extra={"example": 35.0},
    )
    helpfulness: Optional[float] = Field(
        None, ge=0.0, le=1.0,
        description="Utilidad de la reseña (0.0 – 1.0).",
        json_schema_extra={"example": 0.8},
    )
    total_feedback_count: Optional[float] = Field(
        None, ge=0,
        description="Total de votos de utilidad recibidos.",
        json_schema_extra={"example": 25.0},
    )
    total_pos_feedback_count: Optional[float] = Field(
        None, ge=0,
        description="Votos positivos de utilidad.",
        json_schema_extra={"example": 20.0},
    )
    total_neg_feedback_count: Optional[float] = Field(
        None, ge=0,
        description="Votos negativos de utilidad.",
        json_schema_extra={"example": 5.0},
    )
    skin_tone: Optional[str] = Field(
        None,
        description="Tono de piel del usuario.",
        json_schema_extra={"example": "medium"},
    )
    eye_color: Optional[str] = Field(
        None,
        description="Color de ojos del usuario.",
        json_schema_extra={"example": "brown"},
    )
    skin_type: Optional[str] = Field(
        None,
        description="Tipo de piel del usuario.",
        json_schema_extra={"example": "combination"},
    )
    hair_color: Optional[str] = Field(
        None,
        description="Color de cabello del usuario.",
        json_schema_extra={"example": "brunette"},
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "rating": 4.5,
                "price_usd": 35.0,
                "helpfulness": 0.8,
                "total_feedback_count": 25.0,
                "total_pos_feedback_count": 20.0,
                "total_neg_feedback_count": 5.0,
                "skin_tone": "medium",
                "eye_color": "brown",
                "skin_type": "combination",
                "hair_color": "brunette",
            }
        }
    }


class PredictResponse(BaseModel):
    """Resultado de la predicción de recomendación."""

    is_recommended: int = Field(
        ..., description="Predicción: 1 = recomendado, 0 = no recomendado."
    )
    probability_recommended: float = Field(
        ..., description="Probabilidad de que el producto sea recomendado (0–1)."
    )
    probability_not_recommended: float = Field(
        ..., description="Probabilidad de que el producto NO sea recomendado (0–1)."
    )
    model: str = Field(
        ..., description="Nombre del modelo usado para la predicción."
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "is_recommended": 1,
                "probability_recommended": 0.9234,
                "probability_not_recommended": 0.0766,
                "model": "GradientBoostingClassifier",
            }
        }
    }


class BatchPredictRequest(BaseModel):
    """Lista de registros para predicción en batch (máximo 100)."""

    records: list[PredictRequest] = Field(
        ..., min_length=1, max_length=100,
        description="Lista de features para predecir (1–100 registros).",
    )


class BatchPredictResponse(BaseModel):
    """Respuesta de predicción en batch."""

    total: int = Field(..., description="Cantidad de predicciones realizadas.")
    predictions: list[PredictResponse] = Field(
        ..., description="Predicciones en el mismo orden que los registros enviados."
    )


# ---------------------------------------------------------------------------
# Lógica de predicción
# ---------------------------------------------------------------------------

def _to_dataframe(records: list[PredictRequest]) -> pd.DataFrame:
    """Convierte una lista de PredictRequest a un DataFrame de pandas.

    Los campos None quedan como NaN para que el preprocessor los impute.
    """
    return pd.DataFrame([
        {
            "rating"                  : r.rating,
            "price_usd"               : r.price_usd,
            "helpfulness"             : r.helpfulness,
            "total_feedback_count"    : r.total_feedback_count,
            "total_pos_feedback_count": r.total_pos_feedback_count,
            "total_neg_feedback_count": r.total_neg_feedback_count,
            "skin_tone"               : r.skin_tone,
            "eye_color"               : r.eye_color,
            "skin_type"               : r.skin_type,
            "hair_color"              : r.hair_color,
        }
        for r in records
    ])


def _run_prediction(df: pd.DataFrame, model, preprocessor) -> list[PredictResponse]:
    """Aplica preprocessor y modelo y devuelve las respuestas."""
    try:
        if preprocessor is not None:
            X = preprocessor.transform(df)
        else:
            # Fallback sin preprocessor: imputar medianas en columnas numéricas
            num_cols = df.select_dtypes(include="number").columns
            df[num_cols] = df[num_cols].fillna(df[num_cols].median())
            X = df[num_cols].values

        preds  = model.predict(X)
        probas = model.predict_proba(X)

        classes  = np.asarray(getattr(model, "classes_", [0, 1]))
        pos_idx  = int(np.where(classes == 1)[0][0]) if 1 in classes else 1
        neg_idx  = 1 - pos_idx

        return [
            PredictResponse(
                is_recommended=int(pred),
                probability_recommended=float(prob[pos_idx]),
                probability_not_recommended=float(prob[neg_idx]),
                model=model.__class__.__name__,
            )
            for pred, prob in zip(preds, probas)
        ]

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Error en predicción: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=f"Error interno durante la predicción: {exc}",
        ) from exc


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post(
    "",
    response_model=PredictResponse,
    summary="Predecir si un producto será recomendado",
    description=(
        "Recibe las características de una reseña/producto y usa el modelo "
        "**GradientBoostingClassifier** entrenado en EP2 para predecir si "
        "el usuario lo recomendaría (`is_recommended = 1`) o no (`0`), "
        "junto con la probabilidad de cada clase.\n\n"
        "Todos los campos son opcionales — el modelo imputa valores faltantes "
        "con medianas/modas. Los campos más influyentes son `rating` y `price_usd`."
    ),
)
def predict_single(request: PredictRequest) -> PredictResponse:
    """Predicción individual de recomendación de producto.

    Raises
    ------
    HTTPException
        503 si el modelo no está disponible.
        500 si ocurre un error durante la predicción.
    """
    model, preprocessor = _load_artifacts()
    df = _to_dataframe([request])
    return _run_prediction(df, model, preprocessor)[0]


@router.post(
    "/batch",
    response_model=BatchPredictResponse,
    summary="Predicción en batch (hasta 100 productos)",
    description=(
        "Recibe hasta 100 registros y devuelve predicciones para todos. "
        "Las predicciones se devuelven en el **mismo orden** que los registros enviados."
    ),
)
def predict_batch(request: BatchPredictRequest) -> BatchPredictResponse:
    """Predicción en batch de hasta 100 registros.

    Raises
    ------
    HTTPException
        503 si el modelo no está disponible.
        500 si ocurre un error durante la predicción.
    """
    model, preprocessor = _load_artifacts()
    df      = _to_dataframe(request.records)
    results = _run_prediction(df, model, preprocessor)
    return BatchPredictResponse(total=len(results), predictions=results)