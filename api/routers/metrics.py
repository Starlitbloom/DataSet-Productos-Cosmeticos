"""api.routers.metrics
======================
Endpoints relacionados con métricas del modelo de ML.

Endpoints
---------
GET /metrics/model   — métricas del modelo final (GB + clustering)
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import text

from api.database import get_db
from api.models import ModelMetric

router = APIRouter(prefix="/metrics", tags=["Metrics"])


@router.get("/model", response_model=ModelMetric, summary="Métricas del modelo final")
def get_model_metrics(db: Session = Depends(get_db)):
    """Retorna las métricas del modelo de ML entrenado en EP2.

    Incluye parámetros del GradientBoostingClassifier y métricas
    de clustering KMeans (silhouette score, inercia, mejor k).

    Lanza **404** si la tabla de métricas no existe o está vacía.
    """
    try:
        row = db.execute(
            text("SELECT * FROM model_metrics LIMIT 1")
        ).mappings().first()
    except Exception:
        raise HTTPException(
            status_code=404,
            detail="Tabla model_metrics no disponible. Ejecuta el pipeline ETL primero."
        )

    if row is None:
        raise HTTPException(status_code=404, detail="No hay métricas registradas.")

    return dict(row)
