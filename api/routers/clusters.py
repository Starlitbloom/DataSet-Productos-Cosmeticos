"""api.routers.clusters
=======================
Endpoints relacionados con los clusters KMeans del análisis no supervisado.

Endpoints
---------
GET /clusters   — resumen de todos los clusters evaluados
"""

from __future__ import annotations

from typing import List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import text
from pydantic import BaseModel
from typing import Optional

from api.database import get_db

router = APIRouter(prefix="/clusters", tags=["Clusters"])


class ClusterSummary(BaseModel):
    k:          int
    inertia:    float
    silhouette: float
    is_best:    bool

    model_config = {"from_attributes": True}


@router.get("", response_model=List[ClusterSummary], summary="Resumen de clusters KMeans")
def get_clusters(db: Session = Depends(get_db)):
    """Retorna el resumen de los clusters evaluados durante el análisis no supervisado.

    Para cada valor de k evaluado, incluye:
    - **k**: número de clusters
    - **inertia**: inercia del modelo (suma de distancias al centroide)
    - **silhouette**: coeficiente de silueta (mayor = mejor separación)
    - **is_best**: True para el k seleccionado como óptimo
    """
    try:
        rows = db.execute(
            text("SELECT k, inertia, silhouette, is_best FROM clusters ORDER BY k")
        ).mappings().all()
    except Exception:
        raise HTTPException(
            status_code=404,
            detail="Tabla clusters no disponible. Ejecuta el pipeline ETL primero."
        )

    if not rows:
        raise HTTPException(status_code=404, detail="No hay datos de clustering.")

    return [dict(r) for r in rows]
