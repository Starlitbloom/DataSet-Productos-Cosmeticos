"""api.routers.reviews
======================
Endpoints relacionados con reseñas de productos.

Endpoints
---------
GET /reviews   — lista paginada, filtrable por product_id
"""

from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from sqlalchemy import text

from api.database import get_db
from api.models import Review

router = APIRouter(prefix="/reviews", tags=["Reviews"])


@router.get("", response_model=List[Review], summary="Listar reseñas")
def get_reviews(
    product_id: Optional[str] = Query(None, description="Filtrar por product_id"),
    min_rating: Optional[float] = Query(None, ge=1, le=5, description="Rating mínimo"),
    is_recommended: Optional[int] = Query(None, ge=0, le=1, description="1=recomendado, 0=no recomendado"),
    limit: int = Query(50, ge=1, le=5000, description="Máximo de resultados"),
    offset: int = Query(0, ge=0, description="Desplazamiento para paginación"),
    db: Session = Depends(get_db),
):
    """Retorna reseñas con filtros opcionales.

    - **product_id**: filtra reseñas de un producto específico
    - **min_rating**: solo reseñas con rating >= valor
    - **is_recommended**: 1 para recomendadas, 0 para no recomendadas
    - **limit / offset**: paginación
    """
    filters = []
    params: dict = {"limit": limit, "offset": offset}

    if product_id:
        filters.append("product_id = :product_id")
        params["product_id"] = product_id
    if min_rating is not None:
        filters.append("rating >= :min_rating")
        params["min_rating"] = min_rating
    if is_recommended is not None:
        filters.append("is_recommended = :is_recommended")
        params["is_recommended"] = float(is_recommended)

    where = ("WHERE " + " AND ".join(filters)) if filters else ""
    query = text(
        f"SELECT author_id, product_id, product_name, brand_name, rating, "
        f"is_recommended, helpfulness, total_feedback_count, "
        f"total_neg_feedback_count, total_pos_feedback_count, "
        f"submission_time, review_text, review_title, "
        f"skin_tone, eye_color, skin_type, hair_color, price_usd "
        f"FROM clean_reviews {where} LIMIT :limit OFFSET :offset"
    )

    rows = db.execute(query, params).mappings().all()
    return [dict(r) for r in rows]
