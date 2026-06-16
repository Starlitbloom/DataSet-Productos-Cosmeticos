"""api.routers.products
=======================
Endpoints relacionados con productos de Sephora.

Endpoints
---------
GET /products          — lista paginada con filtros opcionales
GET /products/{id}     — detalle de un producto por product_id
"""

from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import text

from api.database import get_db
from api.models import Product

router = APIRouter(prefix="/products", tags=["Products"])


@router.get("", response_model=List[Product], summary="Listar productos")
def get_products(
    category: Optional[str] = Query(None, description="Filtrar por primary_category"),
    brand: Optional[str]    = Query(None, description="Filtrar por brand_name"),
    min_price: Optional[float] = Query(None, ge=0, description="Precio mínimo en USD"),
    max_price: Optional[float] = Query(None, ge=0, description="Precio máximo en USD"),
    limit: int  = Query(50, ge=1, le=500, description="Máximo de resultados"),
    offset: int = Query(0, ge=0, description="Desplazamiento para paginación"),
    db: Session = Depends(get_db),
):
    """Retorna una lista paginada de productos con filtros opcionales.

    - **category**: filtra por categoría principal (ej. `Skincare`, `Makeup`)
    - **brand**: filtra por nombre de marca (ej. `NARS`, `Charlotte Tilbury`)
    - **min_price / max_price**: rango de precio en USD
    - **limit / offset**: paginación
    """
    filters = []
    params: dict = {"limit": limit, "offset": offset}

    if category:
        filters.append("LOWER(primary_category) = LOWER(:category)")
        params["category"] = category
    if brand:
        filters.append("LOWER(brand_name) = LOWER(:brand)")
        params["brand"] = brand
    if min_price is not None:
        filters.append("price_usd >= :min_price")
        params["min_price"] = min_price
    if max_price is not None:
        filters.append("price_usd <= :max_price")
        params["max_price"] = max_price

    where = ("WHERE " + " AND ".join(filters)) if filters else ""
    query = text(f"SELECT * FROM clean_products {where} LIMIT :limit OFFSET :offset")

    rows = db.execute(query, params).mappings().all()
    return [dict(r) for r in rows]


@router.get("/{product_id}", response_model=Product, summary="Detalle de un producto")
def get_product(product_id: str, db: Session = Depends(get_db)):
    """Retorna el detalle completo de un producto por su ``product_id``.

    Lanza **404** si el producto no existe.
    """
    row = db.execute(
        text("SELECT * FROM clean_products WHERE product_id = :pid"),
        {"pid": product_id},
    ).mappings().first()

    if row is None:
        raise HTTPException(status_code=404, detail=f"Producto '{product_id}' no encontrado.")

    return dict(row)
