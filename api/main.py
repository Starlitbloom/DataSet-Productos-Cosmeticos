"""api.main
===========
Aplicación FastAPI principal del proyecto Sephora.

Expone los datos del data warehouse y resultados del modelo ML
a través de una API REST documentada automáticamente en /docs.

Uso local
---------
    uvicorn api.main:app --reload

Endpoints disponibles
---------------------
    GET  /products              — lista de productos con filtros
    GET  /products/{id}         — detalle de un producto
    GET  /reviews               — reseñas con filtros
    GET  /metrics/model         — métricas del modelo ML
    GET  /clusters              — resumen de clusters KMeans
    GET  /exchange-rate         — último tipo de cambio USD→CLP
    POST /predict               — predicción individual (GradientBoosting EP2)
    POST /predict/batch         — predicción en batch (hasta 100 registros)
    GET  /                      — health check
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from sqlalchemy import text

from api.database import get_db, check_connection
from api.models import ExchangeRate
from api.routers import products, reviews, metrics, clusters, predict

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("api.main")

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Sephora Intelligence API",
    description=(
        "API REST que expone productos, reseñas, métricas del modelo ML, "
        "resultados de clustering y predicciones en tiempo real "
        "del proyecto Sephora EP3."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS — permite que el dashboard (otro contenedor/puerto) consuma la API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

app.include_router(products.router)
app.include_router(reviews.router)
app.include_router(metrics.router)
app.include_router(clusters.router)
app.include_router(predict.router)

# ---------------------------------------------------------------------------
# Endpoints adicionales en main
# ---------------------------------------------------------------------------

@app.get("/", tags=["Health"], summary="Health check")
def root():
    """Verifica que la API está activa y con conexión a la base de datos."""
    db_ok = check_connection()
    return {
        "status"  : "ok",
        "database": "connected" if db_ok else "disconnected",
        "docs"    : "/docs",
    }


@app.get(
    "/exchange-rate",
    response_model=ExchangeRate,
    tags=["Exchange Rate"],
    summary="Último tipo de cambio USD→CLP",
)
def get_exchange_rate(db: Session = Depends(get_db)):
    """Retorna el último tipo de cambio USD→CLP cargado por el pipeline ETL.

    Lanza **404** si no hay datos de tipo de cambio en la base.
    """
    row = db.execute(
        text("SELECT * FROM exchange_rates ORDER BY loaded_at DESC LIMIT 1")
    ).mappings().first()

    if row is None:
        raise HTTPException(status_code=404, detail="No hay datos de tipo de cambio.")

    return dict(row)


# ---------------------------------------------------------------------------
# Evento de startup
# ---------------------------------------------------------------------------

@app.on_event("startup")
def on_startup() -> None:
    """Registra el estado de la conexión a la DB al iniciar la API."""
    if check_connection():
        logger.info("Conexión a la base de datos verificada correctamente.")
    else:
        logger.warning(
            "No se pudo conectar a la base de datos al iniciar. "
            "Los endpoints de datos devolverán errores hasta que la DB esté disponible."
        )