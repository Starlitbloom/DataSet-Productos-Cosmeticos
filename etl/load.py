"""etl.load
===========
Carga de datos limpios al Data Warehouse (PostgreSQL).

Escribe las tablas limpias y los resultados del pipeline al data warehouse,
incluyendo:
- ``clean_products``: productos transformados con precio en CLP.
- ``clean_reviews``: reseñas transformadas.
- ``sentiment_results``: resultados del análisis de sentimiento.
- ``exchange_rates``: historial de tipos de cambio consultados.
- ``model_metrics``: métricas del modelo de EP2 reutilizado.
- ``clusters``: asignaciones de cluster del modelo K-Means de EP2.

Notas de implementación
-----------------------
- Se usa ``chunksize`` en ``to_sql`` para manejar datasets grandes.
- El parámetro ``if_exists`` por defecto es ``"replace"`` para corridas
  idempotentes — cada corrida del ETL sobreescribe con datos frescos.
- Los errores se registran con ``logging`` y se relanza la excepción
  para que ``pipeline.py`` pueda decidir cómo manejarlos.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Función auxiliar de carga genérica
# ---------------------------------------------------------------------------

def _load_table(
    df: pd.DataFrame,
    table_name: str,
    engine: Engine,
    if_exists: str = "replace",
    chunksize: int = 10_000,
) -> None:
    """Carga un DataFrame a PostgreSQL como tabla.

    Parameters
    ----------
    df:
        DataFrame a cargar.
    table_name:
        Nombre de la tabla destino en PostgreSQL.
    engine:
        Engine de SQLAlchemy.
    if_exists:
        ``"replace"`` (default), ``"append"`` o ``"fail"``.
    chunksize:
        Filas por lote para inserciones eficientes.

    Raises
    ------
    SQLAlchemyError
        Si la carga falla a nivel de base de datos.
    """
    logger.info(
        "  → Cargando '%s': %d filas × %d columnas (if_exists='%s')...",
        table_name, len(df), df.shape[1], if_exists,
    )

    try:
        df.to_sql(
            table_name,
            engine,
            if_exists=if_exists,
            index=False,
            chunksize=chunksize,
            method="multi",
        )
        logger.info("  ✓ '%s' cargada correctamente.", table_name)
    except SQLAlchemyError as exc:
        logger.error("Error al cargar tabla '%s': %s", table_name, exc)
        raise


# ---------------------------------------------------------------------------
# Funciones de carga por tabla
# ---------------------------------------------------------------------------

def load_clean_products(
    df: pd.DataFrame,
    engine: Engine,
    if_exists: str = "replace",
) -> None:
    """Carga productos limpios a la tabla ``clean_products``.

    Parameters
    ----------
    df:
        DataFrame limpio de productos (salida de ``transform_products``).
    engine:
        Engine de SQLAlchemy conectado al data warehouse.
    if_exists:
        Comportamiento si la tabla ya existe.
    """
    _load_table(df, "clean_products", engine, if_exists=if_exists)


def load_clean_reviews(
    df: pd.DataFrame,
    engine: Engine,
    if_exists: str = "replace",
) -> None:
    """Carga reseñas limpias a la tabla ``clean_reviews``.

    Parameters
    ----------
    df:
        DataFrame limpio de reseñas (salida de ``transform_reviews``).
    engine:
        Engine de SQLAlchemy conectado al data warehouse.
    if_exists:
        Comportamiento si la tabla ya existe.
    """
    _load_table(df, "clean_reviews", engine, if_exists=if_exists, chunksize=50_000)


def load_sentiment_results(
    df: pd.DataFrame,
    engine: Engine,
    if_exists: str = "replace",
) -> None:
    """Carga resultados de sentimiento a la tabla ``sentiment_results``.

    Parameters
    ----------
    df:
        DataFrame de sentimiento (salida de ``transform_sentiment``).
    engine:
        Engine de SQLAlchemy conectado al data warehouse.
    if_exists:
        Comportamiento si la tabla ya existe.
    """
    _load_table(df, "sentiment_results", engine, if_exists=if_exists)


def load_exchange_rate(
    rate_dict: dict,
    engine: Engine,
    if_exists: str = "append",
) -> None:
    """Guarda el tipo de cambio consultado en la tabla ``exchange_rates``.

    Parameters
    ----------
    rate_dict:
        Diccionario con claves ``date``, ``rate`` y ``source``
        (salida de ``extract_exchange_rate``).
    engine:
        Engine de SQLAlchemy conectado al data warehouse.
    if_exists:
        ``"append"`` por defecto para mantener historial de tasas.
    """
    df = pd.DataFrame([{
        "date":       rate_dict["date"],
        "rate":       rate_dict["rate"],
        "source":     rate_dict["source"],
        "loaded_at":  datetime.now(timezone.utc).isoformat(),
    }])
    _load_table(df, "exchange_rates", engine, if_exists=if_exists, chunksize=1)


def load_model_metrics(
    metrics: dict,
    engine: Engine,
    if_exists: str = "append",
) -> None:
    """Guarda métricas del modelo en la tabla ``model_metrics``.

    Parameters
    ----------
    metrics:
        Diccionario con métricas del modelo (accuracy, f1, roc_auc, etc.).
        Puede venir de ``results/metrics/`` generados en EP2.
    engine:
        Engine de SQLAlchemy conectado al data warehouse.
    if_exists:
        ``"append"`` por defecto para mantener historial de corridas.
    """
    df = pd.DataFrame([{
        **metrics,
        "loaded_at": datetime.utcnow().isoformat(),
    }])
    _load_table(df, "model_metrics", engine, if_exists=if_exists, chunksize=1)


def load_clusters(
    df_clusters: pd.DataFrame,
    engine: Engine,
    if_exists: str = "replace",
) -> None:
    """Carga asignaciones de cluster K-Means a la tabla ``clusters``.

    Parameters
    ----------
    df_clusters:
        DataFrame con columnas ``author_id`` (o ``product_id``) y ``cluster``.
        Puede venir del modelo K-Means entrenado en EP2.
    engine:
        Engine de SQLAlchemy conectado al data warehouse.
    if_exists:
        Comportamiento si la tabla ya existe.
    """
    _load_table(df_clusters, "clusters", engine, if_exists=if_exists)


# ---------------------------------------------------------------------------
# Función unificada
# ---------------------------------------------------------------------------

def load_all(
    clean_products: pd.DataFrame,
    clean_reviews: pd.DataFrame,
    clean_sentiment: pd.DataFrame,
    exchange_rate: dict,
    engine: Engine,
    model_metrics: Optional[dict] = None,
    df_clusters: Optional[pd.DataFrame] = None,
) -> None:
    """Ejecuta todas las cargas al data warehouse en una sola llamada.

    Parameters
    ----------
    clean_products:
        DataFrame limpio de productos.
    clean_reviews:
        DataFrame limpio de reseñas.
    clean_sentiment:
        DataFrame limpio de sentimiento.
    exchange_rate:
        Diccionario con tasa USD→CLP.
    engine:
        Engine de SQLAlchemy conectado al data warehouse.
    model_metrics:
        Métricas del modelo (opcional).
    df_clusters:
        Asignaciones de cluster (opcional).
    """
    logger.info("=== LOAD — inicio ===")

    try:
        load_clean_products(clean_products, engine)
    except Exception as exc:
        logger.error("Falló carga de clean_products: %s", exc)
        raise

    try:
        load_clean_reviews(clean_reviews, engine)
    except Exception as exc:
        logger.error("Falló carga de clean_reviews: %s", exc)
        raise

    try:
        load_sentiment_results(clean_sentiment, engine)
    except Exception as exc:
        logger.error("Falló carga de sentiment_results: %s", exc)
        raise

    try:
        load_exchange_rate(exchange_rate, engine)
    except Exception as exc:
        logger.error("Falló carga de exchange_rates: %s", exc)
        raise

    if model_metrics:
        try:
            load_model_metrics(model_metrics, engine)
        except Exception as exc:
            logger.warning("No se pudieron cargar model_metrics: %s", exc)

    if df_clusters is not None and not df_clusters.empty:
        try:
            load_clusters(df_clusters, engine)
        except Exception as exc:
            logger.warning("No se pudieron cargar clusters: %s", exc)

    logger.info("=== LOAD — completado ===")
