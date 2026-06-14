"""etl.transform
================
Transformación y limpieza de datos crudos del pipeline ETL de Sephora.

Adapta el código de ``etl/transformers.py`` y ``etl/pipeline.py`` (EP2)
para recibir los DataFrames crudos de las 4 fuentes y devolver tablas
limpias listas para cargar al data warehouse.

Transformaciones aplicadas
--------------------------
Productos:
  - Normalización de nombres de columna (lowercase, sin espacios).
  - Imputación de nulos: mediana para numéricos, moda para categóricos.
  - Capping de outliers en ``price_usd`` y ``loves_count`` (IQR × 1.5).
  - Conversión de precio a CLP usando la tasa obtenida de la API.
  - Eliminación de duplicados por ``product_id``.

Reseñas:
  - Normalización de tipos: ``is_recommended`` → float.
  - Limpieza de ``review_text``: strip de espacios, nulos como vacío.
  - Eliminación de reseñas sin ``product_id`` ni ``author_id``.
  - Eliminación de duplicados por (``author_id``, ``product_id``).

Notas de implementación
-----------------------
- Se reutilizan los transformers de ``etl/transformers.py`` (EP2) para
  mantener consistencia con el preprocesamiento del modelado.
- Los errores se registran con ``logging``; no se usa ``print``.
- Todas las funciones son puras: reciben DataFrames y devuelven DataFrames.
"""

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Transformación de productos
# ---------------------------------------------------------------------------

def transform_products(
    df: pd.DataFrame,
    exchange_rate: float = 1.0,
) -> pd.DataFrame:
    """Limpia y transforma el DataFrame de productos crudos.

    Parameters
    ----------
    df:
        DataFrame crudo de productos (salida de ``extract_csv`` o ``extract_db``).
    exchange_rate:
        Tasa de cambio USD → CLP obtenida de la API (salida de
        ``extract_exchange_rate``). Default 1.0 si no está disponible.

    Returns
    -------
    pd.DataFrame
        DataFrame limpio con columna ``price_clp`` agregada.
    """
    logger.info("=== TRANSFORM (productos) — inicio: %d filas ===", len(df))
    df = df.copy()

    # 1. Normalizar nombres de columnas
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # 2. Eliminar duplicados por product_id
    before = len(df)
    df = df.drop_duplicates(subset=["product_id"], keep="first")
    logger.info("Duplicados eliminados (product_id): %d", before - len(df))

    # 3. Limpiar strings: strip de espacios
    for col in df.select_dtypes(include=["string"]).columns:
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].replace({"nan": np.nan, "None": np.nan, "": np.nan})

    # 4. Convertir tipos numéricos
    for col in ["price_usd", "rating", "loves_count"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 5. Imputar nulos
    if "price_usd" in df.columns:
        median_price = df["price_usd"].median()
        n_nulls = df["price_usd"].isna().sum()
        df["price_usd"] = df["price_usd"].fillna(median_price)
        logger.info("price_usd: %d nulos imputados con mediana %.2f", n_nulls, median_price)

    if "rating" in df.columns:
        df["rating"] = df["rating"].fillna(df["rating"].median())

    if "loves_count" in df.columns:
        df["loves_count"] = df["loves_count"].fillna(0.0)

    # 6. Capping de outliers (IQR × 1.5) en price_usd y loves_count
    for col in ["price_usd", "loves_count"]:
        if col in df.columns:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            n_capped = ((df[col] < lower) | (df[col] > upper)).sum()
            df[col] = np.clip(df[col], lower, upper)
            logger.info("%s: %d outliers capeados (%.2f, %.2f)", col, n_capped, lower, upper)

    # 7. Eliminar precios negativos o cero
    if "price_usd" in df.columns:
        before = len(df)
        df = df[df["price_usd"] > 0]
        logger.info("Filas con price_usd <= 0 eliminadas: %d", before - len(df))

    # 8. Convertir precio a CLP
    df["price_clp"] = (df["price_usd"] * exchange_rate).round(0)
    logger.info("Columna price_clp agregada (tasa: %.2f USD/CLP)", exchange_rate)

    logger.info("=== TRANSFORM (productos) — completado: %d filas ===", len(df))
    return df


# ---------------------------------------------------------------------------
# Transformación de reseñas
# ---------------------------------------------------------------------------

def transform_reviews(df: pd.DataFrame) -> pd.DataFrame:
    """Limpia y transforma el DataFrame de reseñas crudas.

    Parameters
    ----------
    df:
        DataFrame crudo de reseñas (salida de ``extract_csv`` o ``extract_db``).

    Returns
    -------
    pd.DataFrame
        DataFrame limpio listo para cargar al data warehouse.
    """
    logger.info("=== TRANSFORM (reseñas) — inicio: %d filas ===", len(df))
    df = df.copy()

    # 1. Normalizar nombres de columnas
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # 2. Eliminar filas sin author_id o product_id (no identificables)
    before = len(df)
    df = df.dropna(subset=["author_id", "product_id"])
    logger.info("Filas sin author_id/product_id eliminadas: %d", before - len(df))

    # 3. Convertir is_recommended a float (puede venir como int, bool o str)
    if "is_recommended" in df.columns:
        df["is_recommended"] = pd.to_numeric(df["is_recommended"], errors="coerce")
        n_nulls = df["is_recommended"].isna().sum()
        if n_nulls > 0:
            logger.warning("is_recommended: %d nulos — se eliminan estas filas", n_nulls)
            df = df.dropna(subset=["is_recommended"])
        df["is_recommended"] = df["is_recommended"].astype(float)

    # 4. Convertir rating a float
    if "rating" in df.columns:
        df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
        df["rating"] = df["rating"].fillna(df["rating"].median())
        # Asegurar rango válido
        df = df[df["rating"].between(1, 5)]

    # 5. Limpiar review_text
    if "review_text" in df.columns:
        df["review_text"] = df["review_text"].astype(str).str.strip()
        df["review_text"] = df["review_text"].replace(
            {"nan": np.nan, "None": np.nan, "": np.nan}
        )

    # 6. Convertir tipos numéricos de feedback
    for col in ["total_feedback_count", "total_neg_feedback_count", "total_pos_feedback_count"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(float)

    # 7. Limpiar columnas demográficas (skin_tone, etc.)
    demo_cols = ["skin_tone", "eye_color", "skin_type", "hair_color"]
    for col in demo_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.lower()
            df[col] = df[col].replace({"nan": np.nan, "none": np.nan, "unknown": np.nan, "": np.nan})

    # 8. Eliminar duplicados por (author_id, product_id)
    before = len(df)
    df = df.drop_duplicates(subset=["author_id", "product_id"], keep="first")
    logger.info("Duplicados eliminados (author_id, product_id): %d", before - len(df))

    logger.info("=== TRANSFORM (reseñas) — completado: %d filas ===", len(df))
    return df


# ---------------------------------------------------------------------------
# Transformación de resultados de sentimiento
# ---------------------------------------------------------------------------

def transform_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """Limpia el DataFrame de resultados de sentimiento de HuggingFace.

    Parameters
    ----------
    df:
        DataFrame con columnas ``author_id``, ``sentiment``, ``sentiment_score``,
        ``is_recommended`` (salida de ``extract_sentiment``).

    Returns
    -------
    pd.DataFrame
        DataFrame limpio listo para cargar como tabla ``sentiment_results``.
    """
    logger.info("=== TRANSFORM (sentimiento) — inicio: %d filas ===", len(df))
    df = df.copy()

    # Eliminar filas con sentimiento desconocido
    if "sentiment" in df.columns:
        before = len(df)
        df = df[df["sentiment"] != "unknown"]
        logger.info("Filas con sentimiento 'unknown' eliminadas: %d", before - len(df))

    # Asegurar tipos
    if "sentiment_score" in df.columns:
        df["sentiment_score"] = pd.to_numeric(df["sentiment_score"], errors="coerce").fillna(0.0)

    if "is_recommended" in df.columns:
        df["is_recommended"] = pd.to_numeric(df["is_recommended"], errors="coerce")

    logger.info("=== TRANSFORM (sentimiento) — completado: %d filas ===", len(df))
    return df


# ---------------------------------------------------------------------------
# Función unificada
# ---------------------------------------------------------------------------

def transform_all(
    df_products: pd.DataFrame,
    df_reviews: pd.DataFrame,
    df_sentiment: pd.DataFrame,
    exchange_rate: float = 1.0,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Ejecuta todas las transformaciones en una sola llamada.

    Parameters
    ----------
    df_products:
        DataFrame crudo de productos.
    df_reviews:
        DataFrame crudo de reseñas.
    df_sentiment:
        DataFrame de resultados de sentimiento.
    exchange_rate:
        Tasa USD → CLP de la API de tipo de cambio.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        (clean_products, clean_reviews, clean_sentiment)
    """
    logger.info("=== TRANSFORM — inicio ===")
    clean_products  = transform_products(df_products, exchange_rate=exchange_rate)
    clean_reviews   = transform_reviews(df_reviews)
    clean_sentiment = transform_sentiment(df_sentiment)
    logger.info("=== TRANSFORM — completado ===")
    return clean_products, clean_reviews, clean_sentiment
