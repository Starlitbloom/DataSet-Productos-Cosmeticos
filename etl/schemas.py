"""etl.schemas
==============
Contratos de datos con pandera para validar los DataFrames en cada etapa del ETL.

Define los esquemas esperados para:
- ``raw_products``: datos crudos de productos antes de transformar.
- ``raw_reviews``: datos crudos de reseñas antes de transformar.
- ``clean_products``: productos después de la transformación.
- ``clean_reviews``: reseñas después de la transformación.

Notas de implementación
-----------------------
- Se usa ``pandera`` en modo ``lazy=True`` para acumular TODOS los errores
  antes de lanzar la excepción, no solo el primero.
- Los esquemas de datos crudos son más permisivos (nullable=True en casi todo).
- Los esquemas de datos limpios son más estrictos (rangos, no nulos en clave).
- Las funciones ``validate_*`` envuelven la validación en try/except para
  registrar errores con logging sin detener el pipeline si se configura así.
"""

from __future__ import annotations

import logging
from typing import Tuple

import pandas as pd
import pandera.pandas as pa
from pandera.pandas import Column, DataFrameSchema, Check

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Esquemas de datos CRUDOS (permisivos)
# ---------------------------------------------------------------------------

RAW_PRODUCTS_SCHEMA = DataFrameSchema(
    columns={
        "product_id":       Column(str,   nullable=True),
        "product_name":     Column(str,   nullable=True),
        "brand_name":       Column(str,   nullable=True),
        "price_usd":        Column(float, nullable=True,
                                  checks=Check.greater_than_or_equal_to(0)),
        "rating":           Column(float, nullable=True,
                                  checks=Check.in_range(0, 5)),
        "loves_count":      Column(float, nullable=True,
                                  checks=Check.greater_than_or_equal_to(0)),
        "primary_category": Column(str,   nullable=True),
    },
    coerce=True,   # Intenta convertir tipos automáticamente
    strict=False,  # Permite columnas extra
    name="raw_products",
)

RAW_REVIEWS_SCHEMA = DataFrameSchema(
    columns={
        "author_id":                Column(str,   nullable=True),
        "rating":                   Column(float, nullable=True,
                                          checks=Check.in_range(1, 5)),
        "is_recommended":           Column(float, nullable=True,
                                          checks=Check.isin([0.0, 1.0])),
        "total_feedback_count":     Column(float, nullable=True,
                                          checks=Check.greater_than_or_equal_to(0)),
        "total_neg_feedback_count": Column(float, nullable=True,
                                          checks=Check.greater_than_or_equal_to(0)),
        "total_pos_feedback_count": Column(float, nullable=True,
                                          checks=Check.greater_than_or_equal_to(0)),
        "review_text":              Column(str,   nullable=True),
        "product_id":               Column(str,   nullable=True),
    },
    coerce=True,
    strict=False,
    name="raw_reviews",
)


# ---------------------------------------------------------------------------
# Esquemas de datos LIMPIOS (estrictos)
# ---------------------------------------------------------------------------

CLEAN_PRODUCTS_SCHEMA = DataFrameSchema(
    columns={
        "product_id":       Column(str,   nullable=False),
        "product_name":     Column(str,   nullable=False),
        "brand_name":       Column(str,   nullable=True),
        "price_usd":        Column(float, nullable=False,
                                  checks=Check.greater_than(0)),
        "price_clp":        Column(float, nullable=False,
                                  checks=Check.greater_than(0)),
        "rating":           Column(float, nullable=False,
                                  checks=Check.in_range(0, 5)),
        "loves_count":      Column(float, nullable=False,
                                  checks=Check.greater_than_or_equal_to(0)),
        "primary_category": Column(str,   nullable=True),
    },
    coerce=True,
    strict=False,
    name="clean_products",
)

CLEAN_REVIEWS_SCHEMA = DataFrameSchema(
    columns={
        "author_id":       Column(str,   nullable=False),
        "rating":          Column(float, nullable=False,
                                 checks=Check.in_range(1, 5)),
        "is_recommended":  Column(float, nullable=False,
                                 checks=Check.isin([0.0, 1.0])),
        "review_text":     Column(str,   nullable=True),
        "product_id":      Column(str,   nullable=False),
    },
    coerce=True,
    strict=False,
    name="clean_reviews",
)


# ---------------------------------------------------------------------------
# Funciones de validación con logging
# ---------------------------------------------------------------------------

def validate_raw_products(df: pd.DataFrame, strict: bool = False) -> Tuple[pd.DataFrame, bool]:
    """Valida el DataFrame de productos crudos contra RAW_PRODUCTS_SCHEMA.

    Parameters
    ----------
    df:
        DataFrame a validar.
    strict:
        Si True, relanza la excepción al detectar errores (detiene el pipeline).
        Si False, solo registra el error y devuelve el DataFrame original.

    Returns
    -------
    Tuple[pd.DataFrame, bool]
        (df_validado, es_valido) — df_validado puede tener tipos corregidos.
    """
    logger.info("Validando raw_products (%d filas)...", len(df))
    try:
        df_validated = RAW_PRODUCTS_SCHEMA.validate(df, lazy=True)
        logger.info("✓ raw_products válido.")
        return df_validated, True
    except pa.errors.SchemaErrors as exc:
        logger.warning(
            "raw_products tiene %d errores de validación:\n%s",
            len(exc.failure_cases),
            exc.failure_cases[["schema_context", "column", "check", "failure_case"]].to_string(),
        )
        if strict:
            raise
        return df, False


def validate_raw_reviews(df: pd.DataFrame, strict: bool = False) -> Tuple[pd.DataFrame, bool]:
    """Valida el DataFrame de reseñas crudas contra RAW_REVIEWS_SCHEMA.

    Parameters
    ----------
    df:
        DataFrame a validar.
    strict:
        Si True, relanza la excepción al detectar errores.

    Returns
    -------
    Tuple[pd.DataFrame, bool]
        (df_validado, es_valido).
    """
    logger.info("Validando raw_reviews (%d filas)...", len(df))
    try:
        df_validated = RAW_REVIEWS_SCHEMA.validate(df, lazy=True)
        logger.info("✓ raw_reviews válido.")
        return df_validated, True
    except pa.errors.SchemaErrors as exc:
        logger.warning(
            "raw_reviews tiene %d errores de validación:\n%s",
            len(exc.failure_cases),
            exc.failure_cases[["schema_context", "column", "check", "failure_case"]].head(20).to_string(),
        )
        if strict:
            raise
        return df, False


def validate_clean_products(df: pd.DataFrame, strict: bool = True) -> Tuple[pd.DataFrame, bool]:
    """Valida el DataFrame de productos limpios contra CLEAN_PRODUCTS_SCHEMA.

    Parameters
    ----------
    df:
        DataFrame a validar.
    strict:
        Por defecto True — los datos limpios DEBEN cumplir el contrato.

    Returns
    -------
    Tuple[pd.DataFrame, bool]
        (df_validado, es_valido).
    """
    logger.info("Validando clean_products (%d filas)...", len(df))
    try:
        df_validated = CLEAN_PRODUCTS_SCHEMA.validate(df, lazy=True)
        logger.info("✓ clean_products válido.")
        return df_validated, True
    except pa.errors.SchemaErrors as exc:
        logger.error(
            "clean_products FALLÓ validación con %d errores:\n%s",
            len(exc.failure_cases),
            exc.failure_cases[["schema_context", "column", "check", "failure_case"]].to_string(),
        )
        if strict:
            raise
        return df, False


def validate_clean_reviews(df: pd.DataFrame, strict: bool = True) -> Tuple[pd.DataFrame, bool]:
    """Valida el DataFrame de reseñas limpias contra CLEAN_REVIEWS_SCHEMA.

    Parameters
    ----------
    df:
        DataFrame a validar.
    strict:
        Por defecto True — los datos limpios DEBEN cumplir el contrato.

    Returns
    -------
    Tuple[pd.DataFrame, bool]
        (df_validado, es_valido).
    """
    logger.info("Validando clean_reviews (%d filas)...", len(df))
    try:
        df_validated = CLEAN_REVIEWS_SCHEMA.validate(df, lazy=True)
        logger.info("✓ clean_reviews válido.")
        return df_validated, True
    except pa.errors.SchemaErrors as exc:
        logger.error(
            "clean_reviews FALLÓ validación con %d errores:\n%s",
            len(exc.failure_cases),
            exc.failure_cases[["schema_context", "column", "check", "failure_case"]].to_string(),
        )
        if strict:
            raise
        return df, False
