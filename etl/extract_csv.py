"""etl.extract_csv
=================
Extracción de datos desde archivos CSV locales (Fuente 1: CSV).

Lee ``product_info.csv`` y los archivos ``reviews_*.csv`` desde ``data/raw/``,
realiza validaciones básicas de integridad y devuelve DataFrames listos para
la siguiente etapa del pipeline ETL.

Notas de implementación
-----------------------
- Se usa ``glob`` para descubrir automáticamente todos los archivos de reseñas
  sin necesidad de listarlos manualmente.
- Los errores de lectura se registran con ``logging`` en lugar de ``print``.
- Las funciones son puras: reciben rutas y devuelven DataFrames, sin efectos
  secundarios externos.
"""

from __future__ import annotations

import glob
import logging
from pathlib import Path
from typing import List

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constantes de columnas esperadas
# ---------------------------------------------------------------------------

EXPECTED_PRODUCT_COLS: List[str] = [
    "product_id",
    "product_name",
    "brand_name",
    "price_usd",
    "rating",
    "loves_count",
    "primary_category",
]

EXPECTED_REVIEW_COLS: List[str] = [
    "author_id",
    "rating",
    "is_recommended",
    "helpfulness",
    "total_feedback_count",
    "total_neg_feedback_count",
    "total_pos_feedback_count",
    "submission_time",
    "review_text",
    "review_title",
    "skin_tone",
    "eye_color",
    "skin_type",
    "hair_color",
    "product_id",
]


# ---------------------------------------------------------------------------
# Funciones de extracción
# ---------------------------------------------------------------------------


def extract_products(raw_dir: Path) -> pd.DataFrame:
    """Lee ``product_info.csv`` desde el directorio de datos crudos.

    Parameters
    ----------
    raw_dir:
        Ruta al directorio ``data/raw/`` que contiene ``product_info.csv``.

    Returns
    -------
    pd.DataFrame
        DataFrame con los datos de productos sin procesar.

    Raises
    ------
    FileNotFoundError
        Si ``product_info.csv`` no existe en ``raw_dir``.
    ValueError
        Si faltan columnas clave esperadas en el archivo.
    """
    product_path = raw_dir / "product_info.csv"

    if not product_path.exists():
        raise FileNotFoundError(
            f"No se encontró product_info.csv en: {raw_dir}\n"
            "Asegúrate de colocar el dataset en data/raw/ antes de correr el ETL."
        )

    logger.info("Leyendo product_info.csv desde %s", product_path)

    try:
        df = pd.read_csv(product_path, low_memory=False)
    except Exception as exc:
        logger.error("Error al leer product_info.csv: %s", exc)
        raise

    # Validación básica de columnas
    missing_cols = [c for c in EXPECTED_PRODUCT_COLS if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"product_info.csv no contiene las columnas esperadas: {missing_cols}\n"
            f"Columnas encontradas: {df.columns.tolist()}"
        )

    logger.info(
        "product_info.csv cargado: %d filas × %d columnas",
        len(df),
        df.shape[1],
    )
    return df


def extract_reviews(raw_dir: Path) -> pd.DataFrame:
    """Lee y concatena todos los archivos ``reviews_*.csv`` del directorio.

    Descubre los archivos automáticamente usando glob, los carga uno a uno y
    los une en un único DataFrame.

    Parameters
    ----------
    raw_dir:
        Ruta al directorio ``data/raw/`` que contiene los archivos de reseñas.

    Returns
    -------
    pd.DataFrame
        DataFrame concatenado con todas las reseñas, sin procesar.

    Raises
    ------
    FileNotFoundError
        Si no se encuentra ningún archivo ``reviews_*.csv`` en ``raw_dir``.
    ValueError
        Si faltan columnas clave esperadas en los archivos.
    """
    pattern = str(raw_dir / "reviews_*.csv")
    review_files = sorted(glob.glob(pattern))

    if not review_files:
        raise FileNotFoundError(
            f"No se encontraron archivos reviews_*.csv en: {raw_dir}\n"
            "Asegúrate de colocar los archivos de reseñas en data/raw/."
        )

    logger.info("Archivos de reseñas encontrados: %d", len(review_files))

    dfs: List[pd.DataFrame] = []
    for filepath in review_files:
        try:
            df_chunk = pd.read_csv(filepath, low_memory=False)
            dfs.append(df_chunk)
            logger.info(
                "  ✓ %s — %d filas",
                Path(filepath).name,
                len(df_chunk),
            )
        except Exception as exc:
            logger.error("Error al leer %s: %s — se omite este archivo.", filepath, exc)

    if not dfs:
        raise RuntimeError(
            "Todos los archivos de reseñas fallaron al cargarse. "
            "Revisa el formato de los archivos en data/raw/."
        )

    df = pd.concat(dfs, ignore_index=True)

    # Validación básica de columnas
    missing_cols = [c for c in EXPECTED_REVIEW_COLS if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Los archivos de reseñas no contienen columnas esperadas: {missing_cols}\n"
            f"Columnas encontradas: {df.columns.tolist()}"
        )

    logger.info(
        "Reseñas concatenadas: %d filas × %d columnas (de %d archivos)",
        len(df),
        df.shape[1],
        len(dfs),
    )
    return df


def extract_all_csv(raw_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Extrae productos y reseñas en una sola llamada.

    Parameters
    ----------
    raw_dir:
        Ruta al directorio ``data/raw/``.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (df_products, df_reviews) listos para la etapa de transformación.
    """
    logger.info("=== EXTRACT (CSV) — inicio ===")
    df_products = extract_products(raw_dir)
    df_reviews = extract_reviews(raw_dir)
    logger.info(
        "=== EXTRACT (CSV) — completado: %d productos, %d reseñas ===",
        len(df_products),
        len(df_reviews),
    )
    return df_products, df_reviews
