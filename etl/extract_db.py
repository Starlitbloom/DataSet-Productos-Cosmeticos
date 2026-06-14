"""etl.extract_db
=================
Extracción de datos desde PostgreSQL (Fuente 3: base de datos SQL).

Este módulo cumple dos roles:

1. **Carga inicial** (``load_raw_csvs_to_db``): se ejecuta UNA VEZ para subir
   los CSVs crudos de Sephora a PostgreSQL como tablas ``raw_products`` y
   ``raw_reviews``. Esto convierte la base de datos en la "fuente SQL" del
   pipeline, simulando un escenario de producción donde los datos ya viven
   en un sistema transaccional.

2. **Extracción normal** (``extract_raw_from_db``): se ejecuta en cada corrida
   del ETL para leer las tablas crudas desde la base de datos.

La conexión usa SQLAlchemy con credenciales cargadas desde ``.env`` para
evitar hardcodear contraseñas en el código.

Notas de implementación
-----------------------
- Se usa ``create_engine`` de SQLAlchemy con ``pool_pre_ping=True`` para
  detectar conexiones muertas automáticamente.
- Los errores de conexión se capturan con ``try/except`` y se registran.
- ``pd.read_sql`` es más seguro que ``pd.read_sql_query`` para consultas
  simples; usar ``text()`` de SQLAlchemy para consultas parametrizadas.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Optional

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import OperationalError, SQLAlchemyError

load_dotenv()

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Conexión a la base de datos
# ---------------------------------------------------------------------------


def get_engine() -> Engine:
    """Crea y devuelve un engine de SQLAlchemy conectado a PostgreSQL.

    Lee las credenciales desde variables de entorno (archivo ``.env``).

    Returns
    -------
    sqlalchemy.engine.Engine
        Engine listo para usar con pandas o SQLAlchemy ORM.

    Raises
    ------
    EnvironmentError
        Si alguna variable de entorno requerida no está definida.
    OperationalError
        Si no se puede establecer la conexión con la base de datos.
    """
    required_vars = [
        "POSTGRES_USER",
        "POSTGRES_PASSWORD",
        "POSTGRES_HOST",
        "POSTGRES_PORT",
        "POSTGRES_DB",
    ]
    missing = [v for v in required_vars if not os.getenv(v)]
    if missing:
        raise EnvironmentError(
            f"Variables de entorno faltantes: {missing}\n"
            "Copia .env.example a .env y completa las credenciales."
        )

    user     = os.getenv("POSTGRES_USER")
    password = os.getenv("POSTGRES_PASSWORD")
    host     = os.getenv("POSTGRES_HOST", "localhost")
    port     = os.getenv("POSTGRES_PORT", "5432")
    db       = os.getenv("POSTGRES_DB")

    url = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}"

    engine = create_engine(
        url,
        pool_pre_ping=True,    # Verifica que la conexión está viva antes de usarla
        pool_size=5,
        max_overflow=10,
        connect_args={"connect_timeout": 10},
    )

    logger.info("Engine de base de datos creado: %s:%s/%s", host, port, db)
    return engine


# ---------------------------------------------------------------------------
# Carga inicial: CSVs → PostgreSQL (se ejecuta solo una vez)
# ---------------------------------------------------------------------------


def load_raw_csvs_to_db(
    df_products: pd.DataFrame,
    df_reviews: pd.DataFrame,
    engine: Optional[Engine] = None,
    if_exists: str = "replace",
) -> None:
    """Sube los DataFrames crudos a PostgreSQL como tablas ``raw_products``
    y ``raw_reviews``.

    Esta función se usa para la **carga inicial** del pipeline: convierte los
    archivos CSV en una fuente de datos SQL, simulando un entorno de producción
    donde los datos ya existen en una base de datos transaccional.

    Parameters
    ----------
    df_products:
        DataFrame con los datos crudos de productos (de ``extract_csv.py``).
    df_reviews:
        DataFrame con los datos crudos de reseñas (de ``extract_csv.py``).
    engine:
        Engine de SQLAlchemy. Si no se proporciona, se crea uno nuevo.
    if_exists:
        Comportamiento si la tabla ya existe: ``"replace"`` (por defecto)
        la sobreescribe, ``"append"`` agrega filas, ``"fail"`` lanza error.

    Notes
    -----
    - Se usan chunksizes para manejar eficientemente datasets grandes.
    - La carga inicial puede tomar varios minutos con millones de reseñas.
    """
    if engine is None:
        engine = get_engine()

    logger.info(
        "Cargando datos crudos a PostgreSQL (if_exists='%s')...", if_exists
    )

    try:
        # Subir productos
        logger.info(
            "  → raw_products: %d filas × %d columnas", len(df_products), df_products.shape[1]
        )
        df_products.to_sql(
            "raw_products",
            engine,
            if_exists=if_exists,
            index=False,
            chunksize=10_000,
            method="multi",
        )
        logger.info("  ✓ raw_products cargada.")

        # Subir reseñas (dataset más grande)
        logger.info(
            "  → raw_reviews: %d filas × %d columnas", len(df_reviews), df_reviews.shape[1]
        )
        df_reviews.to_sql(
            "raw_reviews",
            engine,
            if_exists=if_exists,
            index=False,
            chunksize=50_000,
            method="multi",
        )
        logger.info("  ✓ raw_reviews cargada.")

    except SQLAlchemyError as exc:
        logger.error("Error al cargar datos a PostgreSQL: %s", exc)
        raise

    logger.info("Carga inicial completada.")


# ---------------------------------------------------------------------------
# Extracción normal: PostgreSQL → DataFrames
# ---------------------------------------------------------------------------


def extract_raw_from_db(
    engine: Optional[Engine] = None,
    sample_size: Optional[int] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Lee las tablas ``raw_products`` y ``raw_reviews`` desde PostgreSQL.

    Esta es la función de extracción normal del pipeline ETL: se ejecuta en
    cada corrida para obtener los datos crudos desde la base de datos.

    Parameters
    ----------
    engine:
        Engine de SQLAlchemy. Si no se proporciona, se crea uno nuevo.
    sample_size:
        Si se especifica, se limita la lectura a ``sample_size`` filas por
        tabla. Útil para desarrollo y pruebas. En producción debe ser ``None``.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (df_products, df_reviews) con los datos crudos.

    Raises
    ------
    OperationalError
        Si no se puede conectar a la base de datos.
    """
    if engine is None:
        engine = get_engine()

    logger.info("=== EXTRACT (DB) — inicio ===")

    try:
        # Verificar que las tablas existen
        with engine.connect() as conn:
            for table in ["raw_products", "raw_reviews"]:
                result = conn.execute(
                    text(
                        "SELECT EXISTS (SELECT 1 FROM information_schema.tables "
                        "WHERE table_name = :t)"
                    ),
                    {"t": table},
                )
                exists = result.scalar()
                if not exists:
                    raise RuntimeError(
                        f"La tabla '{table}' no existe en la base de datos.\n"
                        "Ejecuta load_raw_csvs_to_db() primero para la carga inicial."
                    )

        # Construir query con limit opcional para desarrollo
        limit_clause = f"LIMIT {sample_size}" if sample_size else ""

        query_products = f"SELECT * FROM raw_products {limit_clause}"
        query_reviews  = f"SELECT * FROM raw_reviews {limit_clause}"

        logger.info("Leyendo raw_products desde PostgreSQL...")
        df_products = pd.read_sql(query_products, engine)
        logger.info("  ✓ raw_products: %d filas", len(df_products))

        logger.info("Leyendo raw_reviews desde PostgreSQL...")
        df_reviews = pd.read_sql(query_reviews, engine)
        logger.info("  ✓ raw_reviews: %d filas", len(df_reviews))

    except OperationalError as exc:
        logger.error(
            "No se pudo conectar a PostgreSQL: %s\n"
            "¿Está corriendo el contenedor Docker? Ejecuta: docker compose up -d db",
            exc,
        )
        raise

    logger.info(
        "=== EXTRACT (DB) — completado: %d productos, %d reseñas ===",
        len(df_products),
        len(df_reviews),
    )
    return df_products, df_reviews


def extract_exchange_rates_from_db(
    engine: Optional[Engine] = None,
) -> pd.DataFrame:
    """Lee el historial de tipos de cambio desde la tabla ``exchange_rates``.

    Parameters
    ----------
    engine:
        Engine de SQLAlchemy. Si no se proporciona, se crea uno nuevo.

    Returns
    -------
    pd.DataFrame
        DataFrame con columnas ``date`` y ``rate``, ordenado por fecha.
    """
    if engine is None:
        engine = get_engine()

    try:
        df = pd.read_sql(
            "SELECT * FROM exchange_rates ORDER BY date DESC",
            engine,
        )
        logger.info("exchange_rates leída: %d registros", len(df))
        return df
    except Exception as exc:
        logger.warning(
            "No se pudo leer exchange_rates: %s. "
            "La tabla puede no existir todavía.",
            exc,
        )
        return pd.DataFrame(columns=["date", "rate", "source"])
