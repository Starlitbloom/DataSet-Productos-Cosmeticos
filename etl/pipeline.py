"""etl.pipeline
===============
Orquestador del pipeline ETL de Sephora end-to-end.

Ejecuta en orden: Extract → Validate → Transform → Load
con logging de inicio/fin y manejo de excepciones por etapa.

Uso desde terminal
------------------
    python -m etl.pipeline

Uso desde código
----------------
    from etl.pipeline import run_pipeline
    run_pipeline()

Uso en modo desarrollo (con muestra)
--------------------------------------
    from etl.pipeline import run_pipeline
    run_pipeline(sample_size=1000, sentiment_sample=50)
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Configuración de logging
# ---------------------------------------------------------------------------

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)

logger = logging.getLogger("etl.pipeline")

# ---------------------------------------------------------------------------
# Rutas del proyecto
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_RAW_DIR    = BASE_DIR / "data" / "raw"
MODELS_DIR      = BASE_DIR / "models" / "trained_models"
METRICS_DIR     = BASE_DIR / "results" / "metrics"


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

def run_pipeline(
    sample_size: Optional[int] = None,
    sentiment_sample: Optional[int] = None,
    use_sentiment_cache: bool = True,
) -> dict:
    """Ejecuta el pipeline ETL completo: Extract → Validate → Transform → Load.

    Parameters
    ----------
    sample_size:
        Si se especifica, limita la lectura de reseñas a N filas.
        Útil para desarrollo y testing. En producción debe ser ``None``.
    sentiment_sample:
        Número de reseñas a enviar a HuggingFace API. Si es None,
        usa el valor de ``SENTIMENT_SAMPLE_SIZE`` en ``.env`` (default 500).
    use_sentiment_cache:
        Si True, usa el caché local para no re-consultar la API.

    Returns
    -------
    dict
        Resumen de la ejecución con tiempos y conteos por etapa.
    """
    pipeline_start = time.time()
    logger.info("=" * 60)
    logger.info("PIPELINE ETL — INICIO")
    logger.info("=" * 60)

    summary = {
        "status": "running",
        "stages": {},
        "errors": [],
    }

    # ── Imports aquí para evitar circular imports ──────────────────────────
    from etl.extract_csv import extract_all_csv
    from etl.extract_api import extract_sentiment, extract_exchange_rate
    from etl.extract_db  import get_engine, load_raw_csvs_to_db, extract_raw_from_db
    from etl.schemas     import (validate_raw_products, validate_raw_reviews,
                                  validate_clean_products, validate_clean_reviews)
    from etl.transform   import transform_all
    from etl.load        import load_all

    engine = get_engine()

    # ──────────────────────────────────────────────────────────────────────
    # ETAPA 1: EXTRACT
    # ──────────────────────────────────────────────────────────────────────
    logger.info("── ETAPA 1: EXTRACT ─────────────────────────────────────")
    t0 = time.time()

    try:
        # 1a. CSV → DataFrames crudos
        df_products_raw, df_reviews_raw = extract_all_csv(DATA_RAW_DIR)

        # 1b. Carga inicial a PostgreSQL (si las tablas no existen aún)
        #     En corridas sucesivas, los datos ya están en la DB.
        try:
            from sqlalchemy import text
            with engine.connect() as conn:
                result = conn.execute(
                    text("SELECT EXISTS (SELECT 1 FROM information_schema.tables "
                         "WHERE table_name = 'raw_products')")
                )
                tables_exist = result.scalar()

            if not tables_exist:
                logger.info("Carga inicial: subiendo CSVs a PostgreSQL por primera vez...")
                load_raw_csvs_to_db(df_products_raw, df_reviews_raw, engine)
            else:
                logger.info("Tablas raw ya existen en PostgreSQL — omitiendo carga inicial.")

        except Exception as exc:
            logger.warning("No se pudo verificar tablas raw: %s — usando datos CSV directamente.", exc)

        # 1c. API HuggingFace — sentimiento
        df_sentiment_raw = extract_sentiment(
            df_reviews_raw,
            sample_size=sentiment_sample,
            use_cache=use_sentiment_cache,
        )

        # 1d. API tipo de cambio USD→CLP
        exchange_rate_dict = extract_exchange_rate()
        exchange_rate = exchange_rate_dict["rate"]

        elapsed = time.time() - t0
        summary["stages"]["extract"] = {
            "status": "ok",
            "products": len(df_products_raw),
            "reviews": len(df_reviews_raw),
            "sentiment": len(df_sentiment_raw),
            "exchange_rate": exchange_rate,
            "elapsed_s": round(elapsed, 1),
        }
        logger.info("✓ EXTRACT completado en %.1fs", elapsed)

    except Exception as exc:
        logger.error("✗ EXTRACT falló: %s", exc)
        summary["stages"]["extract"] = {"status": "error", "error": str(exc)}
        summary["errors"].append(f"EXTRACT: {exc}")
        summary["status"] = "failed"
        return summary

    # ──────────────────────────────────────────────────────────────────────
    # ETAPA 2: VALIDATE (datos crudos)
    # ──────────────────────────────────────────────────────────────────────
    logger.info("── ETAPA 2: VALIDATE (raw) ──────────────────────────────")
    t0 = time.time()

    try:
        df_products_raw, valid_p = validate_raw_products(df_products_raw, strict=False)
        df_reviews_raw,  valid_r = validate_raw_reviews(df_reviews_raw,   strict=False)

        elapsed = time.time() - t0
        summary["stages"]["validate_raw"] = {
            "status": "ok",
            "products_valid": valid_p,
            "reviews_valid": valid_r,
            "elapsed_s": round(elapsed, 1),
        }
        logger.info("✓ VALIDATE (raw) completado en %.1fs", elapsed)

    except Exception as exc:
        logger.error("✗ VALIDATE (raw) falló: %s", exc)
        summary["stages"]["validate_raw"] = {"status": "error", "error": str(exc)}
        summary["errors"].append(f"VALIDATE_RAW: {exc}")
        # No detenemos el pipeline — continuamos con los datos sin validar

    # ──────────────────────────────────────────────────────────────────────
    # ETAPA 3: TRANSFORM
    # ──────────────────────────────────────────────────────────────────────
    logger.info("── ETAPA 3: TRANSFORM ───────────────────────────────────")
    t0 = time.time()

    try:
        clean_products, clean_reviews, clean_sentiment = transform_all(
            df_products_raw,
            df_reviews_raw,
            df_sentiment_raw,
            exchange_rate=exchange_rate,
        )

        elapsed = time.time() - t0
        summary["stages"]["transform"] = {
            "status": "ok",
            "clean_products": len(clean_products),
            "clean_reviews": len(clean_reviews),
            "clean_sentiment": len(clean_sentiment),
            "elapsed_s": round(elapsed, 1),
        }
        logger.info("✓ TRANSFORM completado en %.1fs", elapsed)

    except Exception as exc:
        logger.error("✗ TRANSFORM falló: %s", exc)
        summary["stages"]["transform"] = {"status": "error", "error": str(exc)}
        summary["errors"].append(f"TRANSFORM: {exc}")
        summary["status"] = "failed"
        return summary

    # ──────────────────────────────────────────────────────────────────────
    # ETAPA 4: VALIDATE (datos limpios)
    # ──────────────────────────────────────────────────────────────────────
    logger.info("── ETAPA 4: VALIDATE (clean) ────────────────────────────")
    t0 = time.time()

    try:
        clean_products, valid_p = validate_clean_products(clean_products, strict=False)
        clean_reviews,  valid_r = validate_clean_reviews(clean_reviews,   strict=False)

        elapsed = time.time() - t0
        summary["stages"]["validate_clean"] = {
            "status": "ok",
            "products_valid": valid_p,
            "reviews_valid": valid_r,
            "elapsed_s": round(elapsed, 1),
        }
        logger.info("✓ VALIDATE (clean) completado en %.1fs", elapsed)

    except Exception as exc:
        logger.warning("VALIDATE (clean) con errores: %s — continuando de todas formas.", exc)
        summary["stages"]["validate_clean"] = {"status": "warning", "error": str(exc)}

    # ──────────────────────────────────────────────────────────────────────
    # ETAPA 5: Cargar métricas de EP2 (opcional)
    # ──────────────────────────────────────────────────────────────────────
    model_metrics = None
    metrics_path = METRICS_DIR / "hyperparameter_optimization_report.json"
    if metrics_path.exists():
        try:
            with open(metrics_path, encoding="utf-8") as f:
                report = json.load(f)
            model_metrics = {
                "model_name":    report.get("best_model", "unknown"),
                "f1_test":       report.get("best_f1_test"),
                "f1_cv_grid":    report.get("grid_search", {}).get("f1_cv"),
                "f1_cv_random":  report.get("random_search", {}).get("f1_cv"),
            }
            logger.info("Métricas de EP2 cargadas desde %s", metrics_path)
        except Exception as exc:
            logger.warning("No se pudieron leer métricas de EP2: %s", exc)

    # ──────────────────────────────────────────────────────────────────────
    # ETAPA 6: LOAD
    # ──────────────────────────────────────────────────────────────────────
    logger.info("── ETAPA 6: LOAD ────────────────────────────────────────")
    t0 = time.time()

    try:
        load_all(
            clean_products=clean_products,
            clean_reviews=clean_reviews,
            clean_sentiment=clean_sentiment,
            exchange_rate=exchange_rate_dict,
            engine=engine,
            model_metrics=model_metrics,
        )

        elapsed = time.time() - t0
        summary["stages"]["load"] = {
            "status": "ok",
            "elapsed_s": round(elapsed, 1),
        }
        logger.info("✓ LOAD completado en %.1fs", elapsed)

    except Exception as exc:
        logger.error("✗ LOAD falló: %s", exc)
        summary["stages"]["load"] = {"status": "error", "error": str(exc)}
        summary["errors"].append(f"LOAD: {exc}")
        summary["status"] = "failed"
        return summary

    # ──────────────────────────────────────────────────────────────────────
    # RESUMEN FINAL
    # ──────────────────────────────────────────────────────────────────────
    total_elapsed = time.time() - pipeline_start
    summary["status"] = "success"
    summary["total_elapsed_s"] = round(total_elapsed, 1)

    logger.info("=" * 60)
    logger.info("PIPELINE ETL — COMPLETADO en %.1fs", total_elapsed)
    logger.info("Productos limpios: %d", len(clean_products))
    logger.info("Reseñas limpias:   %d", len(clean_reviews))
    logger.info("Sentimiento:       %d", len(clean_sentiment))
    logger.info("Tasa USD→CLP:      %.2f [%s]", exchange_rate, exchange_rate_dict["source"])
    logger.info("=" * 60)

    return summary


# ---------------------------------------------------------------------------
# Punto de entrada
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Pipeline ETL Sephora")
    parser.add_argument("--sample",    type=int, default=None,
                        help="Limitar reseñas a N filas (solo para desarrollo)")
    parser.add_argument("--sentiment", type=int, default=None,
                        help="Número de reseñas a analizar con HuggingFace")
    parser.add_argument("--no-cache",  action="store_true",
                        help="Ignorar caché de sentimiento y re-consultar la API")
    args = parser.parse_args()

    result = run_pipeline(
        sample_size=args.sample,
        sentiment_sample=args.sentiment,
        use_sentiment_cache=not args.no_cache,
    )

    if result["status"] != "success":
        logger.error("Pipeline terminó con errores: %s", result["errors"])
        sys.exit(1)
