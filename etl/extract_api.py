"""etl.extract_api
==================
Extracción de sentimiento desde la API de Groq (Fuente 2: API REST).

Toma una muestra de ``review_text`` y consulta un modelo LLM de Groq para
obtener una etiqueta (positive / neutral / negative) y un score de confianza.

Justificación del modelo
-------------------------
- Groq ofrece acceso gratuito a modelos como ``llama3-8b-8192``, optimizados
  para inferencia rápida. Son adecuados para clasificación de sentimiento en
  texto informal como reseñas de cosméticos.
- La API de Groq es gratuita con una cuenta registrada y permite miles de
  llamadas por día con key gratuita.

Justificación de la muestra
-----------------------------
El análisis de sentimiento vía API externa tiene costo O(n) en llamadas de red.
Con ~1 millón de reseñas, enviarlo todo es inviable. Se toma una muestra
estratificada de ``SAMPLE_SIZE`` reseñas (configurable en .env), suficiente
para el análisis estadístico sentimiento↔is_recommended y para la presentación.

Notas de implementación
-----------------------
- Se usa ``requests`` con manejo de errores por reseña individual.
- Los errores se registran y se marca el sentimiento como ``"unknown"``
  sin detener el pipeline.
- El resultado se cachea en disco para no re-consultar la API en cada corrida.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

GROQ_API_URL: str = "https://api.groq.com/openai/v1/chat/completions"
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama3-8b-8192")

# Número de reseñas a analizar (configurable en .env)
SAMPLE_SIZE: int = int(os.getenv("SENTIMENT_SAMPLE_SIZE", "500"))

# Pausa entre llamadas en segundos (evitar rate limiting)
REQUEST_DELAY: float = float(os.getenv("SENTIMENT_REQUEST_DELAY", "0.1"))

# Timeout por llamada
REQUEST_TIMEOUT: int = int(os.getenv("SENTIMENT_TIMEOUT", "30"))

# Ruta del caché local (evita re-consultar en corridas sucesivas)
CACHE_PATH: Path = Path(os.getenv("SENTIMENT_CACHE_PATH", "data/processed/sentiment_cache.csv"))

# Prompt del sistema para clasificación de sentimiento
SYSTEM_PROMPT: str = (
    "You are a sentiment classifier for cosmetic product reviews. "
    "Classify the sentiment of the review as exactly one of: positive, neutral, negative. "
    "Respond with ONLY a JSON object in this format: "
    '{\"label\": \"positive\", \"score\": 0.95} '
    "where score is your confidence between 0.0 and 1.0. "
    "No explanation, no extra text, only the JSON."
)


# ---------------------------------------------------------------------------
# Llamada individual a la API de Groq
# ---------------------------------------------------------------------------

def _get_sentiment_single(text: str) -> dict:
    """Consulta el sentimiento de un texto individual via Groq API.

    Parameters
    ----------
    text:
        Texto de la reseña (se trunca a 500 caracteres).

    Returns
    -------
    dict con claves ``label`` (str) y ``score`` (float).
    Devuelve ``{"label": "unknown", "score": 0.0}`` si la llamada falla.
    """
    text_truncated = str(text)[:500].strip()

    if not text_truncated:
        return {"label": "unknown", "score": 0.0}

    if not GROQ_API_KEY:
        logger.warning("GROQ_API_KEY no configurada en .env.")
        return {"label": "unknown", "score": 0.0}

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Review: {text_truncated}"},
        ],
        "max_tokens": 50,
        "temperature": 0.0,
    }

    try:
        response = requests.post(
            GROQ_API_URL,
            headers=headers,
            json=payload,
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        data = response.json()

        content = data["choices"][0]["message"]["content"].strip()

        # Parsear JSON de respuesta
        import json
        result = json.loads(content)
        label = result.get("label", "unknown").lower()
        if label not in ("positive", "neutral", "negative"):
            label = "unknown"
        score = float(result.get("score", 0.0))
        return {"label": label, "score": round(score, 4)}

    except requests.exceptions.Timeout:
        logger.warning("Timeout al analizar reseña: %s...", text_truncated[:50])
        return {"label": "unknown", "score": 0.0}
    except Exception as exc:
        logger.warning("Error al analizar reseña: %s", exc)
        return {"label": "unknown", "score": 0.0}


# ---------------------------------------------------------------------------
# Extracción principal
# ---------------------------------------------------------------------------

def extract_sentiment(
    df_reviews: pd.DataFrame,
    sample_size: Optional[int] = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Analiza el sentimiento de una muestra de reseñas via Groq API.

    Parameters
    ----------
    df_reviews:
        DataFrame con las reseñas. Debe contener las columnas
        ``review_text``, ``author_id`` e ``is_recommended``.
    sample_size:
        Número de reseñas a analizar. Si es None, usa ``SAMPLE_SIZE``
        del archivo .env (default: 500).
    use_cache:
        Si True, carga resultados previos del caché y solo consulta
        reseñas nuevas. Evita llamadas repetidas a la API.

    Returns
    -------
    pd.DataFrame
        DataFrame con columnas: ``author_id``, ``review_text``,
        ``is_recommended``, ``sentiment``, ``sentiment_score``.
    """
    n = sample_size or SAMPLE_SIZE
    logger.info("=== EXTRACT (API Groq) — inicio ===")
    logger.info("Modelo: %s", GROQ_MODEL)
    logger.info("Muestra objetivo: %d reseñas", n)

    # Validar columnas necesarias
    required_cols = ["review_text", "author_id", "is_recommended"]
    missing = [c for c in required_cols if c not in df_reviews.columns]
    if missing:
        raise ValueError(f"df_reviews no tiene columnas requeridas: {missing}")

    # Filtrar reseñas con texto válido
    df_valid = df_reviews.dropna(subset=["review_text"]).copy()
    df_valid = df_valid[df_valid["review_text"].str.strip().str.len() > 10]

    # Muestra estratificada por is_recommended para representatividad
    try:
        df_valid["is_recommended"] = df_valid["is_recommended"].fillna(0).astype(int)
        df_sample = df_valid.groupby("is_recommended", group_keys=False).apply(
            lambda x: x.sample(min(len(x), n // 2), random_state=42)
        ).reset_index(drop=True)
    except Exception:
        df_sample = df_valid.sample(min(len(df_valid), n), random_state=42).reset_index(drop=True)

    logger.info("Muestra real: %d reseñas (estratificada por is_recommended)", len(df_sample))

    # Cargar caché si existe
    cached_ids: set = set()
    cache_rows: list = []

    if use_cache and CACHE_PATH.exists():
        df_cache = pd.read_csv(CACHE_PATH)
        cached_ids = set(df_cache["author_id"].astype(str))
        cache_rows = df_cache.to_dict("records")
        logger.info("Caché encontrado: %d reseñas ya analizadas.", len(cached_ids))

    # Filtrar reseñas que no están en caché
    df_to_analyze = df_sample[
        ~df_sample["author_id"].astype(str).isin(cached_ids)
    ].reset_index(drop=True)

    logger.info(
        "Reseñas a consultar en API: %d (ya en caché: %d)",
        len(df_to_analyze),
        len(cached_ids),
    )

    # Consultar la API
    new_rows: list = []

    for i, row in df_to_analyze.iterrows():
        result = _get_sentiment_single(row["review_text"])
        new_rows.append({
            "author_id":       row.get("author_id"),
            "review_text":     row.get("review_text"),
            "is_recommended":  row.get("is_recommended", 0),
            "sentiment":       result["label"],
            "sentiment_score": result["score"],
        })

        if (i + 1) % 50 == 0:
            logger.info("  Progreso: %d / %d reseñas analizadas", i + 1, len(df_to_analyze))

        time.sleep(REQUEST_DELAY)

    # Combinar resultados nuevos con caché
    all_rows = cache_rows + new_rows
    df_result = pd.DataFrame(all_rows)

    # Garantizar columnas mínimas aunque todo haya fallado
    expected_cols = ["author_id", "review_text", "is_recommended", "sentiment", "sentiment_score"]
    for col in expected_cols:
        if col not in df_result.columns:
            df_result[col] = None

    # Guardar caché actualizado
    if use_cache and new_rows:
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        df_result.to_csv(CACHE_PATH, index=False)
        logger.info("Caché actualizado: %d entradas totales en %s", len(df_result), CACHE_PATH)

    # Resumen estadístico
    if "sentiment" in df_result.columns and not df_result.empty:
        dist = df_result["sentiment"].value_counts()
        logger.info("Distribución de sentimiento: %s", dist.to_dict())

        if "is_recommended" in df_result.columns:
            cross = df_result.groupby("sentiment")["is_recommended"].mean().round(3)
            logger.info("Tasa de recomendación por sentimiento:\n%s", cross.to_string())

    logger.info(
        "=== EXTRACT (API Groq) — completado: %d reseñas analizadas ===",
        len(df_result),
    )

    return df_result


# ---------------------------------------------------------------------------
# Extracción tipo de cambio USD → CLP (exchangerate-api.com)
# ---------------------------------------------------------------------------

EXCHANGE_API_URL: str = os.getenv(
    "EXCHANGE_API_URL",
    "https://v6.exchangerate-api.com/v6/latest/USD",
)
EXCHANGE_TIMEOUT: int = int(os.getenv("EXCHANGE_API_TIMEOUT", "10"))
FALLBACK_RATE_CLP: float = float(os.getenv("FALLBACK_USD_CLP", "950.0"))


def extract_exchange_rate() -> dict:
    """Consulta la tasa de cambio USD → CLP desde exchangerate-api.com.

    Utiliza la URL configurada en ``EXCHANGE_API_URL`` (incluye la API key
    en la propia URL, según el formato de exchangerate-api.com).

    Si la petición falla por cualquier motivo, devuelve un valor de fallback
    configurado en ``.env`` (``FALLBACK_USD_CLP``) sin detener el pipeline.

    Returns
    -------
    dict con claves:
        - ``date`` (str): fecha del tipo de cambio (YYYY-MM-DD).
        - ``rate`` (float): tasa de cambio USD → CLP.
        - ``source`` (str): ``"api"`` si fue obtenido en línea,
          ``"fallback"`` si se usó el valor por defecto.

    Examples
    --------
    >>> result = extract_exchange_rate()
    >>> print(result)
    {'date': '2025-06-10', 'rate': 948.5, 'source': 'api'}
    """
    from datetime import date as _date

    logger.info("=== EXTRACT (API tipo de cambio) — inicio ===")
    logger.info("URL: %s", EXCHANGE_API_URL)

    try:
        response = requests.get(EXCHANGE_API_URL, timeout=EXCHANGE_TIMEOUT)
        response.raise_for_status()
        data = response.json()

        if data.get("result") == "success":
            rate = float(data["conversion_rates"]["CLP"])
            exchange_date = data.get("time_last_update_utc", str(_date.today()))[:10]
        else:
            raise ValueError(f"Respuesta inesperada de la API: {data}")

        logger.info(
            "=== EXTRACT (API tipo de cambio) — completado: 1 USD = %.2f CLP [fuente: api] ===",
            rate,
        )
        return {"date": exchange_date, "rate": rate, "source": "api"}

    except requests.exceptions.Timeout:
        logger.warning("Timeout al consultar API de tipo de cambio. Usando fallback: %.2f CLP/USD.", FALLBACK_RATE_CLP)
    except requests.exceptions.ConnectionError:
        logger.warning("Sin conexión a API de tipo de cambio. Usando fallback: %.2f CLP/USD.", FALLBACK_RATE_CLP)
    except requests.exceptions.HTTPError as exc:
        logger.warning("Error HTTP en API de tipo de cambio: %s. Usando fallback.", exc)
    except (KeyError, ValueError) as exc:
        logger.warning("Respuesta inesperada de API de tipo de cambio: %s. Usando fallback.", exc)

    from datetime import date as _date
    logger.info(
        "=== EXTRACT (API tipo de cambio) — completado: 1 USD = %.2f CLP [fuente: fallback] ===",
        FALLBACK_RATE_CLP,
    )
    return {"date": str(_date.today()), "rate": FALLBACK_RATE_CLP, "source": "fallback"}