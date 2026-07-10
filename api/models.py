"""api.models
=============
Modelos Pydantic para los response_model de cada endpoint.

Cada modelo refleja exactamente las columnas de las tablas en PostgreSQL:
- ``Product``             → tabla ``clean_products``
- ``Review``              → tabla ``clean_reviews``
- ``SentimentResult``     → tabla ``sentiment_results``
- ``ExchangeRate``        → tabla ``exchange_rates``
- ``ModelMetric``         → tabla ``model_metrics``
- ``SentimentSummaryResponse`` → respuesta calculada de sentimiento
"""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Product — tabla clean_products
# ---------------------------------------------------------------------------

class Product(BaseModel):
    """Esquema de respuesta para un producto de Sephora.

    Incluye precio en USD y en CLP (convertido por el ETL usando el
    tipo de cambio de la API exchangerate-api.com).
    """
    product_id:         str
    product_name:       Optional[str]   = None
    brand_id:           Optional[int]   = None
    brand_name:         Optional[str]   = None
    loves_count:        Optional[float] = None
    rating:             Optional[float] = None
    reviews:            Optional[float] = None
    size:               Optional[str]   = None
    variation_type:     Optional[str]   = None
    variation_value:    Optional[str]   = None
    variation_desc:     Optional[str]   = None
    price_usd:          Optional[float] = Field(None, description="Precio en dólares (USD)")
    value_price_usd:    Optional[float] = None
    sale_price_usd:     Optional[float] = None
    price_clp:          Optional[float] = Field(None, description="Precio en pesos chilenos (CLP)")
    limited_edition:    Optional[int]   = None
    new:                Optional[int]   = None
    online_only:        Optional[int]   = None
    out_of_stock:       Optional[int]   = None
    sephora_exclusive:  Optional[int]   = None
    highlights:         Optional[str]   = None
    primary_category:   Optional[str]   = None
    secondary_category: Optional[str]   = None
    tertiary_category:  Optional[str]   = None
    child_count:        Optional[int]   = None
    child_max_price:    Optional[float] = None
    child_min_price:    Optional[float] = None

    model_config = {"from_attributes": True}


# ---------------------------------------------------------------------------
# Review — tabla clean_reviews
# ---------------------------------------------------------------------------

class Review(BaseModel):
    """Esquema de respuesta para una reseña de producto."""
    author_id:                  Optional[str]   = None
    product_id:                 Optional[str]   = None
    product_name:               Optional[str]   = None
    brand_name:                 Optional[str]   = None
    rating:                     Optional[float] = Field(None, ge=1.0, le=5.0)
    is_recommended:             Optional[float] = Field(None, description="1.0 = recomendado, 0.0 = no recomendado")
    helpfulness:                Optional[float] = None
    total_feedback_count:       Optional[float] = None
    total_neg_feedback_count:   Optional[float] = None
    total_pos_feedback_count:   Optional[float] = None
    submission_time:            Optional[str]   = None
    review_text:                Optional[str]   = None
    review_title:               Optional[str]   = None
    skin_tone:                  Optional[str]   = None
    eye_color:                  Optional[str]   = None
    skin_type:                  Optional[str]   = None
    hair_color:                 Optional[str]   = None
    price_usd:                  Optional[float] = None

    model_config = {"from_attributes": True}


# ---------------------------------------------------------------------------
# SentimentResult — tabla sentiment_results
# ---------------------------------------------------------------------------

class SentimentResult(BaseModel):
    """Esquema de respuesta para un resultado de análisis de sentimiento."""
    author_id:       Optional[str]   = None
    review_text:     Optional[str]   = None
    is_recommended:  Optional[int]   = None
    sentiment:       Optional[str]   = None
    sentiment_score: Optional[float] = None

    model_config = {"from_attributes": True}


# ---------------------------------------------------------------------------
# ExchangeRate — tabla exchange_rates
# ---------------------------------------------------------------------------

class ExchangeRate(BaseModel):
    """Esquema de respuesta para el tipo de cambio USD → CLP."""
    date:      Optional[str]   = Field(None, description="Fecha del tipo de cambio (YYYY-MM-DD)")
    rate:      Optional[float] = Field(None, description="Pesos chilenos por 1 USD")
    source:    Optional[str]   = Field(None, description="'api' si vino de exchangerate-api.com, 'fallback' si se usó el valor del .env")
    loaded_at: Optional[str]   = Field(None, description="Timestamp de carga al data warehouse")

    model_config = {"from_attributes": True}


# ---------------------------------------------------------------------------
# ModelMetric — tabla model_metrics
# ---------------------------------------------------------------------------

class ModelMetric(BaseModel):
    """Esquema de respuesta para las métricas del modelo de ML.

    Combina:
    - Parámetros del GradientBoostingClassifier entrenado en EP2.
    - Métricas de clasificación supervisada (accuracy, F1, ROC-AUC, etc.).
    - Métricas del análisis no supervisado K-Means (silhouette, inercia, k óptimo).
    - Varianza explicada por PCA 2D.

    Esta tabla es cargada por el ETL usando los archivos generados en EP2
    (``results/metrics/model_metrics.json``).
    """
    # Identificación del modelo
    model_name:        Optional[str]   = Field(None, description="Nombre del modelo (ej: GradientBoostingClassifier)")

    # Hiperparámetros del modelo
    n_estimators:      Optional[int]   = Field(None, description="Número de estimadores")
    max_depth:         Optional[int]   = Field(None, description="Profundidad máxima de los árboles")
    learning_rate:     Optional[float] = Field(None, description="Tasa de aprendizaje")
    random_state:      Optional[int]   = Field(None, description="Semilla de aleatoriedad")

    # Métricas de clasificación supervisada (EP2)
    accuracy:          Optional[float] = Field(None, description="Accuracy en el test set")
    f1:                Optional[float] = Field(None, description="F1-Score ponderado en el test set")
    precision:         Optional[float] = Field(None, description="Precision ponderada en el test set")
    recall:            Optional[float] = Field(None, description="Recall ponderado en el test set")
    roc_auc:           Optional[float] = Field(None, description="ROC-AUC en el test set")

    # Métricas de clustering K-Means (EP2)
    silhouette_score:  Optional[float] = Field(None, description="Coeficiente de silueta del K-Means óptimo")
    best_k:            Optional[int]   = Field(None, description="Número óptimo de clusters K-Means")
    final_inertia:     Optional[float] = Field(None, description="Inercia del K-Means con el k óptimo")

    # PCA
    pca_variance_2d:   Optional[float] = Field(None, description="Varianza explicada por las 2 primeras componentes PCA")

    model_config = {"from_attributes": True}


# ---------------------------------------------------------------------------
# SentimentSummaryResponse — respuesta calculada
# ---------------------------------------------------------------------------

class SentimentSummaryResponse(BaseModel):
    """Resumen del análisis de sentimiento vs recomendación."""
    sentiment:           str
    total_reviews:       int
    pct_recommended:     float
    avg_sentiment_score: float

    model_config = {"from_attributes": True}