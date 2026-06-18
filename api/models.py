"""api.models
=============
Modelos Pydantic para los response_model de cada endpoint.

Cada modelo refleja exactamente las columnas de las tablas en PostgreSQL:
- ``Product``        → tabla ``clean_products``
- ``Review``         → tabla ``clean_reviews``
- ``SentimentResult``→ tabla ``sentiment_results``
- ``ExchangeRate``   → tabla ``exchange_rates``
- ``ModelMetric``    → tabla ``model_metrics`` (puede no existir aún)
"""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Product — tabla clean_products
# ---------------------------------------------------------------------------

class Product(BaseModel):
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
    price_usd:          Optional[float] = None
    value_price_usd:    Optional[float] = None
    sale_price_usd:     Optional[float] = None
    price_clp:          Optional[float] = None
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
    author_id:                  Optional[str]   = None
    product_id:                 Optional[str]   = None
    product_name:               Optional[str]   = None
    brand_name:                 Optional[str]   = None
    rating:                     Optional[float] = None
    is_recommended:             Optional[float] = None
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
    date:      Optional[str]   = None
    rate:      Optional[float] = None
    source:    Optional[str]   = None
    loaded_at: Optional[str]   = None

    model_config = {"from_attributes": True}


# ---------------------------------------------------------------------------
# ModelMetric — tabla model_metrics (puede no existir)
# ---------------------------------------------------------------------------

class ModelMetric(BaseModel):
    model_name:        Optional[str]   = None
    n_estimators:      Optional[int]   = None
    max_depth:         Optional[int]   = None
    learning_rate:     Optional[float] = None
    random_state:      Optional[int]   = None
    silhouette_score:  Optional[float] = None
    best_k:            Optional[int]   = None
    final_inertia:     Optional[float] = None
    pca_variance_2d:   Optional[float] = None

    model_config = {"from_attributes": True}
