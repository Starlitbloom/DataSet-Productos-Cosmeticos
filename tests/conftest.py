"""tests.conftest
=================
Fixtures de pytest compartidas para todos los tests del pipeline ETL.

Proporciona DataFrames sintéticos pequeños que simulan los datos reales
de Sephora sin depender de archivos CSV ni conexión a la base de datos.
"""

from __future__ import annotations

import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Fixtures: datos crudos (con nulos y valores problemáticos intencionales)
# ---------------------------------------------------------------------------

@pytest.fixture
def raw_products_df() -> pd.DataFrame:
    """DataFrame sintético de productos crudos con casos problemáticos."""
    return pd.DataFrame({
        "product_id":       ["P001", "P002", "P003", "P004", "P001"],  # P001 duplicado
        "product_name":     ["Lipstick A", "Foundation B", None, "Mascara D", "Lipstick A"],
        "brand_name":       ["Brand X", "Brand Y", "Brand Z", None, "Brand X"],
        "price_usd":        [25.0, 45.0, None, -5.0, 25.0],   # None y negativo
        "rating":           [4.5, 3.2, 5.0, 2.1, 4.5],
        "loves_count":      [1000.0, 5000.0, None, 200.0, 1000.0],
        "primary_category": ["Makeup", "Skincare", "Makeup", None, "Makeup"],
    })


@pytest.fixture
def raw_reviews_df() -> pd.DataFrame:
    """DataFrame sintético de reseñas crudas con casos problemáticos."""
    return pd.DataFrame({
        "author_id":                ["U001", "U002", "U003", None, "U001"],  # None y duplicado
        "rating":                   [5.0, 3.0, 1.0, 4.0, 5.0],
        "is_recommended":           [1.0, 0.0, 0.0, 1.0, 1.0],
        "helpfulness":              [0.9, 0.5, 0.1, None, 0.9],
        "total_feedback_count":     [10, 5, 3, 0, 10],
        "total_neg_feedback_count": [1, 2, 3, 0, 1],
        "total_pos_feedback_count": [9, 3, 0, 0, 9],
        "submission_time":          ["2024-01-01"] * 5,
        "review_text":              [
            "Love this product!",
            "Not great, disappointed.",
            "Terrible quality.",
            "Pretty good overall.",
            "Love this product!",
        ],
        "review_title":             ["Great!", "Meh", "Bad", "Good", "Great!"],
        "skin_tone":                ["fair", "medium", "dark", None, "fair"],
        "eye_color":                ["blue", "brown", "green", "hazel", "blue"],
        "skin_type":                ["dry", "oily", "combination", None, "dry"],
        "hair_color":               ["blonde", "black", "brown", "red", "blonde"],
        "product_id":               ["P001", "P002", "P003", "P004", None],  # None en último
    })


@pytest.fixture
def raw_sentiment_df() -> pd.DataFrame:
    """DataFrame sintético de resultados de sentimiento."""
    return pd.DataFrame({
        "author_id":       ["U001", "U002", "U003"],
        "review_text":     ["Love it!", "Not great", "Terrible"],
        "is_recommended":  [1.0, 0.0, 0.0],
        "sentiment":       ["positive", "negative", "unknown"],  # 'unknown' debe filtrarse
        "sentiment_score": [0.95, 0.82, 0.0],
    })


@pytest.fixture
def exchange_rate_dict() -> dict:
    """Diccionario de tipo de cambio sintético."""
    return {
        "date":   "2025-06-01",
        "rate":   950.0,
        "source": "api",
    }


# ---------------------------------------------------------------------------
# Fixtures: datos limpios esperados
# ---------------------------------------------------------------------------

@pytest.fixture
def clean_products_df() -> pd.DataFrame:
    """DataFrame de productos ya limpios (sin duplicados, sin nulos clave)."""
    return pd.DataFrame({
        "product_id":       ["P001", "P002", "P004"],
        "product_name":     ["Lipstick A", "Foundation B", "Mascara D"],
        "brand_name":       ["Brand X", "Brand Y", "Brand Z"],
        "price_usd":        [25.0, 45.0, 5.0],
        "price_clp":        [23750.0, 42750.0, 4750.0],
        "rating":           [4.5, 3.2, 2.1],
        "loves_count":      [1000.0, 5000.0, 200.0],
        "primary_category": ["Makeup", "Skincare", None],
    })


@pytest.fixture
def clean_reviews_df() -> pd.DataFrame:
    """DataFrame de reseñas ya limpias (sin duplicados, sin nulos clave)."""
    return pd.DataFrame({
        "author_id":       ["U001", "U002", "U003"],
        "rating":          [5.0, 3.0, 1.0],
        "is_recommended":  [1.0, 0.0, 0.0],
        "review_text":     ["Love this product!", "Not great, disappointed.", "Terrible quality."],
        "product_id":      ["P001", "P002", "P003"],
    })
