"""tests.conftest
=================
Fixtures de pytest compartidas para todos los tests del pipeline ETL y la API.

Proporciona DataFrames sintéticos pequeños que simulan los datos reales
de Sephora sin depender de archivos CSV ni conexión a la base de datos.
"""

from __future__ import annotations

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from api.main import app
from api.database import get_db


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


# ---------------------------------------------------------------------------
# Fixtures: base de datos en memoria para tests de API
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def test_engine():
    """Motor SQLite en memoria con tablas y datos sintéticos para tests de API."""
    engine = create_engine("sqlite:///file:testdb?mode=memory&cache=shared&uri=true", connect_args={"check_same_thread": False})

    with engine.begin() as conn:
        # clean_products
        conn.execute(text("""
            CREATE TABLE clean_products (
                product_id TEXT PRIMARY KEY,
                product_name TEXT,
                brand_id INTEGER,
                brand_name TEXT,
                loves_count REAL,
                rating REAL,
                reviews REAL,
                size TEXT,
                variation_type TEXT,
                variation_value TEXT,
                variation_desc TEXT,
                price_usd REAL,
                value_price_usd REAL,
                sale_price_usd REAL,
                price_clp REAL,
                limited_edition INTEGER,
                new INTEGER,
                online_only INTEGER,
                out_of_stock INTEGER,
                sephora_exclusive INTEGER,
                highlights TEXT,
                primary_category TEXT,
                secondary_category TEXT,
                tertiary_category TEXT,
                child_count INTEGER,
                child_max_price REAL,
                child_min_price REAL
            )
        """))
        conn.execute(text("""
            INSERT INTO clean_products
                (product_id, product_name, brand_name, price_usd, price_clp, rating, primary_category)
            VALUES
                ('P001', 'Lipstick A', 'Brand X', 25.0, 23750.0, 4.5, 'Makeup'),
                ('P002', 'Foundation B', 'Brand Y', 45.0, 42750.0, 3.2, 'Skincare'),
                ('P003', 'Mascara C', 'Brand X', 20.0, 19000.0, 4.0, 'Makeup')
        """))

        # clean_reviews
        conn.execute(text("""
            CREATE TABLE clean_reviews (
                author_id TEXT,
                product_id TEXT,
                product_name TEXT,
                brand_name TEXT,
                rating REAL,
                is_recommended REAL,
                helpfulness REAL,
                total_feedback_count REAL,
                total_neg_feedback_count REAL,
                total_pos_feedback_count REAL,
                submission_time TEXT,
                review_text TEXT,
                review_title TEXT,
                skin_tone TEXT,
                eye_color TEXT,
                skin_type TEXT,
                hair_color TEXT,
                price_usd REAL
            )
        """))
        conn.execute(text("""
            INSERT INTO clean_reviews
                (author_id, product_id, rating, is_recommended, review_text)
            VALUES
                ('U001', 'P001', 5.0, 1.0, 'Love it!'),
                ('U002', 'P001', 3.0, 0.0, 'Not great.'),
                ('U003', 'P002', 4.0, 1.0, 'Pretty good.')
        """))

        # exchange_rates
        conn.execute(text("""
            CREATE TABLE exchange_rates (
                date TEXT,
                rate REAL,
                source TEXT,
                loaded_at TEXT
            )
        """))
        conn.execute(text("""
            INSERT INTO exchange_rates (date, rate, source, loaded_at)
            VALUES ('2025-06-01', 950.0, 'api', '2025-06-01')
        """))

        # model_metrics
        conn.execute(text("""
            CREATE TABLE model_metrics (
                model_name TEXT,
                f1_test REAL,
                f1_cv_grid REAL,
                f1_cv_random REAL
            )
        """))
        conn.execute(text("""
            INSERT INTO model_metrics (model_name, f1_test, f1_cv_grid, f1_cv_random)
            VALUES ('GradientBoostingClassifier', 0.85, 0.83, 0.82)
        """))

        # clusters
        conn.execute(text("""
            CREATE TABLE clusters (
                k INTEGER,
                inertia REAL,
                silhouette REAL,
                is_best INTEGER
            )
        """))
        conn.execute(text("""
            INSERT INTO clusters (k, inertia, silhouette, is_best)
            VALUES
                (2, 1240247.5, 0.22, 1),
                (3, 1191078.5, 0.11, 0)
        """))

    return engine


@pytest.fixture(scope="session")
def client(test_engine):
    """TestClient de FastAPI usando la base de datos en memoria."""
    TestingSessionLocal = sessionmaker(
        autocommit=False, autoflush=False, bind=test_engine
    )

    def override_get_db():
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()
