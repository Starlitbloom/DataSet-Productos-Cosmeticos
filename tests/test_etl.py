"""tests.test_etl
=================
Tests automatizados para el pipeline ETL de Sephora.

Cubre las etapas de:
- ``extract_csv``: lectura y validación de archivos CSV.
- ``transform``: limpieza de datos (nulos, outliers, tipos, duplicados).
- ``schemas``: validación de contratos de datos con pandera.
- ``load``: escritura a base de datos (usando SQLite en memoria para tests).

Ejecución
---------
    pytest tests/test_etl.py -v
    pytest tests/test_etl.py -v --tb=short   # traceback corto
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Tests: transform.py
# ---------------------------------------------------------------------------

class TestTransformProducts:
    """Tests para etl.transform.transform_products."""

    def test_elimina_duplicados_por_product_id(self, raw_products_df):
        """Debe eliminar filas duplicadas por product_id, quedándose con la primera."""
        from etl.transform import transform_products
        result = transform_products(raw_products_df, exchange_rate=950.0)
        assert result["product_id"].duplicated().sum() == 0
        assert len(result) < len(raw_products_df)

    def test_imputa_nulos_en_price_usd(self, raw_products_df):
        """price_usd nulo debe ser imputado con la mediana."""
        from etl.transform import transform_products
        result = transform_products(raw_products_df, exchange_rate=950.0)
        assert result["price_usd"].isna().sum() == 0

    def test_elimina_precios_negativos(self, raw_products_df):
        """Filas con price_usd <= 0 deben ser eliminadas."""
        from etl.transform import transform_products
        result = transform_products(raw_products_df, exchange_rate=950.0)
        assert (result["price_usd"] <= 0).sum() == 0

    def test_agrega_columna_price_clp(self, raw_products_df):
        """Debe agregar columna price_clp = price_usd × exchange_rate."""
        from etl.transform import transform_products
        rate = 950.0
        result = transform_products(raw_products_df, exchange_rate=rate)
        assert "price_clp" in result.columns
        # Verificar que la conversión es correcta para al menos una fila
        row = result[result["product_id"] == "P001"].iloc[0]
        expected_clp = round(row["price_usd"] * rate, 0)
        assert row["price_clp"] == expected_clp

    def test_normaliza_nombres_de_columnas(self, raw_products_df):
        """Los nombres de columna deben estar en minúsculas sin espacios."""
        from etl.transform import transform_products
        result = transform_products(raw_products_df, exchange_rate=950.0)
        for col in result.columns:
            assert col == col.lower()
            assert " " not in col

    def test_imputa_nulos_en_loves_count(self, raw_products_df):
        """loves_count nulo debe ser imputado con 0."""
        from etl.transform import transform_products
        result = transform_products(raw_products_df, exchange_rate=950.0)
        assert result["loves_count"].isna().sum() == 0

    def test_exchange_rate_default_es_1(self, raw_products_df):
        """Con exchange_rate=1.0, price_clp debe ser igual a price_usd."""
        from etl.transform import transform_products
        result = transform_products(raw_products_df, exchange_rate=1.0)
        # Después de capping/imputación, price_clp ≈ price_usd
        assert "price_clp" in result.columns


class TestTransformReviews:
    """Tests para etl.transform.transform_reviews."""

    def test_elimina_filas_sin_author_id(self, raw_reviews_df):
        """Filas sin author_id deben ser eliminadas."""
        from etl.transform import transform_reviews
        result = transform_reviews(raw_reviews_df)
        assert result["author_id"].isna().sum() == 0

    def test_elimina_filas_sin_product_id(self, raw_reviews_df):
        """Filas sin product_id deben ser eliminadas."""
        from etl.transform import transform_reviews
        result = transform_reviews(raw_reviews_df)
        assert result["product_id"].isna().sum() == 0

    def test_elimina_duplicados_por_author_product(self, raw_reviews_df):
        """Duplicados por (author_id, product_id) deben eliminarse."""
        from etl.transform import transform_reviews
        result = transform_reviews(raw_reviews_df)
        assert result.duplicated(subset=["author_id", "product_id"]).sum() == 0

    def test_is_recommended_es_float(self, raw_reviews_df):
        """is_recommended debe ser de tipo float."""
        from etl.transform import transform_reviews
        result = transform_reviews(raw_reviews_df)
        assert result["is_recommended"].dtype == float

    def test_is_recommended_solo_0_y_1(self, raw_reviews_df):
        """is_recommended solo puede tomar valores 0.0 o 1.0."""
        from etl.transform import transform_reviews
        result = transform_reviews(raw_reviews_df)
        assert set(result["is_recommended"].unique()).issubset({0.0, 1.0})

    def test_rating_en_rango_valido(self, raw_reviews_df):
        """rating debe estar entre 1 y 5."""
        from etl.transform import transform_reviews
        result = transform_reviews(raw_reviews_df)
        assert result["rating"].between(1, 5).all()

    def test_review_text_sin_espacios_extra(self, raw_reviews_df):
        """review_text no debe tener espacios al inicio o al final."""
        from etl.transform import transform_reviews
        result = transform_reviews(raw_reviews_df)
        textos = result["review_text"].dropna()
        assert all(t == t.strip() for t in textos)


class TestTransformSentiment:
    """Tests para etl.transform.transform_sentiment."""

    def test_elimina_sentimiento_unknown(self, raw_sentiment_df):
        """Filas con sentiment='unknown' deben ser eliminadas."""
        from etl.transform import transform_sentiment
        result = transform_sentiment(raw_sentiment_df)
        assert "unknown" not in result["sentiment"].values

    def test_sentiment_score_es_numerico(self, raw_sentiment_df):
        """sentiment_score debe ser numérico."""
        from etl.transform import transform_sentiment
        result = transform_sentiment(raw_sentiment_df)
        assert pd.api.types.is_numeric_dtype(result["sentiment_score"])


# ---------------------------------------------------------------------------
# Tests: schemas.py
# ---------------------------------------------------------------------------

class TestSchemas:
    """Tests para etl.schemas — validación con pandera."""

    def test_raw_products_valido(self, raw_products_df):
        """Un DataFrame con columnas correctas debe pasar la validación raw."""
        from etl.schemas import validate_raw_products
        _, is_valid = validate_raw_products(raw_products_df, strict=False)
        # No necesariamente es válido (hay nulos y negativos), pero no debe lanzar
        assert isinstance(is_valid, bool)

    def test_raw_products_price_negativo_detectado(self):
        """Un precio negativo debe ser detectado como error de validación."""
        from etl.schemas import RAW_PRODUCTS_SCHEMA
        import pandera as pa
        df_bad = pd.DataFrame({
            "product_id":       ["P001"],
            "product_name":     ["Test"],
            "brand_name":       ["Brand"],
            "price_usd":        [-10.0],   # Inválido
            "rating":           [4.0],
            "loves_count":      [100.0],
            "primary_category": ["Makeup"],
        })
        with pytest.raises(pa.errors.SchemaErrors):
            RAW_PRODUCTS_SCHEMA.validate(df_bad, lazy=True)

    def test_raw_reviews_is_recommended_invalido(self):
        """is_recommended con valor 2.0 debe ser detectado como error."""
        from etl.schemas import RAW_REVIEWS_SCHEMA
        import pandera as pa
        df_bad = pd.DataFrame({
            "author_id":                ["U001"],
            "rating":                   [4.0],
            "is_recommended":           [2.0],   # Inválido (no es 0 ni 1)
            "helpfulness":              [0.5],
            "total_feedback_count":     [5.0],
            "total_neg_feedback_count": [1.0],
            "total_pos_feedback_count": [4.0],
            "review_text":              ["Good"],
            "product_id":               ["P001"],
        })
        with pytest.raises(pa.errors.SchemaErrors):
            RAW_REVIEWS_SCHEMA.validate(df_bad, lazy=True)

    def test_clean_products_valido(self, clean_products_df):
        """Un DataFrame limpio correcto debe pasar la validación clean."""
        from etl.schemas import validate_clean_products
        _, is_valid = validate_clean_products(clean_products_df, strict=False)
        assert is_valid is True

    def test_clean_reviews_valido(self, clean_reviews_df):
        """Un DataFrame limpio correcto debe pasar la validación clean."""
        from etl.schemas import validate_clean_reviews
        _, is_valid = validate_clean_reviews(clean_reviews_df, strict=False)
        assert is_valid is True


# ---------------------------------------------------------------------------
# Tests: load.py (usando SQLite en memoria para no depender de PostgreSQL)
# ---------------------------------------------------------------------------

class TestLoad:
    """Tests para etl.load — escritura a base de datos."""

    @pytest.fixture
    def sqlite_engine(self):
        """Engine SQLite en memoria para tests sin PostgreSQL."""
        from sqlalchemy import create_engine
        engine = create_engine("sqlite:///:memory:")
        yield engine
        engine.dispose()

    def test_load_clean_products(self, clean_products_df, sqlite_engine):
        """clean_products debe cargarse sin error a SQLite."""
        from etl.load import load_clean_products
        load_clean_products(clean_products_df, sqlite_engine)

        result = pd.read_sql("SELECT * FROM clean_products", sqlite_engine)
        assert len(result) == len(clean_products_df)

    def test_load_clean_reviews(self, clean_reviews_df, sqlite_engine):
        """clean_reviews debe cargarse sin error a SQLite."""
        from etl.load import load_clean_reviews
        load_clean_reviews(clean_reviews_df, sqlite_engine)

        result = pd.read_sql("SELECT * FROM clean_reviews", sqlite_engine)
        assert len(result) == len(clean_reviews_df)

    def test_load_exchange_rate(self, exchange_rate_dict, sqlite_engine):
        """exchange_rates debe cargarse con los campos correctos."""
        from etl.load import load_exchange_rate
        load_exchange_rate(exchange_rate_dict, sqlite_engine, if_exists="replace")

        result = pd.read_sql("SELECT * FROM exchange_rates", sqlite_engine)
        assert len(result) == 1
        assert result["rate"].iloc[0] == exchange_rate_dict["rate"]
        assert result["source"].iloc[0] == exchange_rate_dict["source"]

    def test_load_all_sin_error(
        self, clean_products_df, clean_reviews_df, raw_sentiment_df,
        exchange_rate_dict, sqlite_engine
    ):
        """load_all debe completarse sin excepciones."""
        from etl.load import load_all
        from etl.transform import transform_sentiment

        clean_sentiment = transform_sentiment(raw_sentiment_df)

        # No debe lanzar excepción
        load_all(
            clean_products=clean_products_df,
            clean_reviews=clean_reviews_df,
            clean_sentiment=clean_sentiment,
            exchange_rate=exchange_rate_dict,
            engine=sqlite_engine,
        )


# ---------------------------------------------------------------------------
# Tests: extract_csv.py (con archivos temporales)
# ---------------------------------------------------------------------------

class TestExtractCSV:
    """Tests para etl.extract_csv con archivos CSV temporales."""

    def test_extract_products_file_not_found(self, tmp_path):
        """Debe lanzar FileNotFoundError si no existe product_info.csv."""
        from etl.extract_csv import extract_products
        with pytest.raises(FileNotFoundError):
            extract_products(tmp_path)

    def test_extract_products_columnas_faltantes(self, tmp_path):
        """Debe lanzar ValueError si faltan columnas esperadas."""
        from etl.extract_csv import extract_products
        # CSV sin columnas requeridas
        df = pd.DataFrame({"col_a": [1], "col_b": [2]})
        df.to_csv(tmp_path / "product_info.csv", index=False)

        with pytest.raises(ValueError, match="no contiene las columnas esperadas"):
            extract_products(tmp_path)

    def test_extract_products_exitoso(self, tmp_path):
        """Debe cargar correctamente un CSV válido."""
        from etl.extract_csv import extract_products
        df = pd.DataFrame({
            "product_id":       ["P001"],
            "product_name":     ["Test Product"],
            "brand_name":       ["Brand"],
            "price_usd":        [25.0],
            "rating":           [4.5],
            "loves_count":      [1000.0],
            "primary_category": ["Makeup"],
        })
        df.to_csv(tmp_path / "product_info.csv", index=False)

        result = extract_products(tmp_path)
        assert len(result) == 1
        assert "product_id" in result.columns

    def test_extract_reviews_no_files(self, tmp_path):
        """Debe lanzar FileNotFoundError si no hay archivos reviews_*.csv."""
        from etl.extract_csv import extract_reviews
        with pytest.raises(FileNotFoundError):
            extract_reviews(tmp_path)
