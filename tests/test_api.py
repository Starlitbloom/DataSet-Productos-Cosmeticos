"""tests.test_api
=================
Tests de integración para la API FastAPI de Sephora.

Usa TestClient con una base de datos SQLite en memoria para no depender
de PostgreSQL ni de datos reales. Los fixtures están en conftest.py.

Cobertura
---------
- GET /                      health check
- GET /products              lista con y sin filtros
- GET /products/{id}         detalle y 404
- GET /reviews               lista con y sin filtros
- GET /metrics/model         métricas del modelo
- GET /clusters              resumen de clusters
- GET /exchange-rate         tipo de cambio
- Validación 422             query params inválidos
"""

from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

def test_root_ok(client):
    """GET / retorna status ok."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "docs" in data


# ---------------------------------------------------------------------------
# GET /products
# ---------------------------------------------------------------------------

def test_get_products_returns_list(client):
    """GET /products retorna una lista de productos."""
    response = client.get("/products")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) > 0


def test_get_products_has_required_fields(client):
    """Cada producto tiene los campos mínimos esperados."""
    response = client.get("/products")
    product = response.json()[0]
    assert "product_id" in product
    assert "product_name" in product
    assert "price_usd" in product


def test_get_products_filter_by_category(client):
    """GET /products?category=Makeup retorna solo productos de esa categoría."""
    response = client.get("/products", params={"category": "Makeup"})
    assert response.status_code == 200
    data = response.json()
    assert all(
        p["primary_category"] and p["primary_category"].lower() == "makeup"
        for p in data
    )


def test_get_products_filter_by_brand(client):
    """GET /products?brand=Brand X retorna solo productos de esa marca."""
    response = client.get("/products", params={"brand": "Brand X"})
    assert response.status_code == 200
    data = response.json()
    assert all(p["brand_name"] == "Brand X" for p in data)


def test_get_products_filter_by_price_range(client):
    """GET /products con min_price y max_price filtra correctamente."""
    response = client.get("/products", params={"min_price": 20.0, "max_price": 30.0})
    assert response.status_code == 200
    data = response.json()
    for p in data:
        assert p["price_usd"] >= 20.0
        assert p["price_usd"] <= 30.0


def test_get_products_pagination(client):
    """GET /products con limit=1 retorna exactamente 1 producto."""
    response = client.get("/products", params={"limit": 1})
    assert response.status_code == 200
    assert len(response.json()) == 1


def test_get_products_invalid_limit(client):
    """GET /products con limit=0 retorna 422."""
    response = client.get("/products", params={"limit": 0})
    assert response.status_code == 422


def test_get_products_invalid_min_price(client):
    """GET /products con min_price negativo retorna 422."""
    response = client.get("/products", params={"min_price": -10})
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# GET /products/{id}
# ---------------------------------------------------------------------------

def test_get_product_by_id_ok(client):
    """GET /products/P001 retorna el producto correcto."""
    response = client.get("/products/P001")
    assert response.status_code == 200
    data = response.json()
    assert data["product_id"] == "P001"
    assert data["product_name"] == "Lipstick A"


def test_get_product_by_id_not_found(client):
    """GET /products/INVALID retorna 404."""
    response = client.get("/products/INVALID_ID_99999")
    assert response.status_code == 404
    assert "no encontrado" in response.json()["detail"].lower()


# ---------------------------------------------------------------------------
# GET /reviews
# ---------------------------------------------------------------------------

def test_get_reviews_returns_list(client):
    """GET /reviews retorna una lista de reseñas."""
    response = client.get("/reviews")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) > 0


def test_get_reviews_has_required_fields(client):
    """Cada reseña tiene los campos mínimos esperados."""
    response = client.get("/reviews")
    review = response.json()[0]
    assert "author_id" in review
    assert "product_id" in review
    assert "rating" in review
    assert "review_text" in review


def test_get_reviews_filter_by_product_id(client):
    """GET /reviews?product_id=P001 retorna solo reseñas de ese producto."""
    response = client.get("/reviews", params={"product_id": "P001"})
    assert response.status_code == 200
    data = response.json()
    assert len(data) > 0
    assert all(r["product_id"] == "P001" for r in data)


def test_get_reviews_filter_by_is_recommended(client):
    """GET /reviews?is_recommended=1 retorna solo reseñas recomendadas."""
    response = client.get("/reviews", params={"is_recommended": 1})
    assert response.status_code == 200
    data = response.json()
    assert all(r["is_recommended"] == 1.0 for r in data)


def test_get_reviews_invalid_is_recommended(client):
    """GET /reviews con is_recommended=5 retorna 422."""
    response = client.get("/reviews", params={"is_recommended": 5})
    assert response.status_code == 422


def test_get_reviews_invalid_min_rating(client):
    """GET /reviews con min_rating=10 retorna 422."""
    response = client.get("/reviews", params={"min_rating": 10})
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# GET /metrics/model
# ---------------------------------------------------------------------------

def test_get_metrics_ok(client):
    """GET /metrics/model retorna las métricas del modelo."""
    response = client.get("/metrics/model")
    assert response.status_code == 200
    data = response.json()
    assert "model_name" in data


# ---------------------------------------------------------------------------
# GET /clusters
# ---------------------------------------------------------------------------

def test_get_clusters_ok(client):
    """GET /clusters retorna la lista de clusters evaluados."""
    response = client.get("/clusters")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) > 0


def test_get_clusters_has_required_fields(client):
    """Cada cluster tiene k, inertia, silhouette e is_best."""
    response = client.get("/clusters")
    cluster = response.json()[0]
    assert "k" in cluster
    assert "inertia" in cluster
    assert "silhouette" in cluster
    assert "is_best" in cluster


def test_get_clusters_has_best(client):
    """Al menos un cluster tiene is_best=True."""
    response = client.get("/clusters")
    data = response.json()
    assert any(c["is_best"] for c in data)


# ---------------------------------------------------------------------------
# GET /exchange-rate.
# ---------------------------------------------------------------------------

def test_get_exchange_rate_ok(client):
    """GET /exchange-rate retorna el tipo de cambio."""
    response = client.get("/exchange-rate")
    assert response.status_code == 200
    data = response.json()
    assert "rate" in data
    assert "date" in data
    assert "source" in data
    assert data["rate"] > 0
