# Referencia de la API — Sephora Intelligence

## Información general

| Campo | Valor |
|---|---|
| URL base | `http://localhost:8000` |
| Documentación interactiva | `http://localhost:8000/docs` |
| Documentación alternativa | `http://localhost:8000/redoc` |
| Versión | 1.0.0 |
| Formato de respuesta | JSON |

---

## Endpoints

### GET `/`
Health check de la API.

**Respuesta exitosa (200):**
```json
{
  "status": "ok",
  "database": "connected",
  "docs": "/docs"
}
```

---

### GET `/products`
Lista paginada de productos con filtros opcionales.

**Parámetros de query:**

| Parámetro | Tipo | Requerido | Descripción |
|---|---|---|---|
| `category` | string | No | Filtrar por categoría (ej. `Makeup`, `Skincare`) |
| `brand` | string | No | Filtrar por marca (ej. `NARS`) |
| `min_price` | float ≥ 0 | No | Precio mínimo en USD |
| `max_price` | float ≥ 0 | No | Precio máximo en USD |
| `limit` | int 1-500 | No | Máximo de resultados (default: 50) |
| `offset` | int ≥ 0 | No | Desplazamiento para paginación (default: 0) |

**Ejemplo:**
```
GET /products?category=Makeup&min_price=10&max_price=50&limit=20
```

**Respuesta exitosa (200):**
```json
[
  {
    "product_id": "P123456",
    "product_name": "Lipstick Rouge",
    "brand_name": "Charlotte Tilbury",
    "price_usd": 38.0,
    "price_clp": 36100.0,
    "rating": 4.5,
    "loves_count": 12500.0,
    "primary_category": "Makeup"
  }
]
```

**Errores:**
- `422`: parámetros inválidos (ej. `limit=0`, `min_price=-10`)

---

### GET `/products/{product_id}`
Detalle de un producto por su ID.

**Parámetros de ruta:**

| Parámetro | Tipo | Descripción |
|---|---|---|
| `product_id` | string | ID único del producto |

**Ejemplo:**
```
GET /products/P18736
```

**Respuesta exitosa (200):** mismo esquema que `/products`

**Errores:**
- `404`: producto no encontrado

---

### GET `/reviews`
Lista paginada de reseñas con filtros opcionales.

**Parámetros de query:**

| Parámetro | Tipo | Requerido | Descripción |
|---|---|---|---|
| `product_id` | string | No | Filtrar reseñas de un producto específico |
| `min_rating` | float 1-5 | No | Rating mínimo |
| `is_recommended` | int 0 o 1 | No | 1=recomendado, 0=no recomendado |
| `limit` | int 1-5000 | No | Máximo de resultados (default: 50) |
| `offset` | int ≥ 0 | No | Desplazamiento (default: 0) |

**Ejemplo:**
```
GET /reviews?product_id=P18736&is_recommended=1&limit=100
```

**Respuesta exitosa (200):**
```json
[
  {
    "author_id": "U789012",
    "product_id": "P18736",
    "rating": 5.0,
    "is_recommended": 1.0,
    "review_text": "Amazing product, highly recommend!",
    "skin_tone": "fair",
    "skin_type": "dry"
  }
]
```

---

### GET `/reviews/sentiment/summary`
Análisis de sentimiento vs tasa de recomendación.

Devuelve el resumen del análisis realizado sobre 500 reseñas usando la API de HuggingFace.

**Sin parámetros.**

**Respuesta exitosa (200):**
```json
[
  {
    "sentiment": "positive",
    "total_reviews": 358,
    "pct_recommended": 95.5,
    "avg_sentiment_score": 0.9412
  },
  {
    "sentiment": "neutral",
    "total_reviews": 39,
    "pct_recommended": 61.5,
    "avg_sentiment_score": 0.7823
  },
  {
    "sentiment": "negative",
    "total_reviews": 103,
    "pct_recommended": 9.7,
    "avg_sentiment_score": 0.8901
  }
]
```

**Errores:**
- `404`: tabla `sentiment_results` no disponible (ejecutar pipeline ETL primero)

---

### GET `/metrics/model`
Métricas del modelo de Machine Learning entrenado en EP2.

**Sin parámetros.**

**Respuesta exitosa (200):**
```json
{
  "model_name": "RF RandomizedSearchCV",
  "n_estimators": 200,
  "max_depth": 20,
  "learning_rate": null,
  "random_state": 42,
  "silhouette_score": 0.312,
  "best_k": 3,
  "final_inertia": 45000.0,
  "pca_variance_2d": 0.18
}
```

**Errores:**
- `404`: tabla `model_metrics` no disponible

---

### GET `/clusters`
Resumen de los clusters K-Means evaluados durante el análisis no supervisado.

**Sin parámetros.**

**Respuesta exitosa (200):**
```json
[
  {"k": 2, "inertia": 60000.0, "silhouette": 0.28, "is_best": false},
  {"k": 3, "inertia": 45000.0, "silhouette": 0.31, "is_best": true},
  {"k": 4, "inertia": 38000.0, "silhouette": 0.29, "is_best": false}
]
```

**Errores:**
- `404`: tabla `clusters` no disponible

---

### GET `/exchange-rate`
Último tipo de cambio USD → CLP cargado por el pipeline ETL.

**Sin parámetros.**

**Respuesta exitosa (200):**
```json
{
  "date": "2025-06-25",
  "rate": 950.5,
  "source": "api",
  "loaded_at": "2025-06-25T10:30:00"
}
```

El campo `source` puede ser `"api"` (obtenido en tiempo real) o `"fallback"` (valor por defecto del `.env`).

**Errores:**
- `404`: sin datos de tipo de cambio disponibles

---

## Códigos de error

| Código | Significado |
|---|---|
| `200` | Éxito |
| `404` | Recurso no encontrado o tabla no disponible |
| `422` | Error de validación en parámetros de entrada |
| `500` | Error interno del servidor |

---

## Autenticación

La API no requiere autenticación. Está diseñada para uso interno en red local o Docker.
