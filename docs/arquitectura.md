# Arquitectura del Sistema — Inteligencia Predictiva Sephora

## Descripción general

Este sistema construye un pipeline **end-to-end** de análisis de datos sobre el dataset público de reseñas de productos cosméticos de Sephora. Integra cuatro fuentes de datos distintas, aplica transformaciones y modelos de machine learning desarrollados en EP2, y expone los resultados a través de una API REST y un dashboard interactivo con vistas diferenciadas por audiencia.

---

## Diagrama de arquitectura

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           FUENTES DE DATOS                                  │
│                                                                             │
│  ┌─────────────┐  ┌──────────────────────┐  ┌─────────────────┐  ┌──────┐ │
│  │ CSV (local) │  │  HuggingFace API     │  │exchangerate-api │  │  DB  │ │
│  │             │  │  (Sentimiento)       │  │  (USD → CLP)    │  │      │ │
│  │product_info │  │                      │  │                 │  │ Post │ │
│  │reviews_*.cs │  │ positive/neutral/    │  │  tipo de cambio │  │ greS │ │
│  │             │  │ negative             │  │  en tiempo real │  │  QL  │ │
│  └──────┬──────┘  └──────────┬───────────┘  └────────┬────────┘  └──┬───┘ │
└─────────┼────────────────────┼─────────────────────── ┼─────────────┼─────┘
          │                    │                         │             │
          ▼                    ▼                         ▼             ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            ETL PIPELINE                                     │
│                                                                             │
│  extract_csv.py    extract_api.py (HF)   extract_api.py (CLP)  extract_db  │
│        │                 │                       │                  │       │
│        └─────────────────┴───────────────────────┴──────────────────┘       │
│                                      │                                      │
│                                      ▼                                      │
│                               transform.py                                  │
│                         (reutiliza src/ de EP2)                             │
│                        • limpieza y nulos                                   │
│                        • encoding y escalado                                │
│                        • precio en CLP (con tasa API)                      │
│                        • columna sentiment (de HuggingFace)                │
│                        • validación con pandera                             │
│                                      │                                      │
│                                      ▼                                      │
│                                   load.py                                   │
│             • clean_products     • clean_reviews    • sentiment_results     │
│             • exchange_rates     • model_metrics    • clusters              │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
               ┌───────────────────────┼───────────────────────┐
               ▼                                               ▼
┌──────────────────────┐                         ┌────────────────────────┐
│    DATA WAREHOUSE    │                         │    MODELOS ML (EP2)    │
│    (PostgreSQL)      │                         │                        │
│                      │                         │  • RandomForest        │
│  clean_products      │◄────────────────────────│  • Gradient Boosting   │
│  clean_reviews       │                         │  • K-Means             │
│  sentiment_results   │                         │  • PCA                 │
│  exchange_rates      │                         └────────────────────────┘
│  model_metrics       │
│  clusters            │
└──────────┬───────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           CAPA DE EXPOSICIÓN                                │
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │                       API REST (FastAPI)                               │ │
│  │                                                                       │ │
│  │  GET /products          GET /reviews                                  │ │
│  │  GET /metrics/model     GET /clusters                                 │ │
│  │  GET /exchange-rate     GET /sentiment                                │ │
│  │                                                                       │ │
│  │  Swagger UI en /docs                                                  │ │
│  └───────────────────────────────────┬───────────────────────────────────┘ │
│                                      │                                      │
│  ┌───────────────────────────────────▼───────────────────────────────────┐ │
│  │                      DASHBOARD (Streamlit)                             │ │
│  │                                                                       │ │
│  │  👔 Vista ejecutiva  — KPIs, precios en CLP, top marcas               │ │
│  │  🔬 Vista técnica    — métricas ML, curva ROC, F1, sentiment vs rec.  │ │
│  │  ⚙️  Vista operativa  — clusters, filtros interactivos por producto    │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
              ┌────────────────────────┴────────────────────────┐
              ▼                                                  ▼
┌─────────────────────┐                              ┌────────────────────┐
│       Docker        │                              │    Git / GitHub    │
│                     │                              │                    │
│  Dockerfile.api     │                              │  ramas feature/    │
│  Dockerfile.dash    │                              │  Pull Requests     │
│  docker-compose.yml │                              │  Issues            │
│  .env               │                              │  Conv. Commits     │
└─────────────────────┘                              └────────────────────┘
```

---

## Justificación de cada fuente de datos

### Fuente 1: CSV local (`data/raw/`)

**Qué es:** Los archivos originales del dataset público de Sephora — `product_info.csv` (≈8.000 productos) y los archivos `reviews_*.csv` (≈1 millón de reseñas).

**Por qué esta fuente:** Es el núcleo del proyecto y el punto de partida natural. Los CSV representan el caso más común de "datos históricos" en proyectos reales: exportaciones de sistemas legacy, descargas de Kaggle, reportes en Excel, etc.

**Cómo se integra:** `etl/extract_csv.py` los carga con validación de columnas y manejo de errores. Esta extracción solo ocurre en la carga inicial; en corridas posteriores del ETL los datos vienen desde PostgreSQL.

---

### Fuente 2: HuggingFace Inference API (análisis de sentimiento)

**Qué es:** API REST de HuggingFace que expone el modelo `cardiffnlp/twitter-roberta-base-sentiment-latest`. Recibe el texto de una reseña y devuelve una etiqueta de sentimiento: `positive`, `neutral` o `negative`, junto con un score de confianza.

**Por qué esta fuente:** Agrega una dimensión cualitativa que los datos originales no tienen. Permite responder preguntas de negocio como: *¿el sentimiento del texto predice mejor la recomendación que el rating numérico?* El análisis sentimiento↔is_recommended es una de las visualizaciones clave del dashboard técnico.

**Por qué este modelo:** `cardiffnlp/twitter-roberta-base-sentiment-latest` está entrenado en texto informal (tweets, reseñas de consumidores), lo que lo hace más adecuado que modelos de texto formal para reviews de cosméticos.

**Decisión de muestra:** Se analiza una muestra estratificada de 500 reseñas (configurable en `.env`) en lugar del dataset completo. Esto es una decisión técnica deliberada: con ~1 millón de reseñas, consultar la API una a una sería inviable en tiempo y cuota. En producción, se usaría un modelo local con batching. La muestra es suficiente para el análisis estadístico.

**Robustez:** El módulo implementa retry automático con backoff exponencial para manejar errores 503 (modelo en cold start), y un sistema de caché local para no re-consultar la API en cada corrida del ETL.

---

### Fuente 3: exchangerate-api.com (tipo de cambio USD → CLP)

**Qué es:** API pública y gratuita que devuelve tipos de cambio en tiempo real. Se usa para obtener la tasa USD → CLP.

**Por qué esta fuente:** Los precios de Sephora están en USD, pero el mercado objetivo del dashboard ejecutivo es Chile. Mostrar precios en CLP hace los datos accionables para una audiencia de negocio chilena sin necesidad de conversión manual.

**Por qué `open.er-api.com` y no Frankfurter:** Frankfurter cubre principalmente monedas del Banco Central Europeo y no incluye CLP de forma confiable. `open.er-api.com` es gratuita, sin clave, y soporta más de 160 monedas incluyendo CLP.

**Robustez:** El módulo incluye un valor de fallback configurable en `.env` (`FALLBACK_USD_CLP`) para que el pipeline no falle si la API no está disponible, usando la última tasa conocida.

---

### Fuente 4: Base de datos PostgreSQL (Docker)

**Qué es:** Una instancia de PostgreSQL 16 corriendo en un contenedor Docker. Cumple dos roles: (1) almacena los datos crudos como "fuente SQL" y (2) actúa como data warehouse con las tablas limpias y resultados del modelo.

**Por qué PostgreSQL:** Es el motor relacional open-source más maduro y usado en la industria. SQLAlchemy ofrece una capa de abstracción que permitiría migrar a otro motor (MySQL, SQLite para tests) con mínimos cambios de código.

**Por qué Docker:** Garantiza reproducibilidad total del entorno. Cualquier persona que clone el repo puede levantar la base con `docker compose up -d db` sin instalar PostgreSQL localmente.

---

## Stack tecnológico y justificación

| Componente | Herramienta | Justificación |
|---|---|---|
| Lenguaje | Python 3.11 | Ecosistema dominante en Data Science; compatibilidad con EP2 |
| Extracción | pandas, requests | Estándar de la industria; ya usado en EP2 |
| Validación de esquemas | pandera | Integración nativa con pandas; contratos de datos explícitos |
| Base de datos | PostgreSQL 16 | Open-source, robusto, soportado por SQLAlchemy |
| ORM / conexión DB | SQLAlchemy | Abstracción de motor; facilita testing con SQLite |
| Sentimiento | HuggingFace Inference API | Modelo preentrenado en texto informal; sin infraestructura propia |
| Tipo de cambio | exchangerate-api.com | Gratuita con key, cubre CLP, estable |
| API REST | FastAPI + uvicorn | Swagger automático, validación Pydantic, alto rendimiento |
| Dashboard | Streamlit | Curva de aprendizaje baja, Python puro, ideal para prototipos |
| Visualización | Plotly | Gráficos interactivos compatibles con Streamlit |
| Testing | pytest | Estándar de la industria; fixtures y parametrización |
| Variables de entorno | python-dotenv | Separación de configuración y código |
| Containerización | Docker + Compose | Reproducibilidad total del entorno |
| Control de versiones | Git + GitHub | Flujo profesional con ramas, PRs e issues |

---

## Flujo de datos resumido

```
CSV (productos/reseñas) ──┐
HuggingFace API           ├──► ETL (extract → transform → load) ──► PostgreSQL (DW)
exchangerate-api (CLP)    │                                               │
PostgreSQL raw            ─┘                                              ▼
                                                                  FastAPI (API REST)
                                                                          │
                                                                          ▼
                                                              Streamlit Dashboard
                                                              (3 vistas por audiencia)
```

---

## Decisiones de diseño clave

**¿Por qué un valor de fallback en la API de tipo de cambio?**
En entornos de producción, las dependencias externas fallan. Un pipeline que se cae porque una API no responde no es robusto. El fallback garantiza que el ETL puede completarse incluso sin conexión a internet, usando la última tasa conocida.

**¿Por qué reutilizar `src/` de EP2 en lugar de reescribir el ETL?**
El código de `data_preprocessing.py` y `transformers.py` ya fue validado y optimizado en EP2. Reutilizarlo reduce el riesgo de bugs.

**¿Por qué separar extract, transform y load en módulos distintos?**
Sigue el principio de responsabilidad única (SRP). Permite testear cada etapa de forma aislada, reemplazar una fuente sin tocar las otras, y facilita el debugging cuando algo falla.

**¿Por qué una muestra para el análisis de sentimiento?**
En entornos de producción, las APIs externas tienen costo y límites de velocidad. Tomar una muestra estratificada es una decisión técnica deliberada que demuestra criterio de ingeniería: se obtiene suficiente poder estadístico sin incurrir en costos innecesarios ni tiempos de espera prohibitivos.
