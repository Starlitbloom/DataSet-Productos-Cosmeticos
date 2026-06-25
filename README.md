# 🌸 Sephora Intelligence

Pipeline ETL end-to-end con análisis de Machine Learning, API REST y dashboard interactivo sobre el dataset público de reseñas de productos cosméticos de Sephora US.

## Descripción

Este proyecto integra **4 fuentes de datos**, aplica modelos de ML entrenados en EP2, y expone los resultados a través de una API REST documentada y un dashboard interactivo con 3 vistas diferenciadas por audiencia.

| Componente | Tecnología |
|---|---|
| Pipeline ETL | Python, pandas, SQLAlchemy, pandera |
| Base de datos | PostgreSQL 16 (Docker) |
| API REST | FastAPI + uvicorn |
| Dashboard | Streamlit + Plotly |
| Containerización | Docker + Docker Compose |
| Testing | pytest (51 tests) |

## Estructura del proyecto

```
DataSet-Productos-Cosmeticos/
├── api/                        # API REST FastAPI
│   ├── main.py                 # App principal + health check
│   ├── database.py             # Conexión SQLAlchemy
│   ├── models.py               # Modelos Pydantic
│   └── routers/                # Endpoints por recurso
│       ├── products.py
│       ├── reviews.py
│       ├── metrics.py
│       └── clusters.py
├── dashboards/                 # Dashboard Streamlit
│   ├── app.py                  # Página principal
│   └── pages/
│       ├── 1_vista_ejecutiva.py
│       ├── 2_vista_tecnica.py
│       └── 3_vista_operativa.py
├── data/
│   ├── raw/                    # CSVs originales de Sephora
│   └── processed/              # Artefactos procesados
├── docker/                     # Containerización
│   ├── docker-compose.yml
│   ├── Dockerfile.api
│   └── Dockerfile.dashboard
├── docs/                       # Documentación técnica
├── etl/                        # Pipeline ETL
│   ├── extract_csv.py
│   ├── extract_api.py          # HuggingFace + tipo de cambio
│   ├── extract_db.py
│   ├── transform.py
│   ├── load.py
│   ├── schemas.py
│   └── pipeline.py
├── models/trained_models/      # Modelos serializados de EP2
├── notebooks/                  # Análisis y modelado EP2
├── results/                    # Métricas y visualizaciones EP2
├── src/                        # Módulos de preprocesamiento EP2
├── tests/                      # Tests automatizados
│   ├── conftest.py
│   ├── test_etl.py             # 29 tests ETL
│   └── test_api.py             # 22 tests API
├── .env.example                # Plantilla de variables de entorno
├── .gitignore
├── requirements.txt
└── README.md
```

## Requisitos previos

- Python 3.12+
- Docker Desktop
- Git

## Instalación rápida

```bash
# 1. Clonar el repositorio
git clone https://github.com/tu-usuario/DataSet-Productos-Cosmeticos.git
cd DataSet-Productos-Cosmeticos

# 2. Crear entorno virtual e instalar dependencias
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt

# 3. Configurar variables de entorno
cp .env.example .env
# Editar .env con tus credenciales

# 4. Levantar la base de datos
cd docker
docker compose up -d db
cd ..

# 5. Correr el pipeline ETL (carga inicial)
python -m etl.pipeline --sample 10000

# 6. Levantar la API
uvicorn api.main:app --reload

# 7. Levantar el dashboard (en otra terminal)
streamlit run dashboards/app.py
```

## Despliegue con Docker (completo)

```bash
cd docker
docker compose up --build -d
```

Servicios disponibles:
- **PostgreSQL**: `localhost:5432`
- **API REST**: `http://localhost:8000` — Swagger en `/docs`
- **Dashboard**: `http://localhost:8501`

## Fuentes de datos

| # | Fuente | Descripción |
|---|---|---|
| 1 | CSV local | `product_info.csv` + `reviews_*.csv` (~1M reseñas) |
| 2 | HuggingFace API | Análisis de sentimiento (`cardiffnlp/twitter-roberta-base-sentiment-latest`) |
| 3 | exchangerate-api.com | Tipo de cambio USD → CLP en tiempo real |
| 4 | PostgreSQL | Data warehouse con tablas limpias y resultados del modelo |

## Tests

```bash
# Tests ETL
pytest tests/test_etl.py -v     # 29 tests

# Tests API
pytest tests/test_api.py -v     # 22 tests

# Todos los tests
pytest tests/ -v                # 51 tests

# Guardar reporte
pytest tests/ -v > docs/reporte_pruebas.md
```

## API Endpoints

| Método | Endpoint | Descripción |
|---|---|---|
| GET | `/` | Health check |
| GET | `/products` | Lista de productos con filtros |
| GET | `/products/{id}` | Producto por ID |
| GET | `/reviews` | Reseñas con filtros |
| GET | `/reviews/sentiment/summary` | Análisis sentimiento vs recomendación |
| GET | `/metrics/model` | Métricas del modelo ML |
| GET | `/clusters` | Resumen clusters K-Means |
| GET | `/exchange-rate` | Tipo de cambio USD→CLP |

Documentación interactiva: `http://localhost:8000/docs`

## Variables de entorno

Ver `.env.example` para la lista completa. Las principales:

```env
POSTGRES_USER=sephora_user
POSTGRES_PASSWORD=changeme_123
POSTGRES_DB=sephora_dw
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
HF_API_KEY=hf_xxxxxxxxxxxx
EXCHANGE_API_URL=https://v6.exchangerate-api.com/v6/TU_KEY/latest/USD
```

## Dependencias principales

```
fastapi, uvicorn, streamlit, plotly
pandas, numpy, scikit-learn, joblib
sqlalchemy, psycopg2-binary
pandera, python-dotenv, requests
pytest, httpx
```

Ver `requirements.txt` para la lista completa con versiones.


Proyecto desarrollado para la asignatura **SCY1101 — Programación para la Ciencia de Datos**, Evaluación Parcial N°2, DuocUC 2026.