# Guía de Instalación y Despliegue

## Requisitos del sistema

| Requisito | Versión mínima |
|---|---|
| Python | 3.12 |
| Docker Desktop | 4.0+ |
| Git | 2.30+ |
| RAM | 8 GB recomendado |
| Disco | 5 GB libres |

## Paso 1: Clonar el repositorio

```bash
git clone https://github.com/tu-usuario/DataSet-Productos-Cosmeticos.git
cd DataSet-Productos-Cosmeticos
```

## Paso 2: Entorno virtual y dependencias

```bash
# Crear entorno virtual
python -m venv venv

# Activar (Windows PowerShell)
venv\Scripts\activate

# Activar (macOS/Linux)
source venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt
```

## Paso 3: Variables de entorno

```bash
# Copiar plantilla
cp .env.example .env
```

Editar `.env` con tus valores reales:

```env
# PostgreSQL
POSTGRES_USER=sephora_user
POSTGRES_PASSWORD=tu_password_seguro
POSTGRES_DB=sephora_dw
POSTGRES_HOST=localhost
POSTGRES_PORT=5432

# HuggingFace (registro gratuito en huggingface.co)
HF_API_KEY=hf_xxxxxxxxxxxxxxxxxxxx
SENTIMENT_SAMPLE_SIZE=500

# Tipo de cambio (registro gratuito en exchangerate-api.com)
EXCHANGE_API_URL=https://v6.exchangerate-api.com/v6/TU_KEY/latest/USD
FALLBACK_USD_CLP=950.0

LOG_LEVEL=INFO
```

## Paso 4: Datos crudos

Colocar los archivos del dataset en `data/raw/`:

```
data/raw/
├── product_info.csv
├── reviews_0-250.csv
├── reviews_250-500.csv
└── reviews_500-end.csv
```

## Paso 5: Levantar PostgreSQL con Docker

```bash
cd docker
docker compose up -d db
docker compose ps    # verificar que está healthy
cd ..
```

## Paso 6: Ejecutar el pipeline ETL

```bash
# Carga inicial completa (puede tardar varios minutos)
python -m etl.pipeline

# Para desarrollo (muestra de 10.000 reseñas)
python -m etl.pipeline --sample 10000 --sentiment 100
```

El pipeline ejecuta: Extract → Validate → Transform → Load

## Paso 7: Levantar la API

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

Verificar: `http://localhost:8000/docs`

## Paso 8: Levantar el dashboard

```bash
# En una terminal separada
streamlit run dashboards/app.py
```

Verificar: `http://localhost:8501`

---

## Despliegue completo con Docker

Para levantar todos los servicios de una sola vez:

```bash
cd docker
docker compose up --build -d
docker compose ps
```

Servicios:

| Servicio | URL |
|---|---|
| PostgreSQL | `localhost:5432` |
| API REST | `http://localhost:8000/docs` |
| Dashboard | `http://localhost:8501` |

Para detener:

```bash
docker compose down
```

Para detener y eliminar los datos:

```bash
docker compose down -v
```

---

## Verificación de la instalación

```bash
# 1. Tests ETL
pytest tests/test_etl.py -v    # debe mostrar 29 passed

# 2. Tests API
pytest tests/test_api.py -v    # debe mostrar 22 passed

# 3. Health check de la API
curl http://localhost:8000/
# {"status": "ok", "database": "connected", "docs": "/docs"}
```

---

## Solución de problemas comunes

**Error: `docker compose up` no encuentra las variables de entorno**
```bash
# Asegurarse de que el .env esté en la carpeta docker/
cp .env docker/.env
```

**Error: `psycopg2` no instala en Windows**
```bash
pip install psycopg2-binary
```

**Error: Puerto 5432 ya en uso**
```bash
# Cambiar el puerto en .env
POSTGRES_PORT=5433
```

**El pipeline ETL tarda mucho**
```bash
# Usar muestra para desarrollo
python -m etl.pipeline --sample 5000 --sentiment 50
```
