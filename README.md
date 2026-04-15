# Sephora Products and Skincare Reviews Dataset Project
Este repositorio contiene un pipeline para un dataset de productos cosméticos (500MB+). El sistema está diseñado para auditar la integridad de los datos, optimizar el uso de memoria RAM y transformar variables complejas para futuros modelos.

## Project Structure
cosmetics_project/
├── data/
│   ├── raw/            # Datasets originales (Gestionados con Git LFS)
│   └── processed/      # Datos limpios y transformados en formato CSV
├── docs/               # Reportes técnicos y hallazgos de auditoría
├── notebooks/          # Jupyter Notebooks para Análisis Exploratorio 
├── src/                # Código fuente modular
│   ├── __init__.py
│   ├── audit.py        # Validación de firmas e integridad
│   ├── utils.py        # Carga masiva y limpieza de strings
│   ├── memory.py       # Módulo de optimización de tipos de datos
│   └── transformers.py # Scikit-Learn Transformers 
├── main.py             # Script maestro de organización
├── requirements.txt    # Dependencias (Pandas, Scikit-Learn, etc.)
└── README.md           # Instrucciones del proyecto

## Setup Instructions
1. Clonar el repositorio y configurar Git LFS
Dado que los archivos de datos superan los 100MB, este proyecto utiliza Git LFS.
git clone https://github.com/tu-usuario/cosmetics_project.git
cd cosmetics_project
git lfs install
git lfs pull

2. Entorno Virtual
python -m venv .venv
# En Windows:
.venv\Scripts\activate

3. Instalación de dependencias
pip install -r requirements.txt

# Pipeline de Procesamiento
El script main.py ejecuta automáticamente las siguientes etapas:

Auditoría: Verifica que los archivos CSV coincidan con los hashes SHA-256 registrados.
Carga Optimizada: Une los fragmentos de datos y reduce el uso de memoria (reducción estimada del 40-60%).
Limpieza Estructural: Normalización de nombres de columnas y limpieza de caracteres especiales.
Pipeline de Transformación:
Tratamiento de valores "Unknown".
Capping de Outliers (método IQR).
Imputación inteligente de valores nulos.
Escalado de variables numéricas y codificación categórica.

# Ejecución
Para procesar el dataset completo y generar el archivo final en data/processed/:
python main.py
