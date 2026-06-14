# Proyecto de Modelado — Sephora Cosmetics Reviews

Solución de machine learning para predecir si un usuario recomendará un producto cosmético (`is_recommended`), integrando modelos supervisados, aprendizaje no supervisado, evaluación comparativa y optimización de hiperparámetros.

---

## Estructura del Proyecto

```
proyecto_modelado/
├── notebooks/
│   ├── 01_exploratory_analysis.ipynb       # EDA, visualizaciones y detección de patrones
│   ├── 02_supervised_modeling.ipynb        # Modelos supervisados con Scikit-learn
│   ├── 02_unsupervised_learning.ipynb      # K-Means, PCA y análisis de clusters
│   ├── 03_model_evaluation.ipynb           # Evaluación comparativa y métricas
│   ├── 04_hyperparameter_optimization.ipynb# GridSearchCV y RandomizedSearchCV
│   └── 05_final_analysis.ipynb             # Integración y análisis final
├── src/
│   ├── data_preprocessing.py               # Limpieza, transformación y pipeline de datos
│   ├── model_training.py                   # Definición y entrenamiento de modelos
│   ├── model_evaluation.py                 # Funciones de evaluación y comparación
│   ├── hyperparameter_tuning.py            # GridSearchCV y RandomizedSearchCV
│   ├── pipeline.py                         # Pipeline scikit-learn completo
│   ├── transformers.py                     # Transformadores personalizados
│   ├── paths.py                            # Rutas centralizadas del proyecto
│   └── utils.py                            # Utilidades generales
├── data/
│   ├── raw/                                # Datos originales sin modificar
│   └── processed/                          # Artefactos preprocesados (.pkl)
├── models/
│   └── trained_models/                     # Modelos serializados con joblib
├── results/
│   ├── metrics/                            # Métricas en JSON y TXT
│   ├── plots/                              # Gráficos generados
│   └── reports/                            # Classification reports
├── main.py                                 # Orquestador end-to-end
└── README.md
```

---

## Requisitos

- Python 3.10 o superior
- Las dependencias se instalan con:

```bash
pip install -r requirements.txt
```

Dependencias principales:

| Librería | Uso |
|---|---|
| `scikit-learn` | Modelos, pipelines, métricas, búsqueda de hiperparámetros |
| `pandas` | Manipulación de datos |
| `numpy` | Operaciones numéricas |
| `matplotlib` / `seaborn` | Visualizaciones |
| `joblib` | Serialización de modelos y paralelismo |
| `jupyter` | Ejecución de notebooks |

---

## Reproducibilidad

Todos los experimentos usan `random_state=42`. Los artefactos de datos procesados se guardan en `data/processed/` y los modelos en `models/trained_models/`, de modo que cualquier notebook puede cargarse de forma independiente sin re-ejecutar etapas anteriores.

Para garantizar resultados idénticos:
1. Usar la misma versión de librerías.
2. No modificar los archivos `.pkl` en `data/processed/`.
3. Ejecutar los notebooks en orden numérico la primera vez.

---

## Ejecución

### Opción A — Pipeline completo desde terminal

Ejecuta preprocesamiento, entrenamiento, optimización y evaluación en un solo paso:

```bash
python main.py
```

### Opción B — Notebooks en orden

```bash
jupyter notebook
```

Ejecutar en el siguiente orden:

1. `01_exploratory_analysis.ipynb`
2. `02_supervised_modeling.ipynb` y `02_unsupervised_learning.ipynb` (independientes entre sí)
3. `03_model_evaluation.ipynb`
4. `04_hyperparameter_optimization.ipynb`
5. `05_final_analysis.ipynb`

---

## Modelos Implementados

| Modelo | Tipo | Justificación |
|---|---|---|
| Logistic Regression | Supervisado — baseline lineal | Interpretable, rápido; sirve como punto de referencia para cuantificar la ganancia de los modelos más complejos |
| Random Forest | Supervisado — ensamble bagging | Robusto ante outliers, captura relaciones no lineales, expone importancia de variables |
| Gradient Boosting | Supervisado — ensamble boosting | Mayor precisión en clasificación tabular; optimizado con GridSearchCV y RandomizedSearchCV |
| K-Means | No supervisado — clustering | Segmentación de productos/usuarios para análisis exploratorio |
| PCA | No supervisado — reducción dimensional | Visualización de clusters y análisis de varianza explicada |

---

## Optimización de Hiperparámetros

Se implementaron dos estrategias en `src/hyperparameter_tuning.py`:

- **GridSearchCV**: búsqueda exhaustiva sobre un grid adaptativo (se reduce en datasets > 50.000 muestras para controlar el tiempo de cómputo).
- **RandomizedSearchCV**: búsqueda aleatoria sobre un espacio más amplio, útil como exploración rápida o cuando el grid es demasiado grande.

Ambos métodos usan `scoring="f1_weighted"` y `cv=3`, y devuelven un diccionario uniforme con `best_model`, `best_params`, `best_score` y `cv_results` para facilitar la comparación.

---

## Resultados

Los resultados se guardan automáticamente en:

- `results/metrics/` — métricas en formato JSON y TXT por modelo
- `results/plots/` — matrices de confusión, curvas ROC y gráficos comparativos
- `results/reports/` — classification reports detallados

---

## Autores

Proyecto desarrollado para la asignatura **SCY1101 — Programación para la Ciencia de Datos**, Evaluación Parcial N°2, DuocUC 2025.
