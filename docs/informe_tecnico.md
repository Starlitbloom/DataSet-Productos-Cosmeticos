# Informe Técnico: Análisis y Limpieza del Dataset Reseñas de productos y cuidado de la piel de Sephora

**Fecha:** Abril 15, 2026  
**Autoras:** Maria Calfileo, Rocio Cruces, Keiny Navarro
**Versión:** 1.0  

## Resumen Ejecutivo

Este informe presenta el análisis exploratorio, limpieza y transformación del dataset de reseñas de productos de cuidado de la piel de Sephora, obtenido en Kaggle. El dataset contiene multiple información sobre productos cosméticos, incluyendo marcas, nombres, tipos, descripciones e imágenes y sus reseñas. El proceso de preparación de datos incluyó la unificación de múltiples fuentes, eliminación de duplicados, tratamiento de valores nulos y valores inconsistentes, estandarización de variables textuales, codificación de variables categóricas y escalamiento de variables numéricas mediante un pipeline de machine learning reproducible.

El dataset final resultante está estructurado y optimizado para tareas de modelado predictivo, específicamente para la predicción de la probabilidad de recomendación de productos a partir de características del usuario y del producto.

## Análisis Exploratorio Inicial

### Estadísticas Descriptivas

- **Tamaño del dataset:** 6 filas × 122 columnas
- **Columnas principales:**
  - price_usd: Precio del producto (numérico continuo)
  - rating: Calificación otorgada al producto (escala numérica)
  - helpfulness: Nivel de utilidad de la reseña
  - total_feedback_count: Total de interacciones en la reseña
  - total_pos_feedback_count: Número de comentarios positivos
  - total_neg_feedback_count: Número de comentarios negativos

  - **Variables categóricas:**
  - brand_name: Marca del producto
  - skin_type: Tipo de piel del usuario
  - eye_color: Color de ojos
  - hair_color: Color de cabello

### Caracterización del Dataset

- **Distribución de recomendaciones:** La variable objetivo is_recommended presenta un comportamiento desbalanceado, con predominancia de valores positivos (recomendación).
- **Distribución de ratings:** Los valores de rating se concentran en el rango alto, indicando una tendencia general a evaluaciones positivas de los productos.
- **Frecuencia de marcas:** Las marcas más representadas en el dataset corresponden a aquellas con mayor volumen de reseñas, destacando e.l.f., MAC, Sephora y otras marcas de alta rotación
- **Variables demográficas:** Las características del usuario, como tipo de piel, color de ojos y color de cabello, presentan distribuciones heterogéneas, siendo skin_type la variable más influyente en la segmentación del dataset.
- **Nivel de detalle:** El dataset se encuentra a nivel de interacción usuario-producto, lo que genera múltiples registros por producto debido a la existencia de múltiples reseñas.

### Visualizaciones Iniciales

Se generaron histogramas y gráficos de barras para visualizar:
- Distribución de ratings
- Distribución de la variable objetivo (is_recommended)
- Distribución de precios (price_usd)
- Frecuencia de marcas:
- Distribución de variables del usuario


## Metodología de Transformación

### Técnicas Aplicadas

1. **Limpieza y preparación de datos:**
   - Eliminación de valores duplicados a nivel de interacción usuario-producto
   - Estandarización de valores inconsistentes en variables categóricas
   - Conversión de valores no válidos a nulos (NaN) para su posterior imputación

2. **Tratamiento de valores faltantes:**
   - Imputación de variables numéricas mediante mediana
   - Imputación de variables categóricas mediante moda
   - Eliminación de columnas con alto porcentaje de valores faltantes (≥90%)

3. **Transformación de variables:**
   - Escalamiento de variables numéricas mediante StandardScaler
   - Codificación de variables categóricas mediante One-Hot Encoding
   - Tratamiento de valores atípicos mediante método IQR (capping)

4. **Ingeniería de datos:**
   - Unificación de múltiples fuentes de datos (productos y reseñas)
   - Estandarización de nombres de columnas tras merge
   - Conversión de variables de texto a formato consistente (minúsculas, limpieza básica)

5. **Validación de datos:**
   - Verificación de integridad del dataset final
   - Confirmación de estructura consistente para entrenamiento de modelos
   - Preparación del dataset para tareas de clasificación supervisada

### Justificaciones de Decisiones

- **Eliminación de columnas con alta proporción de valores faltantes:** Se eliminaron variables con más del 90% de valores nulos para evitar ruido y reducir dimensionalidad sin pérdida significativa de información.
- **Estandarización de variables categóricas:** Se aplicó limpieza de texto para asegurar consistencia en variables como marcas y características del usuario, reduciendo variaciones innecesarias en los datos.
- **Tratamiento de valores nulos:** Se aplicaron estrategias diferenciadas según el tipo de variable:
   - Mediana para variables numéricas
   - Moda para variables categóricas
- **Eliminación de duplicados:** Se eliminaron registros duplicados a nivel de interacción usuario-producto para evitar sesgos en el entrenamiento del modelo.
- **Codificación y escalamiento:** Se aplicaron técnicas de transformación para convertir variables categóricas en numéricas (One-Hot Encoding) y estandarizar variables numéricas (StandardScaler), asegurando compatibilidad con modelos de machine learning.

## Resultados y Validación

### Dataset Transformado

- **Tamaño final:** 1097385 filas × 43 columnas
- **Columnas finales:** price_usd, rating, helpfulness, total_feedback_count, total_neg_feedback_count, total_pos_feedback_count, brand_name, skin_type, eye_color, hair_color (más variables categóricas generadas por One-Hot Encoding)
- **Valores nulos restantes:**
  El dataset final no presenta valores nulos tras el proceso de imputación, ya que:
  - Las variables numéricas fueron imputadas con la mediana
  - Las variables categóricas fueron imputadas con la moda
  - Las columnas con más del 90% de valores nulos fueron eliminadas previamente

### Evidencia de Efectividad

- **Integridad verificada:** No se detectaron registros duplicados tras la unificación y limpieza del dataset.
- **Consistencia mejorada:** Las variables categóricas fueron estandarizadas mediante limpieza de texto y normalización de valores.
- **Métricas adicionales:** Se eliminaron columnas con alto porcentaje de valores nulos, mejorando la calidad del dataset para modelamiento.
- **Archivo de salida:** El conjunto de datos procesado fue guardado en formato optimizado para análisis y entrenamiento de modelos.


### Validación de Calidad

- **Completitud:** Las variables críticas del modelo fueron completadas mediante estrategias de imputación (mediana para numéricas y moda para categóricas).
- **Exactitud:** Los tipos de datos fueron corregidos y estandarizados.
- **Consistencia:** Las variables categóricas fueron normalizadas y codificadas .
- **Unicidad:** Se eliminó la presencia de duplicados .

## Conclusiones y Recomendaciones

### Reflexión Final

El proceso de preprocesamiento permitió transformar un dataset con problemas de calidad de datos (valores nulos, inconsistencias en variables categóricas y diferencias de escala entre variables numéricas) en un conjunto de datos completamente preparado para modelamiento.
Las decisiones metodológicas se orientaron a preservar la mayor cantidad de información útil, eliminando únicamente variables redundantes o con alta proporción de valores faltantes.
La estructura modular del pipeline implementado permite su reutilización, escalabilidad y mantenimiento en futuros proyectos de análisis de datos o machine learning.

### Dificultades Encontradas

- Presencia de valores nulos en múltiples variables numéricas y categóricas, lo que requirió estrategias diferenciadas de imputación.
- Inconsistencias en variables categóricas, especialmente en nombres de marcas y atributos del usuario, lo que requirió normalización de texto.
- Diferencias de escala entre variables numéricas, lo que hizo necesario el uso de estandarización para evitar sesgos en el modelo.

### Sugerencias

1. **Mejora del dataset original:** CSe recomienda trabajar con fuentes de datos más completas o actualizadas, con menor proporción de valores faltantes, para reducir la necesidad de imputación intensiva.
2. **Análisis adicional:** Explorar otros algoritmos de machine learning como Gradient Boosting o XGBoost para mejorar el desempeño predictivo.
3. **Enriquecimiento:** Evaluar la creación de nuevas variables derivadas a partir de las existentes (por ejemplo, ratios entre métricas de interacción) para enriquecer el poder predictivo del modelo.
4. **Automatización:** Mantener el pipeline modular permite su reutilización en futuros proyectos, facilitando la automatización del proceso de limpieza y preprocesamiento de datos.

### Archivos Generados

- `notebooks/analisis_reseñas.ipynb`: Notebook completo con análisis, limpieza y modelamiento del dataset.
- `data/sephora_limpio.csv`: Dataset procesado listo para modelamiento.
- `src/transformers.py`: Implementación de transformadores personalizados reutilizables.
- `outputs/`: Resultados, métricas y visualizaciones generadas durante el análisis.

Este trabajo cumple con los requisitos de organización, calidad de análisis y buenas prácticas de codificación, proporcionando una base sólida para análisis posteriores de productos de Sephora.