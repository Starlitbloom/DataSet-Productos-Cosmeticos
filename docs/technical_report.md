# Technical Report: Cosmetics Products Dataset

## Overview
Este reporte técnico documenta el análisis exploratorio de datos (EDA) y el procesamiento de un conjunto de datos de productos cosméticos. El proyecto incluye la adaptación de notebooks de ejemplo para el contexto específico del dataset, la implementación de un pipeline modular de preprocesamiento utilizando transformadores personalizados de scikit-learn, y la creación de módulos para auditoría de datos, optimización de memoria y orquestación del flujo completo.

El objetivo principal es preparar los datos para tareas de aprendizaje automático, asegurando integridad, limpieza y transformación adecuada de características numéricas y categóricas.

## Data Source
- **Datos crudos**: `data/raw/products.csv`
  - Columnas originales: ID, brand, name, product_type, img, rating, dupes, description, shade_img, price_site, view_count
  - Formato: CSV con separadores automáticos detectados por pandas
  - Tamaño aproximado: 12.4 MB antes de optimización

- **Datos procesados**: `data/processed/processed_data.csv`
  - Dimensiones finales: 10,667 filas × 545 columnas
  - Características: view_count (numérica escalada) + codificación one-hot para brand y product_type
  - Optimización de memoria: Reducción del 1.1% en uso de memoria

## Methodology

### 1. Auditoría de Datos (`src/audit.py`)
- Verificación de integridad mediante hash SHA-256
- Creación/validación de metadatos en `metadata.json`
- Detección de alteraciones en archivos crudos

### 2. Optimización de Memoria (`src/optimization.py`)
- Downcasting de tipos numéricos (int64 → int32, float64 → float32)
- Procesamiento en chunks para archivos grandes
- Reducción de footprint de memoria sin pérdida de información

### 3. Pipeline de Preprocesamiento (`src/pipeline.py`)
El pipeline utiliza transformadores personalizados implementados en `src/transformers.py`:

1. **Eliminación de columnas de fuga**: ID, img, name, description, shade_img, dupes, price_site
2. **Limpieza de texto**: Eliminación de espacios y saltos de línea en columnas categóricas
3. **Conversión de 'unknown' a NaN**: Normalización de valores faltantes
4. **Eliminación de columnas con alto porcentaje de nulos**: Threshold del 80%
5. **Imputación inteligente**: Mediana para numéricas, moda para categóricas
6. **Capping de outliers**: Método IQR para variables numéricas
7. **Eliminación de varianza cero**: Columnas constantes
8. **Escalado y codificación**:
   - Escalado estándar para numéricas
   - One-hot encoding para categóricas (brand, product_type)

### 4. Orquestación (`main.py`)
Script principal que coordina:
- Auditoría → Carga → Optimización → Preprocesamiento → Guardado
- Manejo de errores robusto con mensajes en español
- Logging de progreso y dimensiones finales

## Results

### Estadísticas del Dataset
- **Registros totales**: 10,667 productos cosméticos
- **Características finales**: 545 (1 numérica + 544 categóricas codificadas)
- **Marcas únicas**: Más de 300 marcas representadas
- **Tipos de producto**: Diversidad completa (labios, ojos, rostro, uñas, etc.)

### Optimización de Memoria
- Uso original: 12.40 MB
- Uso optimizado: 12.26 MB
- Ahorro: 1.1%

### Hallazgos Clave
- El dataset contiene productos de marcas premium y masivas
- Alta variabilidad en tipos de producto con presencia de saltos de línea en datos crudos
- view_count como métrica principal de engagement
- Limpieza efectiva eliminó ruido textual y normalizó categorías

## Conclusion
El pipeline de procesamiento se ejecutó exitosamente, generando un dataset limpio y optimizado listo para modelado de machine learning. La arquitectura modular permite fácil mantenimiento y extensión para nuevos requerimientos de preprocesamiento. Los datos procesados preservan la información relevante mientras eliminan ruido y características de fuga, proporcionando una base sólida para análisis predictivos en el dominio de productos cosméticos. 

Próximos pasos recomendados: Exploración de correlaciones, modelado de recomendación basado en marcas/tipos de producto, o análisis de tendencias de engagement. 