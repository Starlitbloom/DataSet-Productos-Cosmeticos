# Manual de Usuario — Sephora Intelligence Dashboard

## Descripción general

Sephora Intelligence es un dashboard interactivo que presenta análisis de datos sobre productos y reseñas cosméticas de Sephora US. Está diseñado para tres tipos de audiencia con necesidades distintas.

## Acceso al dashboard

Con el sistema corriendo, abrir en el navegador:

```
http://localhost:8501
```

---

## Navegación

El sidebar izquierdo permite moverse entre las 4 secciones:

| Sección | Audiencia | Contenido |
|---|---|---|
| **Inicio** | Todos | Resumen del proyecto y acceso a vistas |
| **Vista Ejecutiva** | Gerencia / negocio | KPIs, precios en CLP, top marcas |
| **Vista Técnica** | Data scientists | Métricas ML, clustering, PCA |
| **Vista Operativa** | Analistas / operaciones | Filtros interactivos, sentimiento |

---

## Vista Ejecutiva 💼

Diseñada para tomadores de decisiones sin perfil técnico.

### Indicadores clave (KPIs)
- **Catálogo**: total de productos disponibles
- **Precio promedio USD / CLP**: precio medio del catálogo en ambas monedas
- **Rating**: calificación promedio de productos (escala 1-5)
- **Tasa de recomendación**: porcentaje de compradores que recomienda el producto

> El tipo de cambio USD→CLP se obtiene en tiempo real desde exchangerate-api.com y se muestra bajo los KPIs.

### Top 10 Marcas más Amadas
Gráfico de barras horizontal con las marcas que tienen mayor promedio de "loves" (indicador de popularidad en Sephora). Las barras más oscuras representan mayor popularidad.

### Productos por Categoría
Gráfico de dona mostrando la distribución del catálogo entre categorías: Makeup, Skincare, Hair, Fragrance, etc.

### Distribución de Precios en CLP
Histograma de frecuencia de precios y tabla comparativa por categoría con precio mínimo, máximo y promedio.

### Insights automáticos
Al final de la página, cuatro conclusiones automáticas destacando los hallazgos más relevantes del catálogo.

---

## Vista Técnica 🔬

Diseñada para el equipo de datos y ciencia de datos.

### Modelo Supervisado
- **Modelo utilizado**: Gradient Boosting Classifier (mejor modelo del EP2)
- **Parámetros**: n_estimators, max_depth, learning_rate, random_state
- **Descripción**: el modelo predice si un usuario recomendará o no un producto

### Clustering K-Means
Resultados del análisis no supervisado del EP2:

- **K óptimo**: número de clusters seleccionado por Silhouette Score
- **Silhouette Score**: métrica de calidad del clustering (mayor = mejor)
- **Inercia**: suma de distancias intra-cluster (menor = más compacto)

Los dos gráficos muestran cómo varían estas métricas para K = 2 a 10, con una estrella marcando el K seleccionado.

### Reducción de Dimensionalidad (PCA)
- Varianza explicada por los primeros 2 componentes principales
- Número de componentes necesarios para capturar el 90% y 95% de la información

---

## Vista Operativa ⚙️

Diseñada para analistas que trabajan con el catálogo día a día.

### Filtros interactivos

| Filtro | Descripción |
|---|---|
| **Categoría** | Filtra por categoría principal (Makeup, Skincare, Hair...) |
| **Marca** | Filtra por nombre de marca. Se actualiza según la categoría seleccionada |
| **Rango de precio** | Slider para definir precio mínimo y máximo en USD |

Los KPIs y gráficos se actualizan automáticamente al cambiar los filtros.

### Gráficos

**Rating vs Precio**: scatter plot que muestra la relación entre precio y calidad percibida, coloreado por categoría. Al pasar el cursor sobre un punto se muestra el nombre del producto y la marca.

**Top marcas (filtrado)**: ranking de marcas más populares dentro de los filtros aplicados.

### Análisis de Sentimiento
Resultados del análisis de sentimiento realizado sobre 500 reseñas usando la API de HuggingFace (`cardiffnlp/twitter-roberta-base-sentiment-latest`):

- **Positivo**: reseñas con tono favorable
- **Negativo**: reseñas con tono desfavorable
- **Neutral**: reseñas sin carga emocional clara

El panel derecho muestra el porcentaje de reseñas positivas y su relación con la tasa de recomendación.

### Catálogo de Productos
Tabla interactiva con los 100 productos más populares según los filtros aplicados. Columnas: nombre, marca, categoría, precio USD, rating y loves.

---

## Preguntas frecuentes

**¿Por qué el precio aparece en CLP?**
El dashboard consume la API de exchangerate-api.com en tiempo real para convertir los precios originales en USD a pesos chilenos.

**¿Por qué solo hay 500 reseñas analizadas con sentimiento?**
El análisis de sentimiento requiere llamadas a una API externa. Se usa una muestra estratificada de 500 reseñas para mantener el tiempo de procesamiento razonable y respetar los límites de la API gratuita.

**¿Los datos se actualizan en tiempo real?**
Los datos del dashboard se cachean por 5 minutos para mejorar el rendimiento. Para forzar una actualización, presiona `R` en el teclado o usa el botón de recarga de Streamlit.

**¿Puedo exportar los datos?**
Las tablas del dashboard permiten descargar los datos como CSV usando el botón de descarga que aparece al pasar el cursor sobre la tabla.
