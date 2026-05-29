## Informe Técnico: Arquitectura Predictiva, Segmentación No Supervisada y Optimización de Modelos en Ecosistemas 
Asignatura: Programación para la Ciencia de Datos  
 Fecha: 28 de Mayo, 2026
   Institución: Duoc UC 
   Carrera: Ingeniería en Informática   
   Autoras: Maria Calfileo, Rocio Cruces, Keiny Navarro  
   Equipo: 8  
   Versión: 2.0

## 1. Resumen Ejecutivo
El presente informe documenta el diseño, desarrollo e implementación de un ecosistema analítico y predictivo avanzado sobre el catálogo masivo y las reseñas de productos de cuidado de la piel (skincare) de Sephora. A partir de un volumen inicial crudo superior al millón de registros proveniente de extracciones web (scraping), el proyecto evolucionó desde una curación de datos básica hacia la construcción de una infraestructura industrial de aprendizaje automatizado.  

 **Los objetivos principales se centraron en:**
   -Resolver problemas críticos de integridad estructural, valores faltantes generalizados e inconsistencias textuales mediante la creación de un Pipeline modular reproducible.
   -Identificar agrupaciones y patrones latentes en los perfiles de los consumidores utilizando técnicas de Aprendizaje No Supervisado (K-Means, Clustering Jerárquico y PCA).
   -Desarrollar un sistema clasificador supervisado robusto para predecir la probabilidad de recomendación de un producto (is_recommended).

Los resultados finales determinaron que el preprocesamiento limpio de los datos fue el factor determinante del éxito del sistema. El modelo clasificador final alcanzó una exactitud del 96.28%y un F1-Score del 97.75% en el conjunto de prueba, demostrando una convergencia óptima y una alta capacidad de generalización. El artefacto analítico resultante fue serializado y persistido para permitir un despliegue en producción ágil y escalable.

## 2. Marco Metodológico
La construcción de la arquitectura analítica se fundamentó en la separación estricta de responsabilidades del software y en el cumplimiento del contrato de diseño de Scikit-Learn. El flujo de datos se estructuró en capas desacopladas y modulares (src/) coordinadas por un script orquestador central (main.py) para mitigar de forma absoluta el riesgo de data leakage (fuga de datos) y asegurar la reproducibilidad científica en entornos de producción.

                [ Dataset Unificado de Ingesta Masiva ]
                                   │
                    ┌──────────────┴──────────────┐
                    ▼                             ▼
              [ Train Set ]                 [ Test Set ]
          (739,532 Obs. Totales)        (184,884 Obs. Totales)
                    │                             │
             .fit_transform()                 .transform()
                    │                             │
                    └──────────────┬──────────────┘
                                   ▼
                       [ Modelado Predictivo ]

**2.1. Protocolo de Aislamiento de Datos (Split Train/Test)**
Para garantizar una evaluación inmune al sesgo de selección y simular una puesta en producción real, el dataset unificado fue sometido a una partición estricta a través de la función modular (load_and_split_data()) parametrizada en (src/utils.py):
  - Conjunto de Entrenamiento (Train Set): Correspondiente al 80% del volumen total de los datos procesados (739,532 registros), destinado exclusivamente al ajuste iterativo de los transformadores personalizados y al aprendizaje de los algoritmos.
  - Conjunto de Prueba (Test Set): Correspondiente al 20% restante (184,884 registros), completamente aislado del flujo de entrenamiento y reservado para la evaluación ciega del rendimiento y generalización final del modelo.

**2.2. Mitigación del Data Leakage mediante Inferencia Pasiva**
El pilar metodológico del software exige que el Pipeline ejecute el método (.fit_transform()) de manera exclusiva sobre el conjunto de Train. De este modo, las clases de preprocesamiento localizadas en (src/data_preprocessing.py) tales como (SmartImputerTransformer) u (OutlierCapper) aprenden las estadísticas del negocio —medianas numéricas, modas categóricas y límites de truncamiento del rango intercuartílico IQR— usando únicamente la distribución de entrenamiento.
El conjunto de Test es procesado de forma estrictamente pasiva empleando el método (.transform()), aplicando los parámetros precalculados sin alterarlos. Esto erradica el fenómeno de data leakage, impidiendo que el conocimiento de la distribución del conjunto de prueba contamine el proceso de optimización del algoritmo.

**2.3. Muestreo Estratificado para Clases Desbalanceadas**
Debido a la naturaleza comercial del dataset, la variable objetivo (is_recommended) presenta un comportamiento asimétrico severo. El software controla este fenómeno aplicando un split estrictamente estratificado (stratify=y), lo que obliga a que tanto el set de Train como el de Test conserven con precisión matemática la proporción del 83.99% de recomendaciones positivas frente al 16.00% de rechazos, blindando la validez estadística de las métricas de evaluación frente a datos sesgados.

## 3. Análisis Experimental

**3.1. Infraestructura y Configuración del Entorno**
El desarrollo se ejecutó utilizando Python 3.12 sobre un entorno controlado que automatiza dinámicamente el ruteo de archivos. El sistema detecta de forma transparente si la ejecución se realiza en un servidor de computación en la nube (Google Colab) o (Visual Studio Code), parametrizando las rutas de acceso a los directorios del repositorio de GitHub (../data/raw/).

**3.2. Capa de Ingesta y Unificación Estructural**

La ingesta se diseñó para consolidar de forma eficiente la base de datos cruda: 
 1-Se cargó el catálogo maestro de artículos comerciales (product_info.csv), registrando un total de 8,494 productos cosméticos únicos. 

 2-Se unificaron secuencialmente en un único DataFrame de comentarios masivos las 5 fuentes independientes de opiniones de usuarios (reviews_0-250.csv hasta reviews_1250-end.csv), consolidando un gran volumen de 1,094,411 filas.

 3-Se realizó un cruce de datos hacia la izquierda (how='left') utilizando la variable indexatoria product_id como clave de unión. Esta operación preservó la totalidad de las experiencias de los usuarios, inyectando los atributos del catálogo (marca, precio, ingredientes) a cada reseña individual. Al cargar las fuentes de datos, se incluyó el parámetro na_values=['NULL', 'null', ''], forzando a que las cadenas de texto vacías o placeholders de la base de datos fueran interpretadas de forma nativa por Pandas como valores faltantes reales (NaN).

**3.3. Auditoría del Dataset Crudo y Diagnóstico de Nulos**
Al finalizar el merge, el volumen de datos consolidado registró 45 columnas iniciales. La fase de auditoría técnica aplicó las funciones estadísticas .info() y .describe() para calcular la tasa exacta de valores faltantes:

=== Resumen de Valores Nulos Detectados (Muestra de Auditoría) ===
Variable                      Cantidad de Nulos        Porcentaje (%)
─────────────────────────────────────────────────────────────────────
variation_desc                     1,086,128                99.24%
sale_price_usd                     1,084,658                99.11%
value_price_usd                    1,063,534                97.18%
child_max_price                      641,008                58.57%
helpfulness                          561,592                51.31%
review_title                         310,654                28.39%
skin_type                            111,557                10.19%
review_text                            1,444                 0.13%

Para diagnosticar visualmente la distribución de la ausencia de información sin saturar el procesador, se extrajo una muestra aleatoria representativa de 50,000 registros, graficando un mapa de calor (heatmap) de nulos.

El mapa de calor reveló que las ausencias responden a dos comportamientos bien definidos:

-Vacíos Estructurales Sólidos (Tasa > 90%): Las variables variation_desc y sale_price_usd se muestran casi completamente vacías. Esto demuestra que corresponden a atributos específicos que no aplican a la inmensa mayoría de los productos del catálogo (por ejemplo, artículos que no están en oferta).
-Vacíos de Interacción de Usuario (Tasa entre 10% y 51%): Las columnas del perfil demográfico (skin_type, hair_color) y de retroalimentación (helpfulness) presentan un patrón intermitente aleatorio. Esto se explica porque completar estos campos es opcional para el cliente al momento de registrar su reseña.

**3.4. Arquitectura Definitiva del Pipeline Modular**
Para automatizar la transformación de los datos salvajes hacia una matriz numérica limpia, se implementó una arquitectura orientada a objetos en el (script src/transformers.py), heredando de las clases base de Scikit-Learn.

El flujo orquestado en (pipeline) se configuró de forma secuencial con las siguientes estaciones de procesamiento:

[df_raw] ──► DropColumns ──► ColumnRenamer ──► UnknownToNaN ──► TextCleaner ──► PriceCleaner ──► DropHighMissing ──► SmartImputer ──► [ColumnTransformer]
  -1. DropColumnsTransformer
 Elimina de la entrada del sistema los metadatos irrelevantes y los identificadores que no aportan valor predictivo (como Unnamed: 0, id, img, product_name_y, ingredients, highlights), reduciendo la sobrecarga de memoria desde el inicio del flujo.

  -2. ColumnRenamerTransformer
 Corrige y estandariza los nombres de las columnas que sufrieron duplicaciones tras el merge, renombrando de forma limpia los sufijos automáticos de Pandas (ej. convierte price_usd_x a price_usd).

  -3. UnknownToNaNTransformer
 Sustituye variaciones de texto como 'unknown', 'UNKNOWN' o cadenas vacías por valores nulos matemáticos reales de NumPy (np.nan), permitiendo que las fases de imputación estadística actúen sobre ellos.

  -4. TextCleanerTransformer
 Aplica operaciones de normalización sobre las columnas de texto tipo objeto. Fuerza la conversión a minúsculas (.str.lower()), remueve los espacios en blanco residuales en los extremos (.str.strip()), unifica múltiples espacios internos (\s+) y elimina caracteres especiales o símbolos corruptos mediante expresiones regulares. Esto garantiza que el motor de codificación posterior no cree categorías duplicadas por errores de digitación (por ejemplo, agrupando de forma idéntica "Clinique " y "clinique").

  -5. PriceCleanerTransformer
 Transformador especializado que limpia las columnas financieras mediante expresiones regulares, eliminando el símbolo de moneda $ y las comas de separación, convirtiendo la característica de tipo string a tipo numérico continuo float.

  -6. DropHighMissingTransformer (Umbral = 0.90)
 Calcula la proporción de nulos por columna en el método fit. Si una variable supera el 90% de datos faltantes, se almacena en una lista interna y se elimina de la matriz en el método transform. Esto descartó de forma automática aquellas variables que solo aportaban ruido.

  -7. SmartImputerTransformer (Umbral = 0.10)
 Clasifica dinámicamente las variables con nulos remanentes en dos grupos en función de su gravedad estadística:
   -Columnas Simples (<10% de nulos): Imputadas mediante estrategias tradicionales eficientes, utilizando la mediana para variables numéricas (por su robustez ante asimetrías) y la moda para las categóricas.

   -Columnas Complejas (>10% de nulos): Aísla las variables demográficas del usuario que presentan pérdidas masivas aleatorias. Incorpora además una regla de negocio crítica: los valores nulos en la métrica (helpfulness) son rellenados con un valor constante de 0, asumiendo de forma lógica que corresponden a comentarios nuevos que aún no han recibido interacciones por parte de la comunidad.

   -8. Orquestación del ColumnTransformer (Bifurcación en Paralelo)
 Al final del pipeline principal, se instanció un ColumnTransformer que actúa como un enrutador dinámico de dos rutas paralelas que se ejecutan simultáneamente sobre las columnas limpias:
  Ruta Numérica(num_pipe): Envía las características numéricas continuas a través de tres componentes secuenciales:

    1-OutlierCapper: Calcula los percentiles 25% (Q1) y 75% (Q3) de cada columna numérica. Establece los límites estadísticos mediante la regla del Rango Intercuartílico (IQR = Q3 - Q1):

             {Límite Inferior} = Q1 - 1.5 x IQR
             {Límite Superior} = Q3 + 1.5 x IQR

    Luego, aplica la función np.clip() sobre los datos, truncando o redondeando cualquier valor extremo hacia los límites permitidos. Esto estabiliza la varianza sin destruir filas de datos.

    2-DropZeroVarianceTransformer: Evalúa la desviación estándar de las columnas numéricas. Si una variable presenta una varianza igual a cero, significa que es un valor constante que no aporta información predictiva, procediendo a su eliminación automática.

    3-StandardScaler: Transforma la escala de las variables para que tengan una media de 0 y una desviación estándar de 1. Esto unifica las magnitudes de características dispares (como precios en dólares versus conteos de interacción), asegurando que ninguna variable domine artificialmente los cálculos euclidianos de los modelos.

 Ruta Categórica(cat_pipe): Procesa los descriptores textuales mediante un componente:
   1-OneHotEncoder: Expande las variables de texto en columnas binarias independientes compuestas por ceros y unos, fijando un límite controlado de (max_categories=20) para evitar una explosión dimensional inmanejable. El parámetro (handle_unknown='ignore') asegura que si en datos futuros ingresa una marca o categoría que el modelo nunca vio en el entrenamiento, el software no sufra un fallo crítico (crash), sino que simplemente la ignore de manera segura asignando ceros en esa fila.

   -9. Preservación del Formato DataFrame de Pandas
 Al cierre del constructor, se configuró la instrucción global (full_pipeline.set_output(transform="pandas")). Esto obliga a Scikit-Learn a retener la estructura nativa de Pandas a la salida del pipeline, entregando una matriz limpia con nombres de columnas legibles y trazables, en lugar de matrices opacas de NumPy. El dataset procesado final consolidó exitosamente 1,097,385 filas y 44 variables listas para producción.

## 4. Aprendizaje No Supervisado: Segmentación Estructurada
Para identificar perfiles de consumidores y descubrir agrupaciones naturales dentro de las interacciones de Sephora sin la guía de una variable objetivo, se desplegó un flujo completo de Aprendizaje No Supervisado.

**4.1. Reducción de Dimensionalidad mediante PCA**
Dada la alta cantidad de variables resultantes tras la binarización de categorías del pipeline, resulta inviable modelar o visualizar directamente el espacio geométrico de los datos. Para resolver esto, se aplicó un Análisis de Componentes Principales (PCA), transformando las variables correlacionadas en un nuevo conjunto de componentes ortogonales perpendiculares.
    -Análisis de Varianza Explicada: Los resultados matemáticos arrojaron que el primer componente principal (PC1) captura el 35.01% de la varianza total del dataset, mientras que el segundo componente (PC2) captura el 12.56%.
    -En conjunto, la proyección en un plano bidimensional retiene el 47.57% de la varianza e información total de la matriz de características original.
    -Control del Overplotting: Para generar la visualización gráfica de forma eficiente y evitar la saturación visual por superposición de puntos (overplotting), se extrajo una muestra aleatoria de 10,000 observaciones de la matriz transformada.

La visualización reveló una estructura sumamente interesante: los puntos se distribuyen en patrones lineales continuos y "bandas" verticales perfectamente paralelas. En la ingeniería informática, este comportamiento evidencia la fuerte influencia de las variables discretas binarizadas por el OneHotEncoder. Al tomar valores fijos de 0 y 1, estas columnas restringen la transformación lineal del PCA, obligando a los vectores a proyectarse sobre subplanos paralelos estructurados.

**4.2. Búsqueda del Número Óptimo de Clusters**
Para definir en cuántos segmentos se divide de forma ideal la comunidad de usuarios de Sephora, se entrenó el algoritmo particional K-Means sobre una muestra de optimización de 50,000 registros, contrastando el comportamiento de dos métricas independientes sobre un rango de k de 2 a 10:

=== Resultados de la Auditoría de Parámetros de Clustering ===
Número de Clusters (k)          Inertia (Inercia Total)        Silhouette Score
───────────────────────────────────────────────────────────────────────────────
k = 2                              315,189.80                       0.310007
k = 3                              274,444.25                       0.209474
k = 4                              253,396.40                       0.159835
k = 5                              236,062.79                       0.165141
k = 6                              219,327.70                       0.170197
k = 7                              212,956.96                       0.165024
k = 8                              205,123.00                       0.136940

 -Criterio 1: Método del Codo (Elbow Method): La inercia mide la compactación interna de los grupos calculando la suma de las distancias euclidianas de los puntos a sus respectivos centroides. Al graficar la inercia respecto a k, se observó un descenso pronunciado inicial que cambia bruscamente de dirección en k = 6, aplanándose notablemente a partir de ese umbral. Esto demuestra que establecer más de 6 clusters complejiza el software sin aportar una ganancia real en la reducción de la varianza interna.

 -Criterio 2: Coeficiente de Silueta (Silhouette Score): Esta métrica evalúa simultáneamente la cohesión interna y la distancia de separación entre clusters contiguos, donde valores cercanos a 1 indican una asignación perfecta. La tabla determinó de forma contundente que el valor máximo absoluto se alcanza en k = 2 con un score de 0.310007. A partir de k = 3, el score decae de forma severa, evidenciando que los grupos adicionales comienzan a solaparse entre sí.

**4.3. Entrenamiento Final de K-Means y Mapeo Dimensional**
Para asegurar la estabilidad global del agrupamiento y evitar sesgos de solapamiento, se adoptó la estructura binaria recomendada por el coeficiente de Silueta (k = 2) como la resolución definitiva del modelo. El entrenamiento final sobre la matriz completa asignó las etiquetas correspondientes, distribuyendo las interacciones en dos grandes macrogrupos de comportamiento del mercado:

   -Cluster 0 (Grupo Mayoritario): Concentra 809,670 registros, caracterizados en el espacio de PCA por posicionarse a la izquierda del umbral de corte horizontal fijado en el valor 1 del eje PC1.

   -Cluster 1 (Grupo Minoritario): Registra 287,715 registros, extendiéndose hacia la derecha a lo largo del eje dominante PC1.

La separación de colores en el Scatter Plot ratifica que el Componente Principal 1 es el factor matemático que gobierna la diferenciación absoluta de los perfiles de los clientes dentro de la plataforma de Sephora.

## 5. Modelado Supervisado y Resultados Comparativos
La fase predictiva abordó la construcción de clasificadores supervisados para determinar si un usuario recomendará un producto (is_recommended) en función de las variables numéricas y categóricas limpias del negocio.

**5.1. Validación Cruzada Robusta**
(Cross-Validation en Train) Para obtener evaluaciones estables e independientes de la aleatoriedad de las particiones, el conjunto de entrenamiento de Train (33,706 filas) fue sometido a una Validación Cruzada Estratificada de 5 pliegues. El motor de ejecución dividió los datos iterativamente, utilizando 4 pliegues para ajustar los algoritmos y 1 pliegue como set de control interno, promediando el rendimiento al finalizar los 5 ciclos.

  Se evaluaron de forma competitiva tres familias de clasificadores empleando hiperparámetros de penalización balanceada para controlar el desbalanceo:

    -Regresión Logística (Logistic Regression): Optimizada con un límite extendido de 3,000 iteraciones para asegurar convergencia euclidiana.

    -Bosques Aleatorios (Random Forest Classifier): Ensamble baggizado compuesto por 100 árboles de decisión independientes ejecutados en paralelo.

    -Aumento de Gradiente (Gradient Boosting Classifier): Ensamble secuencial basado en la optimización de los residuos de los árboles previos.

                   ==Tabla Comparativa de Desempeño en Validación Cruzada (Train Set)==
───────────────────────────────────────────────────────────────────────────────────────────────────────────────
Modelo Evaluado         Accuracy Medio       F1-Score Medio      Desviación Estándar      Desviación Estándar 
                        (test_accuracy)        (test_f1)         Accuracy                 F1-Score
───────────────────────────────────────────────────────────────────────────────────────────────────────────────
Logistic Regression      0.964042            0.978317            0.002634                 0.001622
───────────────────────────────────────────────────────────────────────────────────────────────────────────────
Gradient Boosting        0.963894            0.978260            0.002260                 0.001395
───────────────────────────────────────────────────────────────────────────────────────────────────────────────
Random Forest            0.957426            0.974532            0.002070                 0.001293
───────────────────────────────────────────────────────────────────────────────────────────────────────────────

Los tres algoritmos demostraron una gran solidez predictiva. Sin embargo, el modelo de Regresión Logística obtuvo el primer lugar de rendimiento general, alcanzando el F1-Score medio más alto (0.978317) y exhibiendo una desviación estándar de apenas el 0.0016, lo que confirma que su comportamiento es extremadamente estable frente a las variaciones de las particiones de datos. El Gradient Boosting ocupó el segundo lugar por una diferencia infinitesimal a nivel de milésimas partes.

**5.2. Evaluación Definitiva sobre el Conjunto de Test**
El clasificador de Regresión Logística fue seleccionado como el algoritmo definitivo para la auditoría final. Se procedió a clonar el estimador para limpiar la memoria mediante (clone(best_model)) y se entrenó utilizando el 100% de los datos combinados de entrenamiento.
Posteriormente, el modelo congelado realizó predicciones sobre el conjunto de Test (8,427 registros aislados), arrojando el siguiente reporte de clasificación formal (classification_report):

       === Reporte de Clasificación Oficial en Conjunto de Test ===
              Precision    Recall  F1-Score   Support
        0.0      0.8337    0.9582    0.8916      1339
        1.0      0.9919    0.9639    0.9777      7088

   accuracy                          0.9630      8427
  macro avg      0.9128    0.9610    0.9346      8427
weighted avg      0.9667    0.9630    0.9640      8427

Las métricas finales en Test confirman un rendimiento sobresaliente:

  -Test Accuracy: 0.9630 (96.30% de todas las clasificaciones de la muestra ciega fueron correctas).
  -Test F1-Score: 0.9777 (Confirmación de un equilibrio casi perfecto entre la precisión y la sensibilidad de las predicciones).

**5.3. Análisis Cuantitativo de la Matriz de Confusión**
Para evaluar el impacto de las predicciones desde la perspectiva operativa del negocio, se graficó la matriz de confusión utilizando la paleta de degradado rosa de control lineal (cmap_rosa).

Al auditar los cuadrantes reales calculados por la Regresión Logística Balanceada, se extraen las siguientes conclusiones estratégicas:

  -Verdaderos Positivos (6,832): El modelo demuestra una precisión extraordinaria para mapear y confirmar los productos que serán recomendados con éxito por los clientes de Sephora.

  -Verdaderos Negativos (1,283): Identificación certera de los registros donde el usuario rechazó el artículo o experimentó insatisfacción.

  -Falsos Negativos (256): Casos donde el algoritmo predijo erróneamente un rechazo, pero el cliente sí recomendó el producto. Representa un costo de oportunidad menor en inventario.

  -Falsos Positivos (56) — Mitigación Crítica del Área de Riesgo: Este indicador representa el mayor logro de ingeniería del modelo. El sistema solo se equivocó 56 veces prediciendo que un producto sería un éxito cuando en la realidad no gustó. Mantener esta métrica cercana a cero es el requerimiento de oro para Sephora, ya que evita que la compañía invierta presupuesto en campañas de marketing personalizado o distribución prioritaria sobre cosméticos defectuosos o con reseñas deficientes.

**5.4. Regularización y Serialización con Compresión de Software**
Una vez concluida la optimización, se procedió a la persistencia del clasificador final. Para asegurar la transferencia tecnológica y la operatividad del desarrollo, se utilizó la librería (joblib) para serializar el estimador entrenado y guardarlo de forma persistente en un archivo binario.

A través del orquestador (main.py), la persistencia incorporó un parámetro avanzado de optimización de infraestructura mediante la instrucción (joblib.dump(best_model), path, compress=3). Aplicar una compresión Zlib de nivel 3 reduce drásticamente el peso físico del archivo (.pkl) final en el disco duro. Esto optimiza el uso de almacenamiento en el repositorio, agiliza la transferencia de archivos en la red de Git y acelera los tiempos de lectura I/O en memoria cuando el modelo sea invocado en caliente por una API o microservicio web.

Para garantizar la estabilidad a largo plazo del microservicio, el entorno de ejecución fue degradado estratégicamente de la versión experimental de Python 3.13 a la versión estable de producción Python 3.12.2 (.venv). En la ingeniería de software, esta decisión elimina incompatibilidades de bajo nivel en las APIs de compilación de Scikit-Learn y Joblib, asegurando la reproducibilidad del pipeline y la inmutabilidad de los pesos matemáticos empaquetados en el archivo físico (mejor_gradient_boosting.pkl).

## 6. Optimización de Hiperparámetros (GridSearchCV)
Buscando explorar si una arquitectura basada en árboles de decisión optimizados por gradiente podía superar el rendimiento lineal del clasificador base, se desarrolló una fase de sintonización fina de hiperparámetros sobre el algoritmo (GradientBoostingClassifier).

**6.1. Definición de la Malla Paramétrica (GridSizing)**
Se diseñó un espacio de búsqueda exhaustivo estructurado en un diccionario de Python (param_grid), que abarcó 144 combinaciones independientes combinando los siguientes parámetros de control:

  -"n_estimators": [100, 200, 300] (Cantidad de árboles secuenciales en el ensamble).

  -"learning_rate": [0.05, 0.1, 0.2] (Ritmo de contracción de los pasos de optimización).

  -"max_depth": [2, 3] (Profundidad máxima de los árboles para controlar varianza).

  -"subsample": [0.8, 1.0] (Fracción de muestras destinadas al ajuste de cada árbol).

Al vincular la búsqueda a una Validación Cruzada de 3 pliegues (3-Fold CV) y fijar como criterio de selección el F1-Score, el motor de ejecución calculó y procesó un total de 432 entrenamientos individuales (144 candidatos x 3 pliegues) utilizando todos los procesadores del sistema en paralelo (n_jobs=-1).

**6.2. Análisis del Impacto de los Parámetros Ganadores**
Tras concluir las 432 evaluaciones, el GridSearchCV estabilizó un F1-Score máximo en entrenamiento de 0.978358, aislando la siguiente combinación de hiperparámetros óptima:{'learning_rate': 0.05, 'max_depth': 2, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 200, 'subsample': 1.0}
  El comportamiento de estos valores ganadores aporta dos hallazgos de gran valor para la ingeniería de características:

   1-Suavizado de la Convergencia (learning_rate=0.05 y n_estimators=200): El optimizador determinó que es más eficiente reducir la tasa de aprendizaje a la mitad (el valor por defecto de Scikit-Learn es 0.1) y duplicar la cantidad de árboles secuenciales. En los algoritmos de aumento de gradiente, avanzar con pasos más pequeños pero extendiendo el número de estimadores permite que el modelo corrija los residuos de forma gradual y precisa, convergiendo de manera suave hacia el óptimo global sin oscilaciones inestables.

   2-Regularización Natural por Profundidad (max_depth=2): Restringir las ramas a solo 2 niveles de profundidad (árboles planos) actúa como un potente protector contra el sobreajuste (overfitting). Impide que los árboles individuales memoricen el ruido o las particularidades del set de Train, forzando al ensamble a concentrarse exclusivamente en las señales más representativas del negocio cosmético.

**6.3. Validación Cruzada vs. Inferencia Ciega en Test**
Para auditar el verdadero valor incremental del costo computacional invertido en el GridSearch, se extrajo el modelo entrenado con el 100% de los datos de entrenamiento (grid.best_estimator_) gracias a la instrucción refit=True, y se sometió a una inferencia ciega sobre el conjunto de prueba de Test (184,884 filas).

Los resultados se contrastaron directamente en una tabla frente a un Gradient Boosting Base (configurado con los parámetros por defecto de fábrica):

     === Tabla de Validación Final de la Optimización (Test Set) ===
Métrica Clasificadora       Modelo Base (Default)       Modelo Optimizado (GridSearch)
──────────────────────────────────────────────────────────────────────────────────────
Accuracy                           0.962858                         0.962852
F1-Score                           0.977587                         0.977584
ROC-AUC                            0.986086                         0.985732

El análisis cuantitativo de la tabla revela un fenómeno crítico: las métricas de ambos modelos son prácticamente idénticas, registrando variaciones marginales que se sitúan en el orden de las cienmilésimas partes (< 0.00001).

Al contrastar los cuadrantes de las matrices de confusión reales calculadas para ambos clasificadores, se ratifica esta paridad absoluta:

Matriz Modelo Base                     Matriz Modelo Optimizado
┌──────────────────────────────┐       ┌──────────────────────────────┐
│  VN: 28,261   │  FP: 1,330   │       │  VN: 28,259   │  FP: 1,332   │
├───────────────┼──────────────┤       ├───────────────┼──────────────┤
│  FN:  5,537   │  VP: 149,756 │       │  FN:  5,536   │  VP: 149,757 │
└──────────────────────────────┘       └──────────────────────────────┘

El modelo optimizado logró disminuir un (1) falso negativo (pasando de 5,537 a 5,536), pero a costa de incrementar en dos (2) los falsos positivos (de 1,330 a 1,332). Esto demuestra que el clasificador base ya operaba sobre el límite asintótico de optimización, determinando que el esfuerzo computacional de las 432 combinaciones no generó una ventaja predictiva incremental significativa en la práctica.

## 7. Conclusiones y Recomendaciones

**7.1. Reflexión Final y Justificación de Ingeniería**
El desarrollo del proyecto semestral concluyó con éxito la construcción de un ciclo de vida completo de Ciencia de Datos bajo rigurosas prácticas de Ingeniería de Software. El principal hallazgo técnico determina que el éxito predictivo del sistema no radicó en la complejidad del algoritmo, sino en la calidad del Pipeline de Preprocesamiento.

Que un modelo base con sus parámetros por defecto alcance de forma nativa un Accuracy del 96.28% y un poder de discriminación ROC-AUC del 98.60% es la prueba científica de que los transformadores personalizados personalizados limpiaron el ruido, normalizaron las escalas de manera equitativa y controlaron los outliers de forma óptima. Los datos ingresaron tan limpios a la capa de modelamiento que los algoritmos alcanzaron su techo predictivo de forma casi inmediata.

**7.2. Retorno de Inversión y Decisiones de Arquitectura para Sephora**
Desde la perspectiva operativa y económica del negocio, la recomendación técnica definitiva para la empresa es desplegar el Modelo Base de Gradient Boosting o la Regresión Logística Balanceada, descartando el uso de mallas hiperparamétricas continuas para la operación diaria.

Al demostrar rendimientos equivalentes, la puesta en producción del modelo base le ahorra a Sephora los altos costos financieros asociados al consumo de cómputo en la nube (servidores AWS, Google Cloud o Azure) necesarios para ejecutar grillas periódicas de cientos de combinaciones. El negocio obtiene un clasificador de alta fidelidad, sumamente liviano y estable, minimizando la tasa de falsas alarmas a niveles mínimos (solo 1,330 errores frente a más de 149,000 aciertos reales), lo que blinda la experiencia de usuario en la plataforma web.

**7.3. Sugerencias de Trabajo Futuro**

   1-Capa de Procesamiento de Lenguaje Natural (NLP): Aprovechar la infraestructura modular creada para incorporar un transformador de extracción de texto en la carpeta (src/). Se sugiere procesar la columna cruda (review_text) mediante algoritmos de vectorización TF-IDF o modelos preentrenados basados en Transformers (BERT) , convirtiendo el sentimiento y la semántica de los comentarios escritos en nuevas características numéricas para el modelo clasificador.

   2-Dashboard de Explotación y Monitoreo: Desarrollar una interfaz gráfica interactiva utilizando librerías como Streamlit o conectar el archivo procesado (sephora_limpio.csv) directamente a un panel de PowerBI. Esto permitirá a los analistas comerciales de Sephora monitorear el engagement de las marcas líderes (como CLINIQUE o Tatcha) y simular la predicción de recomendaciones de productos en tiempo real mediante la invocación directa del archivo binario persistido con (joblib).

## 8. Evidencia de la Estructura de Archivos del Sistema
El ecosistema de software se organizó bajo una arquitectura profesional de directorios que garantiza el desacoplamiento, la trazabilidad de los datos y el correcto mantenimiento del código fuente a largo plazo:

DATASET-PRODUCTOS-COSMETICOS
│
├── .vscode/
│   └── settings.json .................... Configuración del entorno de desarrollo local
│
├── data/raw
│   └── 
│       ├── product_info.csv ............. Catálogo maestro original (8,494 filas)
│       ├── reviews_0-250.csv
│       ├── reviews_250-500.csv
│       ├── reviews_500-750.csv
│       ├── reviews_750-1250.csv
│       └── reviews_1250-end.csv ......... Fuentes crudas de comentarios unificados (1,094,411 filas)
│ 
├── docs/
│   ├── informe_tecnico.md .............. Documentación informe versión 1.0 
│   ├── informe_tecnico2.md ............. Documentación informe versión 2.0 
│   └── technical_report.md
│
├── notebooks/
│   ├── 01_EDA.ipynb ..................... Análisis Exploratorio, Heatmaps de Nulos e IQR
│   ├── 02_Pipelines.ipynb ............... Orquestación y diseño del ColumnTransformer
│   ├── 02_unsupervised_learning.ipynb ... Segmentación por K-Means, Silueta y PCA
│   ├── 03_supervised_modeling.ipynb ..... Evaluación comparativa de clasificadores y CV
│   ├── 04_hyperparameter_optimization.ipynb . Optimización manual de mallas de entrenamiento
│   └── 05_final_analysis.ipynb .......... Integración analítica y visualización de Test
├── src/
│   ├── __pycache__/
├── models/
│   └── trained_models/
│       └── mejor_gradient_boosting.pkl .. Artefacto binario serializado y comprimido con Joblib
│   ├── _init_.py ........................ Identificador de paquete modular de Python
│   ├── audit.py ......................... Verificación de integridad y hash SHA-256
│   ├── data_preprocessing.py ............ Lógica core de Transformadores Personalizados
│   ├── hyperparameter_tuning.py ......... Motor de optimización GridSearchCV para producción
│   ├── model_evaluation.py .............. Funciones automatizadas de reportes, ROC y matrices
│   ├── model_training.py ................ Scripts de ajuste offline de algoritmos supervisados
│   ├── optimization.py .................. Algoritmos de downcasting numérico y ahorro de memoria
│   ├── pipeline.py ...................... Ensamblado del flujo secuencial de Scikit-Learn
│   ├── transformers.py .................. Lógica core de los Transformadores Personalizados
│   └── utils.py ......................... Utilidades de carga y split estratificado global
│
├── main.py .............................. Orquestador end-to-end del Pipeline de producción
├── README.md
└── requirements.txt ..................... Registro de dependencias estables (Python 3.12.2)