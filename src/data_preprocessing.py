"""src.data_preprocessing

Preprocesamiento robusto para el dataset "Sephora Cosmetics Reviews".

Este módulo genera un preprocesador compatible con scikit-learn para
aprendizaje supervisado, con divisiones estratificadas para evitar sesgos,
protección contra data leakage y persistencia automatizada de artefactos.

El archivo está diseñado para reutilizarse desde notebooks de:
- modelado supervisado
- optimización de hiperparámetros
- evaluación final

Notas de implementación
------------------------
- Los transformadores personalizados heredan de BaseEstimator y
  TransformerMixin.
- Se garantiza el ajuste solo con X_train (evitando leakage).
- Se usan ColumnTransformer y Pipeline para asegurar compatibilidad con
  el ecosistema scikit-learn.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class DropColumnsTransformer(BaseEstimator, TransformerMixin):
    """Transformer que elimina columnas específicas.

    Parameters
    ----------
    columns_to_drop:
        Lista de nombres de columnas que se eliminarán si existen en el
        DataFrame. La comparación se realiza respetando el nombre exacto
        tal como esté en las columnas de X.

    Notes
    -----
    - Si alguna columna no existe en X, se ignora.
    """

    def __init__(self, columns_to_drop: List[str]):
        self.columns_to_drop = columns_to_drop

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "DropColumnsTransformer":
        """Ajusta el transformer.

        Parameters
        ----------
        X:
            DataFrame de entrada.
        y:
            Variable objetivo (no se usa).

        Returns
        -------
        DropColumnsTransformer
            self
        """

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Elimina las columnas especificadas en ``columns_to_drop``.

        Parameters
        ----------
        X:
            DataFrame de entrada.

        Returns
        -------
        pd.DataFrame
            DataFrame con las columnas eliminadas.
        """

        X_copy = X.copy()
        cols = [c for c in self.columns_to_drop if c in X_copy.columns]
        return X_copy.drop(columns=cols, errors="ignore")


class UnknownToNaNTransformer(BaseEstimator, TransformerMixin):
    """Convierte valores "unknown" a ``np.nan`` en columnas específicas.

    Parameters
    ----------
    columns:
        Lista de columnas sobre las cuales se realizará el reemplazo.

    Notes
    -----
    - La detección de "unknown" es case-insensitive.
    - Solo se reemplaza cuando el valor (tras strip) coincide exactamente con
      ``unknown``.
    """

    def __init__(self, columns: List[str]):
        self.columns = columns

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "UnknownToNaNTransformer":
        """Ajusta el transformer.

        Parameters
        ----------
        X:
            DataFrame de entrada.
        y:
            Variable objetivo (no se usa).

        Returns
        -------
        UnknownToNaNTransformer
            self
        """

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Reemplaza "unknown" por ``np.nan`` en las columnas configuradas.

        Parameters
        ----------
        X:
            DataFrame de entrada.

        Returns
        -------
        pd.DataFrame
            DataFrame con los reemplazos aplicados.
        """

        X_copy = X.copy()

        cols = [c for c in self.columns if c in X_copy.columns]
        if not cols:
            return X_copy

        # Regex para case-insensitive: ^\s*unknown\s*$
        pattern = re.compile(r"^\s*unknown\s*$", flags=re.IGNORECASE)

        for col in cols:
            col_series = X_copy[col]

            mask = col_series.apply(
                lambda v: isinstance(v, str)
                and bool(pattern.match(v.strip()))
            )

            X_copy.loc[mask, col] = np.nan

        return X_copy


class SmartImputerTransformer(BaseEstimator, TransformerMixin):
    """Imputación simple y robusta sin leakage (solo aprende en fit).

    Parameters
    ----------
    strategy:
        Estrategia base para imputar.

        - Si ``strategy == 'auto'``: usa mediana para numéricas y mode para
          categóricas.
        - Si ``strategy == 'median'``: mediana para numéricas y mode para
          categóricas.
        - Si ``strategy == 'most_frequent'``: mode para categóricas y mediana
          para numéricas.

    Notes
    -----
    - Este transformer imputa valores en el DataFrame completo.
    - Dentro del Pipeline final se usa además SimpleImputer, por lo que
      este transformer sirve para una robustez adicional previa o cuando se
      use fuera del Pipeline.
    """

    def __init__(self, strategy: str):
        self.strategy = strategy
        self.impute_values_: Dict[str, Any] = {}

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "SmartImputerTransformer":
        """Aprende los valores de imputación a partir de ``X``.

        Parameters
        ----------
        X:
            DataFrame de entrenamiento.
        y:
            Variable objetivo (no se usa).

        Returns
        -------
        SmartImputerTransformer
            self
        """

        self.impute_values_ = {}

        for col in X.columns:
            series = X[col]
            if pd.api.types.is_numeric_dtype(series):
                if self.strategy in {"auto", "median"}:
                    self.impute_values_[col] = series.median()
                else:
                    # most_frequent (para numéricas usaremos mediana para robustez)
                    self.impute_values_[col] = series.median()
            else:
                # categóricas
                if self.strategy in {"auto", "median", "most_frequent"}:
                    mode_val = series.mode(dropna=True)
                    self.impute_values_[col] = mode_val.iloc[0] if not mode_val.empty else "Unknown"
                else:
                    mode_val = series.mode(dropna=True)
                    self.impute_values_[col] = mode_val.iloc[0] if not mode_val.empty else "Unknown"

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Imputa valores faltantes usando ``impute_values_``.

        Parameters
        ----------
        X:
            DataFrame a transformar.

        Returns
        -------
        pd.DataFrame
            DataFrame con valores imputados.
        """

        X_copy = X.copy()
        return X_copy.fillna(self.impute_values_)


class DropHighMissingTransformer(BaseEstimator, TransformerMixin):
    """Elimina columnas con un porcentaje de nulos mayor al umbral.

    Parameters
    ----------
    threshold:
        Umbral en rango [0, 1]. Se eliminarán columnas con
        ``missing_pct > threshold``.
    """

    def __init__(self, threshold: float):
        if not (0.0 <= threshold <= 1.0):
            raise ValueError("threshold debe estar entre 0 y 1")
        self.threshold = threshold
        self.columns_to_drop_: List[str] = []

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "DropHighMissingTransformer":
        """Aprende qué columnas eliminar.

        Parameters
        ----------
        X:
            DataFrame de entrenamiento.
        y:
            Variable objetivo (no se usa).

        Returns
        -------
        DropHighMissingTransformer
            self
        """

        missing_pct = X.isnull().mean()
        self.columns_to_drop_ = missing_pct[missing_pct > self.threshold].index.tolist()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Elimina columnas aprendidas en fit.

        Parameters
        ----------
        X:
            DataFrame a transformar.

        Returns
        -------
        pd.DataFrame
            DataFrame sin las columnas con demasiados nulos.
        """

        return X.drop(columns=self.columns_to_drop_, errors="ignore")


class OutlierCapper(BaseEstimator, TransformerMixin):
    """Winsorization/capping basado en percentiles aprendidos.

    Parameters
    ----------
    lower_quantile:
        Percentil inferior para el cap.
    upper_quantile:
        Percentil superior para el cap.

    Notes
    -----
    - Se aplicará solo a columnas numéricas.
    - En ``fit`` se calculan ``lower_bounds_`` y ``upper_bounds_`` por
      columna.
    """

    def __init__(self, lower_quantile: float = 0.01, upper_quantile: float = 0.99):
        if not (0.0 < lower_quantile < 1.0):
            raise ValueError("lower_quantile debe estar en (0, 1)")
        if not (0.0 < upper_quantile < 1.0):
            raise ValueError("upper_quantile debe estar en (0, 1)")
        if lower_quantile >= upper_quantile:
            raise ValueError("lower_quantile debe ser menor que upper_quantile")

        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile
        self.lower_bounds_: Dict[str, float] = {}
        self.upper_bounds_: Dict[str, float] = {}

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "OutlierCapper":
        """Calcula bounds de cap para cada columna numérica.

        Parameters
        ----------
        X:
            DataFrame de entrenamiento.
        y:
            Variable objetivo (no se usa).

        Returns
        -------
        OutlierCapper
            self
        """

        self.lower_bounds_ = {}
        self.upper_bounds_ = {}

        numeric_cols = X.select_dtypes(include=["number"]).columns
        for col in numeric_cols:
            s = X[col]
            # Manejo defensivo por columnas vacías
            if s.dropna().empty:
                lower = np.nan
                upper = np.nan
            else:
                lower = float(s.quantile(self.lower_quantile))
                upper = float(s.quantile(self.upper_quantile))
            self.lower_bounds_[col] = lower
            self.upper_bounds_[col] = upper

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Aplica capping a columnas numéricas.

        Parameters
        ----------
        X:
            DataFrame a transformar.

        Returns
        -------
        pd.DataFrame
            DataFrame con outliers cappeados.
        """

        X_copy = X.copy()
        numeric_cols = X_copy.select_dtypes(include=["number"]).columns

        for col in numeric_cols:
            if col not in self.lower_bounds_ or col not in self.upper_bounds_:
                continue
            lower = self.lower_bounds_[col]
            upper = self.upper_bounds_[col]
            if pd.isna(lower) or pd.isna(upper):
                continue
            X_copy[col] = X_copy[col].clip(lower=lower, upper=upper)

        return X_copy


def normalize_column_names(columns: Iterable[str]) -> List[str]:
    """Normaliza nombres de columnas.

    Convierte a minúsculas, elimina espacios, y reemplaza caracteres especiales
    por guiones bajos.

    Parameters
    ----------
    columns:
        Iterable de nombres de columnas.

    Returns
    -------
    list[str]
        Lista de nombres normalizados.
    """

    normalized: List[str] = []
    for c in columns:
        c_norm = str(c).strip().lower()
        # Reemplazar espacios por underscore
        c_norm = re.sub(r"\s+", "_", c_norm)
        # Eliminar caracteres no alfanuméricos/underscore
        c_norm = re.sub(r"[^a-z0-9_]+", "_", c_norm)
        # Compactar múltiples underscores
        c_norm = re.sub(r"_+", "_", c_norm)
        normalized.append(c_norm)
    return normalized


def drop_duplicate_rows(df: pd.DataFrame, subset: Optional[List[str]] = None) -> pd.DataFrame:
    """Elimina filas duplicadas de un DataFrame.

    Parameters
    ----------
    df:
        DataFrame de entrada.
    subset:
        Columnas a considerar para detectar duplicados. Si es None, se usan
        todas las columnas.

    Returns
    -------
    pd.DataFrame
        DataFrame sin duplicados.
    """

    return df.drop_duplicates(subset=subset)


def infer_numeric_and_categorical_columns(
    df: pd.DataFrame,
) -> Tuple[List[str], List[str]]:
    """Infiera automáticamente columnas numéricas y categóricas.

    Parameters
    ----------
    df:
        DataFrame de entrada.

    Returns
    -------
    (list[str], list[str])
        Tuple ``(numeric_cols, categorical_cols)``.
    """

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = [c for c in df.columns.tolist() if c not in numeric_cols]
    return numeric_cols, categorical_cols


def missing_values_summary(df: pd.DataFrame, top_n: int = 30) -> str:
    """Genera un resumen textual de valores faltantes por columna.

    Parameters
    ----------
    df:
        DataFrame a resumir.
    top_n:
        Máximo de columnas a reportar ordenadas por cantidad de nulos.

    Returns
    -------
    str
        String con el resumen.
    """

    null_counts = df.isnull().sum().sort_values(ascending=False)
    null_pct = (df.isnull().mean() * 100.0).loc[null_counts.index]

    header = "Resumen de valores faltantes (top)"
    lines = [header, "=" * len(header)]

    n_cols_report = min(top_n, len(null_counts))
    for i in range(n_cols_report):
        col = null_counts.index[i]
        lines.append(f"{i + 1:02d}. {col}: {null_counts.iloc[i]} nulos ({null_pct.iloc[i]:.2f}%)")

    total_nulls = int(df.isnull().sum().sum())
    lines.append("-" * 40)
    lines.append(f"Total nulos en el DataFrame: {total_nulls}")

    summary = "\n".join(lines)
    return summary


def build_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    """Construye un ColumnTransformer con transformadores numéricos y categóricos.

    Parámetros
    ----------
    df:
        DataFrame de referencia para inferir columnas numéricas/categóricas.

    Returns
    -------
    sklearn.compose.ColumnTransformer
        preprocesador combinado.
    """

    numeric_cols, categorical_cols = infer_numeric_and_categorical_columns(df)

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_cols),
            ("categorical", categorical_pipeline, categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return preprocessor


def load_and_split_data(
    filepath: str,
    target_col: str = "is_recommended",
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Carga un CSV y realiza split estratificado.

    Parámetros
    ----------
    filepath:
        Ruta al CSV final.
    target_col:
        Nombre de la columna objetivo.
    test_size:
        Proporción del conjunto de test.
    random_state:
        Semilla para reproducibilidad.

    Returns
    -------
    X_train, X_test, y_train, y_test
        Tupla con los splits.

    Raises
    ------
    FileNotFoundError
        Si ``filepath`` no existe.
    ValueError
        Si ``target_col`` no existe en el CSV.
    """

    fp = Path(filepath)
    if not fp.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {filepath}")

    df = pd.read_csv(fp)

    # Normaliza nombres (opcional pero robusto)
    df = df.copy()
    df.columns = normalize_column_names(df.columns)
    target_col_norm = normalize_column_names([target_col])[0]

    if target_col_norm not in df.columns:
        raise ValueError(f"La columna objetivo '{target_col}' no existe en el dataset (normalizada: '{target_col_norm}').")

    df = df.dropna(subset=[target_col_norm]).copy()

    y = df[target_col_norm]
    X = df.drop(columns=[target_col_norm])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    return X_train, X_test, y_train, y_test


def preprocess_train_test(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, Pipeline]:

    # Detectar columnas categóricas
    _, cat_cols_train = infer_numeric_and_categorical_columns(X_train)

    # Simular los pasos que afectan las columnas
    unknown_cleaner = UnknownToNaNTransformer(columns=cat_cols_train)
    missing_dropper = DropHighMissingTransformer(threshold=0.8)

    X_train_temp = unknown_cleaner.fit_transform(X_train)
    X_train_temp = missing_dropper.fit_transform(X_train_temp)

    # Construir ColumnTransformer con las columnas finales
    base_preprocessor = build_preprocessor(X_train_temp)

    # Pipeline completo
    full_pipeline = Pipeline(
        steps=[
            ("unknown_cleaner", UnknownToNaNTransformer(columns=cat_cols_train)),
            ("missing_dropper", DropHighMissingTransformer(threshold=0.8)),
            ("outlier_capper", OutlierCapper(
                lower_quantile=0.01,
                upper_quantile=0.99
            )),
            ("preprocessor", base_preprocessor)
        ]
    )

    # Evitar explosión de memoria: intentar mantener formato disperso.
    # Si el pipeline devuelve sparse, convertimos a array solo al final.
    X_train_processed = full_pipeline.fit_transform(X_train)
    X_test_processed = full_pipeline.transform(X_test)

    try:
        # Si es scipy sparse, mantenemos csr.
        import scipy.sparse as sp

        if sp.issparse(X_train_processed):
            # GradientBoosting no maneja sparse bien en sklearn,
            # así que convertimos solo si es imprescindible, pero en float32 reduce OOM.
            X_train_processed = X_train_processed.astype(np.float32).toarray()
            X_test_processed = X_test_processed.astype(np.float32).toarray()
        else:
            X_train_processed = np.asarray(X_train_processed, dtype=np.float32)
            X_test_processed = np.asarray(X_test_processed, dtype=np.float32)
    except Exception:
        X_train_processed = np.asarray(X_train_processed, dtype=np.float32)
        X_test_processed = np.asarray(X_test_processed, dtype=np.float32)

    return (X_train_processed, X_test_processed, full_pipeline)



def save_processed_datasets(
    *,
    X_train_processed: np.ndarray,
    X_test_processed: np.ndarray,
    y_train: pd.Series,
    y_test: pd.Series,
    preprocessor: ColumnTransformer,
    output_dir: str = "data/processed",
) -> Dict[str, str]:
    """Guarda artefactos procesados en disco usando joblib.

    Parámetros
    ----------
    X_train_processed:
        Array transformado de entrenamiento.
    X_test_processed:
        Array transformado de test.
    y_train:
        Vector objetivo de entrenamiento.
    y_test:
        Vector objetivo de test.
    preprocessor:
        Preprocesador ajustado.
    output_dir:
        Directorio base para guardar artefactos.

    Returns
    -------
    dict[str, str]
        Mapa de nombre lógico -> ruta de archivo guardado.
    """

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    artifacts = {
        "X_train_processed": out_path / "X_train_processed.pkl",
        "X_test_processed": out_path / "X_test_processed.pkl",
        "y_train": out_path / "y_train.pkl",
        "y_test": out_path / "y_test.pkl",
        "preprocessor": out_path / "preprocessor.pkl",
    }

    joblib.dump(X_train_processed, artifacts["X_train_processed"], compress=3)
    joblib.dump(X_test_processed, artifacts["X_test_processed"], compress=3)
    joblib.dump(y_train, artifacts["y_train"], compress=3)
    joblib.dump(y_test, artifacts["y_test"], compress=3)
    joblib.dump(preprocessor, artifacts["preprocessor"], compress=3)

    return {k: str(v) for k, v in artifacts.items()}


def _load_product_info(product_info_path: Path) -> pd.DataFrame:
    """Carga product_info.csv y normaliza nombres de columnas."""

    df = pd.read_csv(product_info_path)
    df = df.copy()
    df.columns = normalize_column_names(df.columns)
    return df


def _load_reviews_chunks(reviews_dir: Path) -> List[pd.DataFrame]:
    """Carga todos los reviews_*.csv desde ``reviews_dir``."""

    pattern = re.compile(r"^reviews_.*\.csv$", flags=re.IGNORECASE)
    files = [p for p in reviews_dir.glob("reviews_*.csv") if pattern.match(p.name)]
    files = sorted(files, key=lambda p: p.name)

    if not files:
        raise FileNotFoundError(f"No se encontraron archivos reviews_*.csv en: {reviews_dir}")

    dfs: List[pd.DataFrame] = []
    for f in files:
        df = pd.read_csv(f)
        df = df.copy()
        df.columns = normalize_column_names(df.columns)
        dfs.append(df)

    return dfs


def _merge_reviews_with_products(
    *,
    product_df: pd.DataFrame,
    reviews_df: pd.DataFrame,
) -> pd.DataFrame:
    """Une reviews con product_info usando ``product_id``."""

    # Normalizamos nombres por robustez (ya deberían estar normalizados)
    product_df = product_df.copy()
    reviews_df = reviews_df.copy()

    if "product_id" not in product_df.columns:
        raise ValueError("product_info.csv no contiene la columna product_id (normalizada).")
    if "product_id" not in reviews_df.columns:
        raise ValueError("reviews_*.csv no contienen la columna product_id (normalizada).")

    df_final = reviews_df.merge(product_df, on="product_id", how="left")
    return df_final


def run_preprocessing_pipeline(
    *,
    raw_dir: str | None = None,
    processed_dir: str | None = None,

    target_col: str = "is_recommended",
    test_size: float = 0.2,
    random_state: int = 42,
) -> Dict[str, str]:
    """Orquesta el preprocesamiento completo desde los CSV crudos.

    El flujo implementa:
    1) Carga product_info.csv.
    2) Carga reviews_*.csv.
    3) Concatenación de reseñas.
    4) Merge con product_info vía product_id.
    5) Split estratificado.
    6) Fit/transform sin leakage.
    7) Guardado de artefactos .pkl.

    Parameters
    ----------
    raw_dir:
        Directorio base con archivos CSV crudos.
    processed_dir:
        Directorio destino para artefactos.
    target_col:
        Columna objetivo.
    test_size:
        Tamaño de test.
    random_state:
        Semilla.

    Returns
    -------
    dict[str, str]
        Rutas de artefactos guardados.
    """

    # Rutas por defecto basadas en PROJECT_ROOT para evitar fragilidad por cwd
    from src.paths import RAW_DIR, PROCESSED_DIR

    if raw_dir is None:
        raw_dir = str(RAW_DIR)
    if processed_dir is None:
        processed_dir = str(PROCESSED_DIR)

    raw_path = Path(raw_dir)
    product_info_path = raw_path / "product_info.csv"


    print("\n" + "=" * 80)
    print("[1/6] Cargando product_info.csv")
    print("- Ruta:", str(product_info_path))
    product_df = _load_product_info(product_info_path)
    print(f"- Filas product_info: {len(product_df)} | Columnas: {product_df.shape[1]}")

    print("\n" + "=" * 80)
    print("[2/6] Cargando reviews_*.csv")
    reviews_chunks = _load_reviews_chunks(raw_path)
    reviews_df = pd.concat(reviews_chunks, ignore_index=True)
    print(f"- Número de chunks: {len(reviews_chunks)}")
    print(f"- Filas reviews concatenadas: {len(reviews_df)} | Columnas: {reviews_df.shape[1]}")

    print("\n" + "=" * 80)
    print("[3/6] Merge reviews con product_info (product_id)")
    df_final = _merge_reviews_with_products(product_df=product_df, reviews_df=reviews_df)
    print(f"- df_final filas: {len(df_final)} | columnas: {df_final.shape[1]}")

    # Normaliza nombres nuevamente por robustez
    df_final = df_final.copy()
    df_final.columns = normalize_column_names(df_final.columns)
    target_col_norm = normalize_column_names([target_col])[0]

    print("\n" + "=" * 80)
    print("[4/6] División Train/Test estratificada")

    if target_col_norm not in df_final.columns:
        raise ValueError(f"La columna objetivo '{target_col}' no existe en df_final (normalizada: '{target_col_norm}').")

    df_final = df_final.dropna(subset=[target_col_norm]).copy()
    X = df_final.drop(columns=[target_col_norm])
    columns_to_drop = [
        "review_text",
        "review_title",
        "submission_time",
        "product_id",
        "product_name_x",
        "product_name_y",
        "ingredients",
        "highlights",
        "variation_value",
        "size"
    ]

    X = X.drop(
        columns=[c for c in columns_to_drop if c in X.columns],
        errors="ignore"
    )
    columns_to_drop = [
    "author_id"
    ]

    X = X.drop(columns=columns_to_drop, errors="ignore")
    y = df_final[target_col_norm]

    print(f"- Filas con target no nulo: {len(df_final)}")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    print(f"- X_train: {X_train.shape} | y_train: {y_train.shape}")
    print(f"- X_test:  {X_test.shape} | y_test:  {y_test.shape}")

    print("\n" + "=" * 80)
    print("[5/6] Fit/Transform del preprocesador (sin leakage)")

    # Limpieza previa mínima para robustez (sin alterar distribución entre train/test)
    X_train = X_train.copy()
    X_test = X_test.copy()

    print("\nColumnas con mezcla de tipos:\n")

    for col in X_train.columns:
        tipos = X_train[col].dropna().map(type).unique()

        if len(tipos) > 1:
            print(col)
            print(tipos)
            print("-" * 50)

    print("\nCardinalidad de variables categóricas:\n")

    for col in X_train.select_dtypes(exclude=["number"]).columns:
        print(col, "->", X_train[col].nunique(dropna=True))

    X_train_processed, X_test_processed, preprocessor = preprocess_train_test(X_train, X_test)

    print(f"- X_train_processed: {X_train_processed.shape}")
    print(f"- X_test_processed:  {X_test_processed.shape}")

    print("\n" + "=" * 80)
    print("[6/6] Guardando artefactos en:", processed_dir)
    artifacts_paths = save_processed_datasets(
        X_train_processed=X_train_processed,
        X_test_processed=X_test_processed,
        y_train=y_train,
        y_test=y_test,
        preprocessor=preprocessor,
        output_dir=processed_dir,
    )

    print("- Archivos guardados:")
    for k, p in artifacts_paths.items():
        print(f"  - {k}: {p}")

    print("\n" + "=" * 80)
    print("Pipeline finalizado exitosamente")

    return artifacts_paths


if __name__ == "__main__":
    run_preprocessing_pipeline()

