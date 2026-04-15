import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Asegurarse que la ruta de importación sea correcta según la estructura
from src.transformers import (
    DropColumnsTransformer,
    UnknownToNaNTransformer,
    StringCleanerTransformer,
    DropHighMissingTransformer,
    SmartImputerTransformer,
    OutlierCapper,
    DropZeroVarianceTransformer
)

def build_preprocessing_pipeline(columns_to_drop=None):
    # Ajustamos a minúsculas para ser consistentes con la limpieza previa
    default_drop = ['id', 'img', 'name', 'description', 'shade_img', 'dupes', 'price_site']
    
    if columns_to_drop is None:
        columns_to_drop = default_drop
    else:
        # Unimos ambas listas asegurando minúsculas
        columns_to_drop = list(set([c.lower() for c in default_drop + list(columns_to_drop)]))

    # Pipeline para datos numéricos
    num_pipe = Pipeline([
        ('capper', OutlierCapper(apply_capping=True)),
        ('zero_variance', DropZeroVarianceTransformer()),
        ('scaler', StandardScaler())
    ])

    # Pipeline para datos categóricos
    cat_pipe = Pipeline([
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Transformador por columnas
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_pipe, make_column_selector(dtype_include='number')),
            ('cat', cat_pipe, make_column_selector(dtype_exclude='number'))
        ],
        remainder='drop'
    )

    # Pipeline Principal
    full_pipeline = Pipeline([
        ('drop_leaks', DropColumnsTransformer(columns_to_drop=columns_to_drop)),
        ('clean_strings', StringCleanerTransformer()),
        ('clean_unknowns', UnknownToNaNTransformer()),
        ('drop_high_nan', DropHighMissingTransformer(threshold=0.8)),
        ('smart_imputer', SmartImputerTransformer(low_threshold=0.10)),
        ('preprocessing', preprocessor)
    ])

    # Mantiene el formato DataFrame de Pandas
    full_pipeline.set_output(transform="pandas")

    return full_pipeline

def run_pipeline(df, columns_to_drop=None):
    # Antes de entrar al pipeline, normalizamos nombres de columnas
    df_copy = df.copy()
    df_copy.columns = [c.strip().lower() for c in df_copy.columns]
    
    pipeline = build_preprocessing_pipeline(columns_to_drop=columns_to_drop)
    
    if isinstance(df_copy, pd.DataFrame):
        return pipeline.fit_transform(df_copy)
    raise ValueError('Input debe ser un pandas DataFrame')