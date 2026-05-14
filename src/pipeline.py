"""
Pipeline configuration for cosmetics dataset.
Maintains the use of 'set_output(transform="pandas")' for DataFrame output.
"""

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_selection import VarianceThreshold

from src.transformers import (
    DropColumnsTransformer, UnknownToNaNTransformer, StringCleanerTransformer,
    DropHighMissingTransformer, SmartImputerTransformer, OutlierCapper
)

def build_preprocessing_pipeline(columns_to_drop=None):
    """Builds and returns the scikit-learn preprocessing pipeline."""
    default_drop = ['id', 'img', 'name', 'description', 'shade_img', 'dupes', 'price_site']
    
    if columns_to_drop is None:
        columns_to_drop = default_drop
    else:
        columns_to_drop = list(set([c.lower() for c in default_drop + list(columns_to_drop)]))

    # Ruta para variables numéricas (con herramienta nativa para varianza cero)
    num_pipe = Pipeline([
        ('capper', OutlierCapper(apply_capping=True)),
        ('zero_variance', VarianceThreshold(threshold=0.0)),
        ('scaler', StandardScaler())
    ])

    # Ruta para variables categóricas
    cat_pipe = Pipeline([
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Enrutador dinámico
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_pipe, make_column_selector(dtype_include='number')),
            ('cat', cat_pipe, make_column_selector(dtype_exclude='number'))
        ],
        remainder='drop'
    )

    # Pipeline principal
    full_pipeline = Pipeline([
        ('drop_leaks', DropColumnsTransformer(columns_to_drop=columns_to_drop)),
        ('clean_strings', StringCleanerTransformer()),
        ('clean_unknowns', UnknownToNaNTransformer()),
        ('drop_high_nan', DropHighMissingTransformer(threshold=0.8)),
        ('smart_imputer', SmartImputerTransformer(low_threshold=0.10)),
        ('preprocessing', preprocessor)
    ])

    # ¡EL SÚPER PODER DEL GRUPO 8! Mantiene el formato de Pandas automáticamente
    full_pipeline.set_output(transform="pandas")

    return full_pipeline