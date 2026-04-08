"""
Módulo de utilidades para el análisis de productos cosméticos.

Este módulo contiene funciones auxiliares para cargar, limpiar y procesar
el dataset de productos cosméticos.
"""

import pandas as pd
import numpy as np


def load_data(filepath: str) -> pd.DataFrame:
    """
    Carga el dataset de productos cosméticos desde un archivo CSV.

    Args:
        filepath (str): Ruta al archivo CSV.

    Returns:
        pd.DataFrame: DataFrame con los datos cargados.

    Raises:
        FileNotFoundError: Si el archivo no existe.
        pd.errors.EmptyDataError: Si el archivo está vacío.
    """
    try:
        df = pd.read_csv(filepath)
        print(f"Datos cargados exitosamente. Forma: {df.shape}")
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"Archivo no encontrado: {filepath}")
    except pd.errors.EmptyDataError:
        raise pd.errors.EmptyDataError(f"Archivo vacío: {filepath}")


def clean_product_type(product_type: pd.Series) -> pd.Series:
    """
    Limpia la columna product_type removiendo caracteres no deseados.

    Args:
        product_type (pd.Series): Serie con los tipos de producto.

    Returns:
        pd.Series: Serie limpiada.
    """
    return product_type.fillna('Unknown').str.replace('\n', '').str.strip()


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega columnas derivadas al DataFrame.

    Args:
        df (pd.DataFrame): DataFrame original.

    Returns:
        pd.DataFrame: DataFrame con columnas adicionales.
    """
    df = df.copy()
    df['description_length'] = df['description'].str.len()
    df['has_image'] = df['img'].notna()
    df['has_shade_image'] = df['shade_img'].notna()
    return df


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Realiza la limpieza completa del dataset.

    Args:
        df (pd.DataFrame): DataFrame original.

    Returns:
        pd.DataFrame: DataFrame limpio.
    """
    df_clean = df.copy()
    df_clean['product_type'] = clean_product_type(df_clean['product_type'])
    df_clean = add_derived_columns(df_clean)
    # Eliminar columnas no útiles
    columns_to_drop = ['rating', 'dupes', 'price_site']
    df_clean = df_clean.drop(columns=[col for col in columns_to_drop if col in df_clean.columns])
    return df_clean


def validate_dataset(df: pd.DataFrame) -> dict:
    """
    Valida la integridad del dataset.

    Args:
        df (pd.DataFrame): DataFrame a validar.

    Returns:
        dict: Diccionario con métricas de validación.
    """
    validation = {
        'shape': df.shape,
        'duplicates': df.duplicated().sum(),
        'unique_ids': df['ID'].nunique() if 'ID' in df.columns else None,
        'null_counts': df.isnull().sum().to_dict()
    }
    return validation


if __name__ == "__main__":
    # Ejemplo de uso
    data_path = "../data/products.csv"
    df = load_data(data_path)
    df_clean = clean_dataset(df)
    validation = validate_dataset(df_clean)
    print("Validación:", validation)