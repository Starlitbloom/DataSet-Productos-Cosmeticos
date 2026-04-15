"""
Módulo de utilidades para el análisis de productos cosméticos.
"""

import pandas as pd
import numpy as np
import glob
import os

def load_all_data(folder_path: str) -> pd.DataFrame:
    """Busca todos los archivos CSV en la carpeta y los une."""
    all_files = glob.glob(os.path.join(folder_path, "*.csv"))
    
    if not all_files:
        raise FileNotFoundError(f"No se encontraron archivos CSV en {folder_path}")
        
    df_list = []
    for filename in all_files:
        try:
            df = pd.read_csv(filename)
            df_list.append(df)
            print(f"Cargado: {filename} - Filas: {len(df)}")
        except Exception as e:
            print(f"Error al cargar {filename}: {e}")
    
    if not df_list:
        return pd.DataFrame()

    full_df = pd.concat(df_list, ignore_index=True)
    print(f"Dataset completo cargado. Total filas: {len(full_df)}")
    return full_df

def clean_product_type(product_type: pd.Series) -> pd.Series:
    """Limpia la columna product_type."""
    return product_type.fillna('Unknown').astype(str).str.replace('\n', '').str.strip()

def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega columnas derivadas al DataFrame."""
    df = df.copy()
    # Usamos .get() y verificamos existencia para evitar errores si la columna no existe
    if 'description' in df.columns:
        df['description_length'] = df['description'].str.len()
    if 'img' in df.columns:
        df['has_image'] = df['img'].notna()
    if 'shade_img' in df.columns:
        df['has_shade_image'] = df['shade_img'].notna()
    return df

def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Realiza la limpieza completa del dataset."""
    df_clean = df.copy()
    
    # 1. Normalizar nombres de columnas a minúsculas
    df_clean.columns = [col.strip().lower() for col in df_clean.columns]
    
    # 2. Aplicar limpieza de tipo de producto
    if 'product_type' in df_clean.columns:
        df_clean['product_type'] = clean_product_type(df_clean['product_type'])
    
    # 3. Agregar columnas derivadas
    df_clean = add_derived_columns(df_clean)
    
    # 4. Eliminar columnas no útiles si existen
    columns_to_drop = ['rating', 'dupes', 'price_site']
    df_clean = df_clean.drop(columns=[col for col in columns_to_drop if col in df_clean.columns])
    
    return df_clean

def validate_dataset(df: pd.DataFrame) -> dict:
    """Valida la integridad del dataset."""
    # Como normalizamos a minúsculas, buscamos 'id' en lugar de 'ID'
    id_col = 'id' if 'id' in df.columns else ('ID' if 'ID' in df.columns else None)
    
    validation = {
        'shape': df.shape,
        'duplicates': df.duplicated().sum(),
        'unique_ids': df[id_col].nunique() if id_col else "No ID column found",
        'null_counts': df.isnull().sum().to_dict()
    }
    return validation

if __name__ == "__main__":
    # Ruta a los archivos cargados con Git LFS
    data_folder = "data/raw/" 
    
    try:
        # 1. Cargar
        df_raw = load_all_data(data_folder)
        
        if not df_raw.empty:
            # 2. Limpiar
            df_final = clean_dataset(df_raw)
            
            # 3. Validar
            resumen = validate_dataset(df_final)
            
            print("\n--- Resumen de Validación ---")
            print(f"Filas y Columnas: {resumen['shape']}")
            print(f"Duplicados: {resumen['duplicates']}")
            print(f"IDs únicos: {resumen['unique_ids']}")
            # print("Nulos por columna:", resumen['null_counts'])
            
    except Exception as e:
        print(f"Error crítico: {e}")