import pandas as pd
import numpy as np
import glob
import os

def optimize_memory(df):
    """Optimiza tipos numéricos y convierte objetos repetitivos a categorías."""
    original_mem = df.memory_usage(deep=True).sum() / 1024**2
    df_opt = df.copy()

    # 1. Optimizar Números
    for col in df_opt.select_dtypes(include=['number']).columns:
        c_min = df_opt[col].min()
        c_max = df_opt[col].max()
        if pd.api.types.is_integer_dtype(df_opt[col]):
            if c_min >= np.iinfo(np.int8).min and c_max <= np.iinfo(np.int8).max:
                df_opt[col] = df_opt[col].astype(np.int8)
            elif c_min >= np.iinfo(np.int16).min and c_max <= np.iinfo(np.int16).max:
                df_opt[col] = df_opt[col].astype(np.int16)
            else:
                df_opt[col] = df_opt[col].astype(np.int32)
        else:
            df_opt[col] = df_opt[col].astype(np.float32)

    # 2. Optimizar Categóricos (Crucial para cosméticos)
    # Si una columna de texto tiene pocos valores únicos en comparación con el total, es candidata a 'category'
    for col in df_opt.select_dtypes(include=['object']).columns:
        num_unique = df_opt[col].nunique()
        num_total = len(df_opt)
        if num_unique / num_total < 0.5:  # Si menos del 50% son valores únicos
            df_opt[col] = df_opt[col].astype('category')

    final_mem = df_opt.memory_usage(deep=True).sum() / 1024**2
    print(f"Memory: {original_mem:.1f}MB -> {final_mem:.1f}MB (Salvaste {100*(original_mem-final_mem)/original_mem:.1f}%)")
    return df_opt

def load_and_optimize_all(folder_path):
    """Carga todos los archivos y los optimiza uno por uno para no saturar la RAM."""
    all_files = glob.glob(os.path.join(folder_path, "*.csv"))
    df_list = []
    
    for filename in all_files:
        print(f"Cargando y optimizando: {os.path.basename(filename)}")
        chunk = pd.read_csv(filename)
        chunk_opt = optimize_memory(chunk)
        df_list.append(chunk_opt)
        
    full_df = pd.concat(df_list, ignore_index=True)
    return full_df