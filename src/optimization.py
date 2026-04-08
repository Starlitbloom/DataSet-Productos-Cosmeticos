"""
Memory Optimization Module.
Reduces DataFrame memory footprint and processes large files in chunks.
"""

import pandas as pd
import numpy as np


def optimize_memory(df):
    """Reduces the memory usage of a DataFrame by downcasting numeric types."""
    try:
        original_mem = df.memory_usage(deep=True).sum() / 1024**2
        print(f"Original memory usage: {original_mem:.2f} MB")
        df_opt = df.copy()

        for col in df_opt.select_dtypes(include=['int64', 'int32', 'int16', 'int8', 'float64', 'float32']).columns:
            try:
                orig_type = df_opt[col].dtype
                c_min = df_opt[col].min(skipna=True)
                c_max = df_opt[col].max(skipna=True)

                if str(orig_type).startswith('int'):
                    if c_min >= np.iinfo(np.int8).min and c_max <= np.iinfo(np.int8).max:
                        df_opt[col] = df_opt[col].astype(np.int8)
                    elif c_min >= np.iinfo(np.int16).min and c_max <= np.iinfo(np.int16).max:
                        df_opt[col] = df_opt[col].astype(np.int16)
                    elif c_min >= np.iinfo(np.int32).min and c_max <= np.iinfo(np.int32).max:
                        df_opt[col] = df_opt[col].astype(np.int32)

                elif str(orig_type).startswith('float'):
                    if c_min >= np.finfo(np.float32).min and c_max <= np.finfo(np.float32).max:
                        df_opt[col] = df_opt[col].astype(np.float32)

            except Exception as e:
                print(f"⚠️ WARNING: No se pudo optimizar la columna '{col}': {e}")
                continue

        final_mem = df_opt.memory_usage(deep=True).sum() / 1024**2
        savings = 100 * (original_mem - final_mem) / original_mem if original_mem else 0

        print(f"Optimized memory usage: {final_mem:.2f} MB")
        print(f"Total savings: {savings:.1f}%")
        return df_opt

    except Exception as e:
        print(f"CRITICAL ERROR in memory optimization: {e}")
        return df


def optimize_csv_in_chunks(source_path, dest_path, chunk_size=10000):
    """Procesa un CSV grande en bloques y guarda una versión optimizada."""
    try:
        reader = pd.read_csv(source_path, chunksize=chunk_size, na_values=['NULL', 'null', ''])
        optimized_chunks = []

        for i, chunk in enumerate(reader, start=1):
            print(f"🔄 Procesando chunk {i}")
            chunk_opt = optimize_memory(chunk)
            optimized_chunks.append(chunk_opt)

        result = pd.concat(optimized_chunks, ignore_index=True)
        result.to_csv(dest_path, index=False)
        print(f"✅ Archivo optimizado guardado en: {dest_path}")
        return result

    except Exception as e:
        print(f"❌ ERROR optimizando CSV en chunks: {e}")
        return None
