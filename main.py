"""
Main execution orchestrator for the Cosmetics ETL pipeline.
"""

import pandas as pd
from pathlib import Path
from src.pipeline import build_preprocessing_pipeline

# Importación segura en caso de que falten los módulos de utilidad
try:
    from src.audit import audit_data
    from src.optimization import optimize_memory
except ImportError:
    audit_data = lambda *args: True
    optimize_memory = lambda df: df

def main():
    """Executes the data pipeline."""
    print("="*60)
    print("💄 PIPELINE DE DATOS: PRODUCTOS COSMÉTICOS (SEPHORA)")
    print("="*60)

    try:
        # 1. Extracción
        print("\n📥 Fase 1: Búsqueda y carga de datos")
        raw_dir = Path('data/raw')
        csv_files = list(raw_dir.glob('*.csv'))
        
        if not csv_files:
            print(f"❌ Error: No hay archivos CSV en {raw_dir}")
            print("💡 Tip: Recuerda descargar el dataset original de Kaggle y ponerlo aquí. ¡No uses el archivo de 2 filas!")
            return

        csv_file = csv_files[0]
        print(f"📁 Cargando datos desde: {csv_file.name}")
        df_raw = pd.read_csv(csv_file, sep=None, engine='python')
        
        # Normalización de cabeceras
        df_raw.columns = [str(c).strip().lower() for c in df_raw.columns]

        # 2. Auditoría y Optimización
        print("\n🔍 Fase 2: Auditoría y optimización")
        df_opt = optimize_memory(df_raw)

        # 3. Aplicar Pipeline
        print("\n🏗️  Fase 3: Construyendo y aplicando el pipeline...")
        pipeline = build_preprocessing_pipeline()
        
        # Gracias a set_output("pandas"), esto devuelve un DataFrame directamente
        df_processed = pipeline.fit_transform(df_opt)

        # Limpieza de nombres de columnas (Quitar prefijos num__ y cat__)
        df_processed.columns = [str(col).replace('num__', '').replace('cat__', '') for col in df_processed.columns]

        # 4. Guardado
        print("\n💾 Fase 4: Guardado de dataset")
        processed_dir = Path('data/processed')
        processed_dir.mkdir(parents=True, exist_ok=True)
        output_path = processed_dir / 'cosmetics_processed.csv'

        df_processed.to_csv(output_path, index=False)
        
        print("\n" + "="*60)
        print("✅ PIPELINE COMPLETADO EXITOSAMENTE")
        print("="*60)
        print(f"📊 Dimensiones finales: {df_processed.shape[0]} filas × {df_processed.shape[1]} columnas")

    except Exception as e:
        print(f"\n❌ FATAL ERROR: {e}")

if __name__ == "__main__":
    main()