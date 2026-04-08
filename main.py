# main.py
# Main entry point

import os
import pandas as pd
from pathlib import Path
from src.audit import audit_data
from src.optimization import optimize_memory
from src.pipeline import build_preprocessing_pipeline


def main():
    """Main orchestration script for the cosmetics data pipeline."""
    # Ensure working directory is the script's directory
    os.chdir(Path(__file__).parent)

    print('--- Starting Data Pipeline ---\n')

    try:
        # 1. Auditoría
        if not audit_data():
            print('\nPipeline stopped due to audit failure.')
            return

        # 2. Carga dinámica
        raw_dir = Path('data/raw')
        csv_files = list(raw_dir.glob('*.csv'))
        if not csv_files:
            raise FileNotFoundError('No CSV file found in data/raw')

        csv_file = csv_files[0]
        print(f'\nLoading raw data from {csv_file.name}...')
        df_raw = pd.read_csv(csv_file, sep=None, engine='python')

        # 3. Optimización
        print('\nOptimizing memory...')
        df_opt = optimize_memory(df_raw)

        # 4. Pipeline
        print('\nBuilding and applying preprocessing pipeline...')
        leakage_columns = ['duration'] if 'duration' in df_opt.columns else []
        pipeline = build_preprocessing_pipeline(columns_to_drop=leakage_columns)

        processed_matrix = pipeline.fit_transform(df_opt)

        # 5. Guardado
        print('\nSaving processed dataset...')
        feature_names = pipeline.named_steps['preprocessing'].get_feature_names_out()
        feature_names = [name.replace('num__', '').replace('cat__', '') for name in feature_names]

        df_processed = pd.DataFrame(processed_matrix, columns=feature_names)

        processed_dir = Path('data/processed')
        processed_dir.mkdir(parents=True, exist_ok=True)
        output_path = processed_dir / 'processed_data.csv'

        df_processed.to_csv(output_path, index=False)
        print(f'SUCCESS: Processed dataset saved at {output_path}')
        print(f'Final dimensions: {df_processed.shape}')

    except IndexError:
        print('\nCRITICAL ERROR: No se encontró ningún archivo CSV en la carpeta \'data/raw\'.')
    except FileNotFoundError as e:
        print(f'\nCRITICAL ERROR: Archivo o directorio no encontrado: {e}')
    except pd.errors.EmptyDataError:
        print('\nCRITICAL ERROR: El archivo CSV está completamente vacío.')
    except pd.errors.ParserError:
        print('\nCRITICAL ERROR: Pandas no pudo leer el CSV. Revisa si hay comas o separadores rotos en los datos.')
    except Exception as e:
        print(f'\nFATAL ERROR: El pipeline falló inesperadamente: {e}')


if __name__ == '__main__':
    main()