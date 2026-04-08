"""
Data Audit Module.
Verifies data integrity and provenance using metadata and SHA-256 hashing.
"""

import hashlib
import json
from pathlib import Path
import pandas as pd


def generate_file_hash(file_path):
    """Genera una firma SHA-256 para el archivo leyendo en bloques."""
    try:
        sha256_hash = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for byte_block in iter(lambda: f.read(4096), b''):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    except IOError as e:
        print(f'❌ ERROR: Fallo al leer el archivo para auditar: {e}')
        return None


def extract_dataset_metadata(file_path):
    """Extrae metadatos útiles del dataset para auditar y documentar."""
    df = pd.read_csv(file_path, na_values=['NULL', 'null', ''])
    return {
        'file': Path(file_path).name,
        'rows': int(df.shape[0]),
        'columns': int(df.shape[1]),
        'column_names': df.columns.tolist(),
        'missing_counts': df.isna().sum().to_dict(),
    }


def audit_data():
    """Audita el archivo CSV en data/raw y crea o verifica metadata.json."""
    try:
        raw_dir = Path('data/raw')
        csv_files = list(raw_dir.glob('*.csv'))

        if not csv_files:
            print('❌ ERROR: No se encontró ningún archivo .csv en data/raw/')
            return False

        target_file = csv_files[0]
        metadata_path = raw_dir / 'metadata.json'

        print(f'Auditando el archivo: {target_file.name}')
        calculated_hash = generate_file_hash(target_file)
        if not calculated_hash:
            return False

        dataset_metadata = extract_dataset_metadata(target_file)

        if metadata_path.exists():
            with open(metadata_path, 'r', encoding='utf-8') as f:
                saved_metadata = json.load(f)

            if saved_metadata.get('hash_sha256') == calculated_hash:
                print('SUCCESS: Data integrity verified. File has not been altered.')
                return True
            else:
                print('CRITICAL ERROR: Hash mismatch. The dataset has been modified or corrupted.')
                print(f'  - hash en disco: {calculated_hash}')
                print(f'  - hash en metadata: {saved_metadata.get("hash_sha256")}')
                return False
        else:
            new_metadata = {
                'file': dataset_metadata['file'],
                'hash_sha256': calculated_hash,
                'rows': dataset_metadata['rows'],
                'columns': dataset_metadata['columns'],
                'column_names': dataset_metadata['column_names'],
                'missing_counts': dataset_metadata['missing_counts'],
            }
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(new_metadata, f, indent=4, ensure_ascii=False)
            print('Metadata inicial creada correctamente. Agrega este archivo al control de versiones.')
            return True

    except json.JSONDecodeError:
        print('❌ ERROR: El archivo metadata.json está corrupto y no se puede leer.')
        return False
    except Exception as e:
        print(f'❌ UNEXPECTED ERROR during audit: {e}')
        return False
