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
        print(f'ERROR: Fallo al leer el archivo para auditar: {e}')
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
    """Audita todos los archivos CSV en data/raw y gestiona sus hashes."""
    try:
        raw_dir = Path('data/raw')
        csv_files = list(raw_dir.glob('*.csv'))
        metadata_path = raw_dir / 'metadata.json'

        if not csv_files:
            print('ERROR: No se encontraron archivos .csv')
            return False

        current_audit = {}
        all_match = True

        for file_path in csv_files:
            print(f'Auditando: {file_path.name}...')
            f_hash = generate_file_hash(file_path)
            meta = extract_dataset_metadata(file_path)
            
            current_audit[file_path.name] = {
                'hash_sha256': f_hash,
                'rows': meta['rows'],
                'cols': meta['columns']
            }

        if metadata_path.exists():
            with open(metadata_path, 'r', encoding='utf-8') as f:
                saved_metadata = json.load(f)

            # Verificar cada archivo guardado contra el actual
            for filename, data in current_audit.items():
                if filename not in saved_metadata:
                    print(f'{filename} no estaba en la auditoría anterior.')
                    all_match = False
                    continue
                
                if saved_metadata[filename]['hash_sha256'] != data['hash_sha256']:
                    print(f'Hash mismatch en {filename}!')
                    all_match = False
            
            if all_match:
                print('Integridad total verificada.')
            return all_match
        else:
            # Si no existe, creamos el registro inicial para todos
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(current_audit, f, indent=4)
            print('Metadata creada para todos los archivos.')
            return True

    except Exception as e:
        print(f'ERROR: {e}')
        return False