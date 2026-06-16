"""api.database
===============
Conexión SQLAlchemy a PostgreSQL usando variables de entorno.

Provee:
- ``engine``: instancia global del motor SQLAlchemy.
- ``get_db()``: generador de sesiones para usar como dependencia en FastAPI.

Uso en un endpoint
------------------
    from api.database import get_db
    from sqlalchemy.orm import Session

    @router.get("/ejemplo")
    def ejemplo(db: Session = Depends(get_db)):
        ...
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session

load_dotenv()

# ---------------------------------------------------------------------------
# URL de conexión construida desde variables de entorno
# ---------------------------------------------------------------------------

POSTGRES_USER     = os.getenv("POSTGRES_USER", "sephora_user")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "changeme_123")
POSTGRES_HOST     = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT     = os.getenv("POSTGRES_PORT", "5432")
POSTGRES_DB       = os.getenv("POSTGRES_DB", "sephora_dw")

DATABASE_URL = (
    f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}"
    f"@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
)

# ---------------------------------------------------------------------------
# Motor y fábrica de sesiones
# ---------------------------------------------------------------------------

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,   # verifica la conexión antes de usarla
    pool_size=5,
    max_overflow=10,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


# ---------------------------------------------------------------------------
# Dependencia para FastAPI
# ---------------------------------------------------------------------------

def get_db():
    """Generador de sesiones SQLAlchemy para inyección de dependencias.

    Garantiza que la sesión se cierre al finalizar cada request,
    incluso si ocurre una excepción.

    Yields
    ------
    Session
        Sesión activa de SQLAlchemy.
    """
    db: Session = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Verificación de conexión (útil en startup)
# ---------------------------------------------------------------------------

def check_connection() -> bool:
    """Verifica que la base de datos es accesible.

    Returns
    -------
    bool
        True si la conexión es exitosa, False si falla.
    """
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False
