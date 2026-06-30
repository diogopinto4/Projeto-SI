"""
Configuração da ligação à base de dados PostgreSQL.

As variáveis de ambiente são carregadas do ficheiro ``.env`` na raiz do projeto.
Valores por omissão correspondem ao Docker Compose local (``docker-compose.yml``).

Variáveis suportadas:
    DB_HOST     (default: "localhost")
    DB_NAME     (default: "products_db")
    DB_USER     (default: "postgres")
    DB_PASSWORD (default: "postgres")
    DB_PORT     (default: "5432")

Uso::

    from db_config import DB_CONFIG
    conn = psycopg2.connect(**DB_CONFIG)
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

#: Dicionário de ligação passado directamente a ``psycopg2.connect(**DB_CONFIG)``.
DB_CONFIG: dict[str, str] = {
    "host":     os.getenv("DB_HOST",     "localhost"),
    "database": os.getenv("DB_NAME",     "products_db"),
    "user":     os.getenv("DB_USER",     "postgres"),
    "password": os.getenv("DB_PASSWORD", "postgres"),
    "port":     os.getenv("DB_PORT",     "5432"),
}
