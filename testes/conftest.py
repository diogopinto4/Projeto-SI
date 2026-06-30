"""
Fixtures partilhadas por todos os módulos de teste.

Inclui:
  - Registos de produto de exemplo (schema comum do projeto).
  - HTML sintético de páginas Auchan com e sem JSON-LD.
  - DataFrame de histórico de preços mínimo para testes de pipeline.
  - Fixtures de base de dados de teste (ver secção BD ao final do ficheiro).

A BD de teste é uma instância PostgreSQL separada (``products_db_test``) que
deve ser criada uma vez antes de correr os testes::

    docker exec products-db psql -U postgres -c "CREATE DATABASE products_db_test;"
    docker exec -i products-db psql -U postgres -d products_db_test < sql/schema.sql

Os testes que precisam de BD pedem a fixture ``db_clean`` (limpa todas as
tabelas antes do teste). Testes que precisam apenas de conexão pedem ``db_conn``.

NOTA TÉCNICA — override da BD via variável de ambiente:
    O ``db_config.py`` lê o nome da BD via ``os.getenv("DB_NAME", "products_db")``.
    Definimos ``DB_NAME=products_db_test`` aqui **antes** de qualquer import do
    projeto, garantindo que todos os módulos (incluindo os que importam via
    ``sys.path.insert + from db_config import DB_CONFIG``) vêem o valor de teste.
    Esta abordagem é mais robusta que monkey-patching de ``DB_CONFIG`` porque o
    projeto tem caminhos de import duplicados que registam ``db_config`` como
    dois módulos distintos em ``sys.modules`` — patches num não afectam o outro.
"""

from __future__ import annotations

import os

# IMPORTANTE: definir antes de qualquer import do projeto.
os.environ.setdefault("DB_NAME", "products_db_test")

import sys
from pathlib import Path

import pandas as pd
import psycopg2
import pytest

# Adicionar raiz ao path para que os módulos do projeto sejam importáveis
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scrapers"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "models"))


# ---------------------------------------------------------------------------
# Produto de exemplo no schema comum
# ---------------------------------------------------------------------------

@pytest.fixture
def produto_simples():
    """Registo de produto básico sem promoção nem EAN."""
    return {
        "id_externo":     "12345",
        "nome":           "Arroz Agulha Cigala 1kg",
        "marca":          "Cigala",
        "categoria":      "alimentacao/mercearia",
        "preco":          "1.69",
        "preco_original": "",
        "preco_unitario": "1.69€/kg",
        "url":            "https://www.auchan.pt/pt/alimentacao/mercearia/arroz/arroz-agulha/12345.html",
        "imagem":         "https://www.auchan.pt/img/12345.jpg",
        "loja":           "Auchan",
        "data_recolha":   "2026-04-18 10:00:00",
        "ean":            "",
    }


@pytest.fixture
def produto_em_promocao():
    """Registo de produto em promoção com EAN."""
    return {
        "id_externo":     "98765",
        "nome":           "Azeite Virgem Extra Gallo 750ml",
        "marca":          "Gallo",
        "categoria":      "alimentacao/azeite",
        "preco":          "4.49",
        "preco_original": "5.99",
        "preco_unitario": "5.99€/l",
        "url":            "https://www.continente.pt/produto/azeite-gallo-98765",
        "imagem":         "https://www.continente.pt/img/98765.jpg",
        "loja":           "Continente",
        "data_recolha":   "2026-04-18 10:00:00",
        "ean":            "5601252014578",
    }


@pytest.fixture
def produto_pack():
    """Registo de produto em pack (3x250g)."""
    return {
        "id_externo":     "11111",
        "nome":           "Iogurte Natural 3x250g",
        "marca":          "Mimosa",
        "categoria":      "frescos/laticínios",
        "preco":          "1.99",
        "preco_original": "",
        "preco_unitario": "2.65€/kg",
        "url":            "https://www.pingodoce.pt/produto/iogurte-11111",
        "imagem":         "",
        "loja":           "Pingo Doce",
        "data_recolha":   "2026-04-18 09:30:00",
        "ean":            "",
    }


@pytest.fixture
def batch_produtos(produto_simples, produto_em_promocao, produto_pack):
    """Batch de três produtos de lojas diferentes."""
    return [produto_simples, produto_em_promocao, produto_pack]


# ---------------------------------------------------------------------------
# HTML sintético para testes de scraping
# ---------------------------------------------------------------------------

@pytest.fixture
def html_com_json_ld():
    """Página HTML com JSON-LD de produto Auchan completo."""
    return """<!DOCTYPE html>
<html>
<head>
  <meta property="og:image" content="https://www.auchan.pt/og/12345.jpg">
  <script type="application/ld+json">
  {
    "@context": "https://schema.org",
    "@type": "Product",
    "name": "Arroz Agulha Cigala 1kg",
    "sku": "12345",
    "gtin": "5601234567890",
    "brand": {"@type": "Brand", "name": "Cigala"},
    "image": ["https://www.auchan.pt/img/12345.jpg"],
    "offers": {
      "@type": "Offer",
      "price": "1.69",
      "priceCurrency": "EUR",
      "availability": "https://schema.org/InStock"
    }
  }
  </script>
</head>
<body>
  <h1>Arroz Agulha Cigala 1kg</h1>
  <span itemprop="price" content="1.69">1,69 €</span>
  <div class="price-per-unit">1,69 €/kg</div>
</body>
</html>"""


@pytest.fixture
def html_sem_json_ld():
    """Página HTML sem JSON-LD — apenas HTML com itemprop."""
    return """<!DOCTYPE html>
<html>
<head></head>
<body>
  <h1>Leite Mimosa 1l</h1>
  <span class="sales value">0,85 €</span>
  <span itemprop="price" content="0.85">0,85 €</span>
</body>
</html>"""


@pytest.fixture
def html_sem_preco():
    """Página HTML sem preço reconhecível."""
    return """<!DOCTYPE html>
<html>
<head></head>
<body>
  <h1>Produto Sem Preço</h1>
</body>
</html>"""


@pytest.fixture
def html_sem_nome():
    """Página HTML sem nome de produto."""
    return """<!DOCTYPE html>
<html>
<head></head>
<body>
  <span itemprop="price" content="2.99">2,99 €</span>
</body>
</html>"""


# ---------------------------------------------------------------------------
# DataFrame de histórico mínimo para testes do pipeline de dataset
# ---------------------------------------------------------------------------

@pytest.fixture
def df_historico_minimo():
    """DataFrame com duas séries de 3 dias para testes do pipeline."""
    return pd.DataFrame({
        "id_historico":       [1, 2, 3, 4, 5, 6],
        "dia":                pd.to_datetime([
            "2026-01-01", "2026-01-02", "2026-01-03",
            "2026-01-01", "2026-01-02", "2026-01-03",
        ]),
        "id_produto_loja":    [1, 1, 1, 2, 2, 2],
        "id_produto_mestre":  [10, 10, 10, 20, 20, 20],
        "id_loja":            [1, 1, 1, 2, 2, 2],
        "cadeia":             ["Continente"] * 3 + ["Pingo Doce"] * 3,
        "canal":              ["online"] * 6,
        "nome_produto":       ["Arroz Agulha"] * 3 + ["Leite Mimosa"] * 3,
        "marca":              ["Cigala"] * 3 + ["Mimosa"] * 3,
        "categoria_geral":    ["alimentacao"] * 3 + ["frescos"] * 3,
        "quantidade_valor":   [1000.0] * 3 + [1000.0] * 3,
        "quantidade_unidade": ["g"] * 3 + ["ml"] * 3,
        "preco_unitario_valor":   [1.69] * 3 + [0.85] * 3,
        "preco_unitario_unidade": ["kg"] * 3 + ["l"] * 3,
        "preco_atual":        [1.69, 1.69, 1.75, 0.85, 0.82, 0.85],
        "preco_original":     [None, None, 1.99, None, None, None],
        "em_promocao":        [False, False, True, False, False, False],
    })


# ===========================================================================
# Base de dados de teste (PostgreSQL)
# ===========================================================================
# Padrão usado:
#
#   1. ``_override_db_config`` (autouse, scope="session"): muda DB_CONFIG para
#      apontar para ``products_db_test`` no início da sessão de testes.
#      Restaura no final. Como os módulos do projeto fazem
#      ``psycopg2.connect(**DB_CONFIG)``, a mutação in-place do dict afecta
#      todas as ligações abertas durante os testes.
#
#   2. ``db_conn`` (scope="function"): abre uma ligação à BD de teste.
#      Os testes que só precisam de fazer SELECTs podem usar esta.
#
#   3. ``db_clean`` (scope="function"): além de abrir ligação, faz TRUNCATE
#      em todas as tabelas antes do teste — garante isolamento entre testes.
#
# Para correr os testes BD:
#     docker exec products-db psql -U postgres -c "CREATE DATABASE products_db_test;"
#     docker exec -i products-db psql -U postgres -d products_db_test < sql/schema.sql
#     pytest testes/ -m "" --override-ini="markers=" -v
# ---------------------------------------------------------------------------

#: Nome da BD de teste. Definido também via env var ``DB_NAME`` no topo deste
#: módulo, antes de qualquer import — ver nota técnica no docstring.
TEST_DB_NAME = os.environ["DB_NAME"]


@pytest.fixture(scope="session", autouse=True)
def _verificar_db_teste():
    """Valida defensivamente que estamos apontados à BD de teste, não à de produção.

    Aborta a sessão de testes com erro claro se algum import de produção tiver
    conseguido "fixar" outra BD por engano. É uma rede de segurança — em
    funcionamento normal, ``DB_NAME`` foi definido antes de qualquer import.
    """
    from scripts.db_config import DB_CONFIG
    assert DB_CONFIG["database"] == TEST_DB_NAME, (
        f"ATENÇÃO: DB_CONFIG aponta para {DB_CONFIG['database']!r} em vez de "
        f"{TEST_DB_NAME!r}. Os testes BD escreveriam na BD de produção! "
        "Verifica que DB_NAME está definido no topo de testes/conftest.py."
    )
    yield


def _ligar_test_db():
    """Abre ligação à BD de teste; salta o teste se não existir."""
    from scripts.db_config import DB_CONFIG
    try:
        return psycopg2.connect(**DB_CONFIG)
    except psycopg2.OperationalError as exc:
        pytest.skip(
            f"BD de teste '{TEST_DB_NAME}' não acessível ({exc}). "
            "Cria-a com:\n"
            f"  docker exec products-db psql -U postgres -c 'CREATE DATABASE {TEST_DB_NAME};'\n"
            f"  docker exec -i products-db psql -U postgres -d {TEST_DB_NAME} < sql/schema.sql"
        )


@pytest.fixture
def db_conn():
    """Ligação à BD de teste, fechada no final do teste.

    Use esta fixture quando o teste apenas faz SELECTs ou quer controlar
    explicitamente o estado da BD. Para isolamento automático, prefere
    :func:`db_clean`.
    """
    conn = _ligar_test_db()
    try:
        yield conn
    finally:
        conn.close()


#: Tabelas a limpar antes de cada teste (ordem com CASCADE preserva FKs).
_TABELAS_TESTE = (
    "historico_precos",
    "precos_atuais",
    "produtos_loja",
    "produtos_mestre",
    "lojas",
    "lojas_fisicas",
)


@pytest.fixture
def db_clean(db_conn):
    """BD de teste limpa antes do teste (TRUNCATE em todas as tabelas).

    Reinicia também as sequências PRIMARY KEY para que cada teste arranque
    com IDs determinísticos (id_loja=1 para a primeira loja inserida, etc.).

    **Limpa também a cache in-memory de geolocation.lojas_proximas** — sem
    isto, testes que inserem dados frescos podem ver entradas cached de
    testes anteriores (ex: a cache pode dizer "0 lojas no raio" porque foi
    populada antes do INSERT).

    Devolve a mesma ligação ``db_conn`` para o teste poder fazer INSERTs.
    """
    with db_conn.cursor() as cur:
        # TRUNCATE + RESTART IDENTITY: limpa dados e reinicia sequências
        cur.execute(
            f"TRUNCATE TABLE {', '.join(_TABELAS_TESTE)} "
            "RESTART IDENTITY CASCADE"
        )
    db_conn.commit()

    # Invalida a cache in-memory de geolocation — evita falsos positivos
    # entre testes (entradas que cachearam contagens com BD limpa).
    #
    # NOTA: o módulo geolocation pode estar registado em sys.modules sob 2
    # nomes distintos (``models.geolocation`` vs. ``geolocation``) porque
    # outros módulos do projecto fazem ``from geolocation import ...`` via
    # ``sys.path.insert``. Cada instância tem o seu próprio dict de cache.
    # Para isolar testes a 100% precisamos de limpar **as duas**.
    import sys as _sys
    for nome_mod in ("models.geolocation", "geolocation"):
        mod = _sys.modules.get(nome_mod)
        if mod is not None and hasattr(mod, "limpar_cache_lojas_proximas"):
            mod.limpar_cache_lojas_proximas()

    return db_conn


# ---------------------------------------------------------------------------
# Helpers de seed para os testes BD — inserem registos coerentes com o schema.
# ---------------------------------------------------------------------------

@pytest.fixture
def db_seed_basico(db_clean):
    """Insere 1 loja Continente + 1 produto + 1 preço atual.

    Útil para testes que precisam de "ter algo na BD" sem se preocuparem com
    a construção dos registos. Devolve um dict com os IDs criados, para os
    testes poderem referenciar (ex: ``id_loja=ids['id_loja']``).
    """
    conn = db_clean
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO lojas (insignia, formato_loja, localizacao, canal)
            VALUES ('Continente', 'Online', 'Nacional', 'online')
            RETURNING id_loja
        """)
        id_loja = cur.fetchone()[0]

        cur.execute("""
            INSERT INTO produtos_mestre
                (chave_mestre, nome_padronizado, marca,
                 categoria_geral, quantidade_valor, quantidade_unidade)
            VALUES ('ean:5601234567890', 'Arroz Agulha Cigala 1000g',
                    'Cigala', 'mercearia', 1000, 'g')
            RETURNING id_produto_mestre
        """)
        id_produto_mestre = cur.fetchone()[0]

        cur.execute("""
            INSERT INTO produtos_loja
                (id_produto_mestre, id_loja, sku_loja, nome_na_loja,
                 categoria_loja, quantidade_valor, quantidade_unidade)
            VALUES (%s, %s, 'SKU-001', 'Arroz Cigala 1kg',
                    'mercearia/arroz', 1000, 'g')
            RETURNING id_produto_loja
        """, (id_produto_mestre, id_loja))
        id_produto_loja = cur.fetchone()[0]

        cur.execute("""
            INSERT INTO precos_atuais
                (id_produto_loja, preco_atual, em_promocao,
                 data_recolha, agente_origem)
            VALUES (%s, 1.69, FALSE, '2026-05-15 10:00:00+00', 'scraper_continente')
        """, (id_produto_loja,))

    conn.commit()
    return {
        "id_loja":          id_loja,
        "id_produto_mestre": id_produto_mestre,
        "id_produto_loja":  id_produto_loja,
    }


@pytest.fixture
def db_seed_3_cadeias(db_clean):
    """Insere o mesmo produto em 3 cadeias com preços diferentes.

    - Continente: 1.69 €
    - Pingo Doce: 1.79 €
    - Auchan:     1.59 € (mais barato)

    Permite testar ``recomendar_melhor_loja`` e ``otimizar_lista_compras``
    sem dependência de dados reais.
    """
    conn = db_clean

    cadeias_precos = [
        ("Continente", 1.69),
        ("Pingo Doce", 1.79),
        ("Auchan",     1.59),
    ]
    ids_por_cadeia: dict[str, dict] = {}

    with conn.cursor() as cur:
        # Produto-mestre partilhado
        cur.execute("""
            INSERT INTO produtos_mestre
                (chave_mestre, nome_padronizado, marca,
                 categoria_geral, quantidade_valor, quantidade_unidade)
            VALUES ('ean:5601234567890', 'Arroz Agulha Cigala 1000g',
                    'Cigala', 'mercearia', 1000, 'g')
            RETURNING id_produto_mestre
        """)
        id_pm = cur.fetchone()[0]

        for i, (insignia, preco) in enumerate(cadeias_precos, start=1):
            cur.execute("""
                INSERT INTO lojas (insignia, formato_loja, localizacao, canal)
                VALUES (%s, 'Online', 'Nacional', 'online')
                RETURNING id_loja
            """, (insignia,))
            id_loja = cur.fetchone()[0]

            cur.execute("""
                INSERT INTO produtos_loja
                    (id_produto_mestre, id_loja, sku_loja, nome_na_loja,
                     quantidade_valor, quantidade_unidade)
                VALUES (%s, %s, %s, 'Arroz Cigala 1kg', 1000, 'g')
                RETURNING id_produto_loja
            """, (id_pm, id_loja, f"SKU-{i:03d}"))
            id_pl = cur.fetchone()[0]

            cur.execute("""
                INSERT INTO precos_atuais
                    (id_produto_loja, preco_atual, em_promocao,
                     data_recolha, agente_origem)
                VALUES (%s, %s, FALSE, '2026-05-15 10:00:00+00', %s)
            """, (id_pl, preco, f"scraper_{insignia.lower().replace(' ', '_')}"))

            ids_por_cadeia[insignia] = {
                "id_loja":         id_loja,
                "id_produto_loja": id_pl,
                "preco":           preco,
            }

    conn.commit()
    return {"id_produto_mestre": id_pm, **ids_por_cadeia}


@pytest.fixture
def db_seed_lojas_fisicas(db_clean):
    """Insere 4 lojas físicas — 2 em Braga, 1 em Porto, 1 em Lisboa.

    Coordenadas escolhidas para serem realistas. Útil para testar
    :func:`models.geolocation.lojas_proximas` com um ponto conhecido (UMinho).
    """
    conn = db_clean
    lojas = [
        # (insignia, nome, lat, lon, cidade)
        ("Continente", "Continente Braga",      41.5409, -8.4004, "Braga"),
        ("Pingo Doce", "Pingo Doce Braga Hiper", 41.5572, -8.4050, "Braga"),
        ("Auchan",     "Auchan Porto Boavista",  41.1579, -8.6291, "Porto"),
        ("Continente", "Continente Lisboa Colombo", 38.7544, -9.1875, "Lisboa"),
    ]
    ids = []
    with conn.cursor() as cur:
        for i, (insignia, nome, lat, lon, cidade) in enumerate(lojas, start=1):
            cur.execute("""
                INSERT INTO lojas_fisicas
                    (insignia, nome_loja, latitude, longitude, cidade, fonte, external_id)
                VALUES (%s, %s, %s, %s, %s, 'test', %s)
                RETURNING id_loja_fisica
            """, (insignia, nome, lat, lon, cidade, f"test-{i}"))
            ids.append(cur.fetchone()[0])

    conn.commit()
    return ids
