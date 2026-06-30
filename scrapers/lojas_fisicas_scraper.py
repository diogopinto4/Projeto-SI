"""
Scraper de lojas físicas das cadeias de supermercado (Continente, Pingo Doce, Auchan).

Recolhe nome, morada, localidade, código postal, telefone e — crucial para a
funcionalidade de custo de deslocação — as coordenadas GPS (latitude/longitude)
em sistema WGS84.

Fontes e estratégias por cadeia:

1. **Continente** — endpoint SFCC público ``Stores-FindStores``. Devolve JSON
   com todas as lojas quando se passa um centro (lat/long) e um raio amplo
   (radius=300 km cobre Portugal continental + ilhas).

2. **Pingo Doce** — mesmo padrão SFCC (também é Salesforce Commerce Cloud).
   Requer header ``Accept: application/json`` para evitar que o servidor
   devolva HTML em vez de JSON (pelo mesmo endpoint).

3. **Auchan** — sem endpoint AJAX equivalente. As lojas estão embebidas no
   HTML da página ``/pt/lojas`` como atributo ``data-locations`` (JSON array
   com HTML entities escapadas).

Schema comum de saída (cada loja é um dict):
    {
        "insignia":      "Continente" | "Pingo Doce" | "Auchan",
        "nome_loja":     str,
        "morada":        str,
        "codigo_postal": str | None,
        "cidade":        str | None,
        "distrito":      str | None,
        "latitude":      float,
        "longitude":     float,
        "telefone":      str | None,
        "horario":       str | None,
        "external_id":   str,    # ID da loja no site original
        "fonte":         "scraper:continente" | "scraper:pingo_doce" | "scraper:auchan",
    }

Uso::

    # Recolher todas as cadeias
    python scrapers/lojas_fisicas_scraper.py

    # Recolher apenas uma cadeia
    python scrapers/lojas_fisicas_scraper.py --cadeia continente
    python scrapers/lojas_fisicas_scraper.py --cadeia pingo_doce
    python scrapers/lojas_fisicas_scraper.py --cadeia auchan

    # Output personalizado
    python scrapers/lojas_fisicas_scraper.py --output-dir /tmp/lojas
"""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import requests
from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).parent))
from utils import HTTP_TIMEOUT, clean_text, fetch_text, make_session, save_to_json


# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------

OUTPUT_DIR = Path(__file__).parent / "output"

#: Centro geográfico aproximado de Portugal continental, usado como ponto de
#: referência nas queries SFCC com raio amplo. Coordenadas: Coimbra-ish.
PT_CENTRO_LAT = 40.0
PT_CENTRO_LON = -8.0

#: Raios usados para cobrir todo o território nacional. 500 km a partir de
#: Coimbra cobre confortavelmente continente + Açores (mais distantes) e Madeira.
PT_RADIUS_KM = 500

#: Bounding box do território português. Usada para filtrar coordenadas
#: claramente inválidas (ex: lat/lon zero, ou lojas espanholas que possam
#: aparecer por engano nas APIs SFCC).
PT_LAT_MIN, PT_LAT_MAX = 30.0, 43.5     # Madeira (30°N) até Minho (43°N)
PT_LON_MIN, PT_LON_MAX = -32.0, -6.0    # Açores (-32°W) até raia espanhola (-6°E)

#: Endpoints SFCC dos dois sites Salesforce Commerce Cloud.
CONTINENTE_STORES_ENDPOINT = (
    "https://www.continente.pt/on/demandware.store/Sites-continente-Site/default"
    f"/Stores-FindStores?lat={PT_CENTRO_LAT}&long={PT_CENTRO_LON}&radius={PT_RADIUS_KM}"
)
PINGO_DOCE_STORES_ENDPOINT = (
    "https://www.pingodoce.pt/on/demandware.store/Sites-pingo-doce-Site/default"
    f"/Stores-FindStores?lat={PT_CENTRO_LAT}&long={PT_CENTRO_LON}&radius={PT_RADIUS_KM}"
)

#: URL da página de lojas do Auchan. As lojas estão no atributo
#: data-locations do HTML como JSON array escapado.
AUCHAN_STORES_URL = (
    "https://www.auchan.pt/pt/lojas?showMap=true&horizontalView=true&isForm=true"
)


# ---------------------------------------------------------------------------
# Continente — SFCC Stores-FindStores
# ---------------------------------------------------------------------------

def scrape_continente_stores(session: requests.Session) -> list[dict]:
    """Recolhe todas as lojas Continente via endpoint SFCC.

    O endpoint ``Stores-FindStores`` aceita um centro (lat/long) e um raio
    e devolve todas as lojas dentro desse raio. Com (40, -8, 500 km) cobre
    todo o território nacional incluindo ilhas.

    Args:
        session: Sessão HTTP reutilizável.

    Returns:
        Lista de dicionários no schema comum.

    Raises:
        requests.RequestException: Se o pedido falhar.
        json.JSONDecodeError: Se a resposta não for JSON válido.
    """
    print(f"[Continente] A recolher lojas de {CONTINENTE_STORES_ENDPOINT}")
    response = session.get(CONTINENTE_STORES_ENDPOINT, timeout=HTTP_TIMEOUT)
    response.raise_for_status()
    data = response.json()

    raw_stores = data.get("stores", [])
    print(f"[Continente] {len(raw_stores)} lojas recebidas. A normalizar...")

    return [_normalize_sfcc_store(s, insignia="Continente", fonte="scraper:continente")
            for s in raw_stores
            if _has_valid_coords(s)]


# ---------------------------------------------------------------------------
# Pingo Doce — SFCC Stores-FindStores (mesmo padrão)
# ---------------------------------------------------------------------------

def scrape_pingo_doce_stores(session: requests.Session) -> list[dict]:
    """Recolhe todas as lojas Pingo Doce via endpoint SFCC.

    O Pingo Doce usa exactamente a mesma plataforma SFCC do Continente,
    incluindo o mesmo endpoint ``Stores-FindStores``. A única diferença é
    que **requer header ``Accept: application/json``** para forçar resposta
    JSON em vez do HTML default do site.

    Args:
        session: Sessão HTTP reutilizável.

    Returns:
        Lista de dicionários no schema comum.

    Raises:
        requests.RequestException: Se o pedido falhar.
        json.JSONDecodeError: Se a resposta não for JSON válido.
    """
    print(f"[Pingo Doce] A recolher lojas de {PINGO_DOCE_STORES_ENDPOINT}")
    headers = {
        "Accept": "application/json",
        "X-Requested-With": "XMLHttpRequest",   # belt-and-suspenders
    }
    response = session.get(
        PINGO_DOCE_STORES_ENDPOINT, headers=headers, timeout=HTTP_TIMEOUT,
    )
    response.raise_for_status()
    data = response.json()

    raw_stores = data.get("stores", [])
    print(f"[Pingo Doce] {len(raw_stores)} lojas recebidas. A normalizar...")

    return [_normalize_sfcc_store(s, insignia="Pingo Doce", fonte="scraper:pingo_doce")
            for s in raw_stores
            if _has_valid_coords(s)]


# ---------------------------------------------------------------------------
# Auchan — HTML embebido com data-locations
# ---------------------------------------------------------------------------

def scrape_auchan_stores(session: requests.Session) -> list[dict]:
    """Recolhe todas as lojas Auchan a partir do HTML da página de lojas.

    O Auchan não expõe um endpoint AJAX público. Em vez disso, embebe um
    array JSON com todas as lojas no atributo ``data-locations`` de um
    elemento HTML na página ``/pt/lojas``. As entities HTML têm de ser
    desescapadas antes do parsing JSON.

    A morada e ID interno (``data-store-id``) estão dentro de um sub-HTML
    no campo ``infoWindowHtml`` de cada loja — usamos BeautifulSoup para
    extrair essa informação.

    Args:
        session: Sessão HTTP reutilizável.

    Returns:
        Lista de dicionários no schema comum.

    Raises:
        requests.RequestException: Se o pedido falhar.
        ValueError: Se o atributo data-locations não for encontrado.
    """
    print(f"[Auchan] A recolher página {AUCHAN_STORES_URL}")
    html_text = fetch_text(session, AUCHAN_STORES_URL)

    # data-locations="[{...}, ...]" — JSON array com HTML entities escapadas
    match = re.search(r'data-locations="(\[.+?\])"', html_text, re.DOTALL)
    if not match:
        raise ValueError(
            "Atributo data-locations não encontrado na página Auchan. "
            "A estrutura do site pode ter mudado."
        )

    payload = html.unescape(match.group(1))
    raw_stores = json.loads(payload)
    print(f"[Auchan] {len(raw_stores)} lojas no data-locations. A normalizar...")

    normalized = []
    for store in raw_stores:
        try:
            n = _normalize_auchan_store(store)
        except Exception as exc:
            # Loja individual com formato inesperado — saltar com log e continuar.
            print(f"[Auchan] Loja ignorada ({store.get('name', '?')}): {exc}")
            continue
        if n is not None:
            normalized.append(n)

    return normalized


# ---------------------------------------------------------------------------
# Normalização — SFCC (Continente + Pingo Doce)
# ---------------------------------------------------------------------------

def _has_valid_coords(store: dict) -> bool:
    """Verifica se a loja tem coordenadas GPS válidas (não-nulas e dentro de PT).

    Lojas sem coordenadas são silenciosamente descartadas. O CHECK constraint
    do schema rejeitaria os registos, mas validar aqui dá feedback mais útil.
    """
    lat = store.get("latitude")
    lon = store.get("longitude")
    if lat is None or lon is None:
        return False
    try:
        lat_f, lon_f = float(lat), float(lon)
    except (TypeError, ValueError):
        return False
    # Bounding box generosa de Portugal (ver constantes PT_LAT_*/PT_LON_*).
    return (PT_LAT_MIN <= lat_f <= PT_LAT_MAX
            and PT_LON_MIN <= lon_f <= PT_LON_MAX)


def _normalize_sfcc_store(store: dict, insignia: str, fonte: str) -> dict:
    """Converte um dict de loja do formato SFCC para o schema comum.

    Args:
        store: Loja como devolvida pelo endpoint Stores-FindStores.
        insignia: Nome da cadeia ("Continente" ou "Pingo Doce").
        fonte: Identificador da origem ("scraper:continente" etc.).

    Returns:
        Dicionário no schema comum (ver docstring do módulo).
    """
    morada_parts = [clean_text(store.get("address1")), clean_text(store.get("address2"))]
    morada = ", ".join(p for p in morada_parts if p)

    return {
        "insignia":      insignia,
        "nome_loja":     clean_text(store.get("name")) or "(sem nome)",
        "morada":        morada or None,
        "codigo_postal": clean_text(store.get("postalCode")) or None,
        "cidade":        clean_text(store.get("city")) or None,
        # SFCC usa "stateCode" para distrito/região
        "distrito":      clean_text(store.get("stateCode")) or None,
        "latitude":      float(store["latitude"]),
        "longitude":     float(store["longitude"]),
        "telefone":      clean_text(store.get("phone")) or None,
        "horario":       clean_text(store.get("storeHours")) or None,
        "external_id":   str(store.get("ID") or "").strip(),
        "fonte":         fonte,
    }


# ---------------------------------------------------------------------------
# Normalização — Auchan
# ---------------------------------------------------------------------------

#: Regex para extrair código postal português (NNNN-NNN) de texto livre.
CP_RE = re.compile(r"\b(\d{4}-\d{3})\b")


def _normalize_auchan_store(store: dict) -> dict | None:
    """Converte um dict de loja Auchan para o schema comum.

    O Auchan tem uma estrutura particular: ``name`` é o nome da loja,
    ``latitude``/``longitude`` vêm diretos, mas a morada e o ID interno
    estão dentro de ``infoWindowHtml`` (um snippet HTML por loja).

    Args:
        store: Loja como aparece no array data-locations.

    Returns:
        Dicionário no schema comum, ou ``None`` se a loja não tiver
        coordenadas válidas.
    """
    if not _has_valid_coords(store):
        return None

    nome = clean_text(store.get("name")) or "(sem nome)"
    info_html = store.get("infoWindowHtml") or ""

    morada, codigo_postal, cidade, external_id = _parse_auchan_info_window(info_html)

    return {
        "insignia":      "Auchan",
        "nome_loja":     nome,
        "morada":        morada,
        "codigo_postal": codigo_postal,
        "cidade":        cidade,
        "distrito":      None,   # Auchan não expõe distrito separadamente
        "latitude":      float(store["latitude"]),
        "longitude":     float(store["longitude"]),
        "telefone":      None,   # Não disponível no data-locations
        "horario":       None,
        "external_id":   external_id,
        "fonte":         "scraper:auchan",
    }


def _parse_auchan_info_window(info_html: str) -> tuple[str | None, str | None, str | None, str]:
    """Extrai morada, código postal, cidade e ID do snippet HTML do Auchan.

    O snippet tem estrutura como::

        <div class="store-details" data-store-id="B19">
            <div class="store-name">My Auchan D Dinis - Odivelas</div>
            <address>
                Rua D. Dinis, n° 39 B
                Odivelas, Odivelas  2675-332
            </address>
            ...
        </div>

    Args:
        info_html: Snippet HTML de uma loja.

    Returns:
        Tupla (morada, codigo_postal, cidade, external_id).
        Qualquer campo pode ser ``None`` se ausente. ``external_id`` é
        string vazia se ``data-store-id`` não estiver presente.
    """
    soup = BeautifulSoup(info_html, "html.parser")

    # ID interno
    store_div = soup.select_one("[data-store-id]")
    external_id = store_div.get("data-store-id", "") if store_div else ""

    # Morada — está no elemento <address>
    address_el = soup.select_one("address")
    if not address_el:
        return None, None, None, external_id

    address_text = clean_text(address_el.get_text(" ", strip=True))
    if not address_text:
        return None, None, None, external_id

    # Extrair código postal (NNNN-NNN)
    cp_match = CP_RE.search(address_text)
    codigo_postal = cp_match.group(1) if cp_match else None

    # Morada e cidade — heurística: o CP separa a morada da cidade.
    # Em geral o formato é "Rua X, Localidade, Concelho CCCC-CCC".
    # Removemos CP e ficamos com "Rua X, Localidade, Concelho".
    sem_cp = CP_RE.sub("", address_text).strip(" ,")

    # Última vírgula → tudo antes é morada, último item é cidade
    if "," in sem_cp:
        morada, _, cidade = sem_cp.rpartition(",")
        morada = clean_text(morada)
        cidade = clean_text(cidade)
    else:
        morada = sem_cp
        cidade = None

    return morada or None, codigo_postal, cidade, external_id


# ---------------------------------------------------------------------------
# Orquestração — recolher todas as cadeias
# ---------------------------------------------------------------------------

#: Mapeamento cadeia → função de recolha. Permite acrescentar novas cadeias
#: sem modificar a função main.
CADEIAS_DISPONIVEIS = {
    "continente": scrape_continente_stores,
    "pingo_doce": scrape_pingo_doce_stores,
    "auchan":     scrape_auchan_stores,
}


def scrape_all(session: requests.Session, cadeias: list[str]) -> list[dict]:
    """Executa o scraping de todas as cadeias indicadas e agrega os resultados.

    Falhas numa cadeia não impedem o scraping das outras (são reportadas com
    aviso). Isto é importante para resiliência: se o Pingo Doce mudar o seu
    site, o seed do Continente/Auchan continua a funcionar.

    Args:
        session: Sessão HTTP reutilizável.
        cadeias: Lista de chaves de ``CADEIAS_DISPONIVEIS`` a recolher.

    Returns:
        Lista agregada de todas as lojas recolhidas, em qualquer ordem.
    """
    all_stores: list[dict] = []
    for cadeia in cadeias:
        scrape_fn = CADEIAS_DISPONIVEIS.get(cadeia)
        if scrape_fn is None:
            print(f"[AVISO] Cadeia desconhecida: {cadeia!r}. Disponíveis: {list(CADEIAS_DISPONIVEIS)}")
            continue
        try:
            stores = scrape_fn(session)
            print(f"[{cadeia}] {len(stores)} lojas normalizadas com sucesso.\n")
            all_stores.extend(stores)
        except Exception as exc:
            print(f"[ERRO] Falha ao recolher {cadeia}: {exc}\n")

    return all_stores


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    """Ponto de entrada CLI: parseia argumentos e executa o scraping."""
    parser = argparse.ArgumentParser(
        description="Scraper de lojas físicas (Continente, Pingo Doce, Auchan).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  python scrapers/lojas_fisicas_scraper.py
  python scrapers/lojas_fisicas_scraper.py --cadeia continente
  python scrapers/lojas_fisicas_scraper.py --cadeia auchan --output-dir /tmp/lojas

Volumes esperados (totais a nível nacional):
  Continente : ~206 lojas (inclui Continente, Modelo, Bom Dia, Zu)
  Pingo Doce : ~270 lojas
  Auchan     : ~563 lojas (inclui hipermercados e My Auchan)
        """,
    )
    parser.add_argument(
        "--cadeia",
        choices=list(CADEIAS_DISPONIVEIS.keys()),
        help="Recolher apenas uma cadeia (default: todas).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        metavar="DIR",
        help=f"Diretório de saída (default: {OUTPUT_DIR}).",
    )
    args = parser.parse_args()

    cadeias = [args.cadeia] if args.cadeia else list(CADEIAS_DISPONIVEIS.keys())

    print("=" * 60)
    print(f"  Scraper de lojas físicas — {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"  Cadeias: {', '.join(cadeias)}")
    print("=" * 60 + "\n")

    session = make_session()
    stores = scrape_all(session, cadeias)

    if not stores:
        print("[AVISO] Nenhuma loja recolhida. Verifica a ligação à rede.")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_to_json(stores, f"lojas_fisicas_{timestamp}.json", args.output_dir, label="lojas")

    # Resumo por cadeia
    print()
    print("=" * 60)
    print("  Resumo da recolha")
    print("=" * 60)
    for cadeia in cadeias:
        insignia_map = {"continente": "Continente", "pingo_doce": "Pingo Doce", "auchan": "Auchan"}
        ins = insignia_map[cadeia]
        n = sum(1 for s in stores if s["insignia"] == ins)
        print(f"  {ins:15s}: {n} lojas")
    print(f"  {'TOTAL':15s}: {len(stores)} lojas")
    print("=" * 60)


if __name__ == "__main__":
    main()
