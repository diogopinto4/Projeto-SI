"""
Scraper para recolher preços de produtos do Continente Online.

Estratégia de recolha:
    1. Construir tarefas de scraping a partir de categorias e/ou queries livres.
    2. Para cada tarefa, percorrer páginas de listagem com paginação offset-based
       (parâmetros ``start`` e ``sz`` na query string).
    3. Extrair o JSON embutido no atributo ``data-product-tile-impression`` de
       cada product tile — fonte primária para id, nome, marca, categoria e preço.
    4. Complementar com preço original (desconto), preço unitário (€/kg), URL e
       imagem através de seletores CSS secundários.
    5. Normalizar todos os campos, desduplicar por ``id_externo`` e guardar em
       JSON e/ou CSV no diretório de output.

Paragem automática:
    O scraping de uma tarefa termina assim que uma página devolve 0 produtos,
    evitando pedidos desnecessários para além do catálogo disponível.

Uso:
    python continente_scraper.py --categorias mercearia
    python continente_scraper.py --categorias mercearia bebidas --paginas 10
    python continente_scraper.py --queries "arroz,massa,azeite" --formato ambos
    python continente_scraper.py --categorias mercearia --output-dir /tmp/dados
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import NamedTuple
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    build_product_record,
    clean_text,
    is_valid_current_price,
    make_session,
    normalize_category,
    normalize_price_string,
    normalize_unit_price,
    save_to_csv,
    save_to_json,
)

# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------

BASE_URL = "https://www.continente.pt/pesquisa/"
"""URL base para pesquisa livre por termo."""

CATEGORIES: dict[str, str] = {
    "mercearia":   "https://www.continente.pt/mercearia/",
    "bebidas":     "https://www.continente.pt/bebidas-e-garrafeira/",
    "laticinios":  "https://www.continente.pt/laticinios-e-ovos/",
    "congelados":  "https://www.continente.pt/congelados/",
}
"""Mapeamento de categorias disponíveis para o respetivo URL de listagem.

As categorias foram escolhidas para emparelhar com as cadeias concorrentes
(Auchan ``alimentacao``, Pingo Doce ``produtos/*``) e dar ao recomendador um
espaço de produtos comparável entre as 3 cadeias. ``frescos`` foi excluído
porque os preços variam diariamente por gramagem (produtos sem código SKU
estável), o que rebenta a normalização chave-mestre.
"""

PAGE_SIZE   = 35   # produtos devolvidos por página (limite suportado pelo site)
MAX_RETRIES = 3    # tentativas por pedido HTTP antes de desistir
TIMEOUT     = 30   # timeout em segundos por pedido HTTP

OUTPUT_DIR = Path(__file__).parent / "output"


def _stable_url_id(prefix: str, url: str) -> str:
    """Gera um identificador determinístico quando o tile não expõe SKU."""
    if not url:
        return ""
    digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}-{digest}"


# ---------------------------------------------------------------------------
# Tipo auxiliar para tarefas de scraping
# ---------------------------------------------------------------------------

class ScrapeTask(NamedTuple):
    """Representa uma tarefa de scraping: um label, o URL de listagem e a query opcional."""

    label: str
    url: str
    query: str | None


# ---------------------------------------------------------------------------
# Fetch HTTP com retry e backoff exponencial
# ---------------------------------------------------------------------------

def fetch_listing_page(
    session: requests.Session,
    url: str,
    start: int = 0,
    page_size: int = PAGE_SIZE,
    query: str | None = None,
) -> str:
    """Faz GET a uma página de listagem paginada.

    Tenta até ``MAX_RETRIES`` vezes com backoff exponencial entre tentativas.
    Na última tentativa, propaga a exceção para o chamador.

    Args:
        session: Sessão HTTP reutilizável (com headers de browser).
        url: URL base da listagem (categoria ou pesquisa).
        start: Offset de paginação (número do primeiro produto a devolver).
        page_size: Número de produtos por página.
        query: Termo de pesquisa livre (apenas para URLs de pesquisa).

    Returns:
        Conteúdo HTML da resposta como string.

    Raises:
        requests.RequestException: Se todas as tentativas falharem.
    """
    params: dict[str, str | int] = {"start": start, "sz": page_size}
    if query:
        params["q"] = query

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = session.get(url, params=params, timeout=TIMEOUT)
            response.raise_for_status()
            return response.text
        except requests.RequestException as exc:
            if attempt < MAX_RETRIES:
                wait = 1.5 * attempt + random.uniform(0.2, 0.8)
                print(f"    [RETRY {attempt}/{MAX_RETRIES}] {exc} — a aguardar {wait:.1f}s")
                time.sleep(wait)
            else:
                raise


# ---------------------------------------------------------------------------
# Extração de campos do HTML de cada product tile
# ---------------------------------------------------------------------------

def extract_tile_json(tile: BeautifulSoup) -> dict:
    """Extrai os dados estruturados embutidos no atributo do tile.

    O Continente Online serializa id, nome, marca, categoria e preço no
    atributo ``data-product-tile-impression`` de cada tile de produto.

    Args:
        tile: Elemento BeautifulSoup do product tile.

    Returns:
        Dicionário com os campos do produto, ou ``{}`` se o atributo
        estiver ausente ou o JSON for inválido.
    """
    raw = tile.get("data-product-tile-impression", "{}")
    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        return {}


def extract_old_price_from_tile(tile: BeautifulSoup) -> str:
    """Extrai o preço original (antes de desconto) do tile, se presente.

    Tenta vários seletores CSS por ordem de especificidade, parando no
    primeiro que devolva um preço válido.

    Args:
        tile: Elemento BeautifulSoup do product tile.

    Returns:
        Preço original como string decimal (ex: ``"2.99"``), ou ``""``
        se não estiver presente ou for inválido.
    """
    selectors = [
        ".prices-wrapper .list",
        ".pwc-tile--price-old",
        '[class*="old-price"]',
        '[class*="list-price"]',
    ]
    for sel in selectors:
        el = tile.select_one(sel)
        if not el:
            continue
        price = normalize_price_string(el.get_text(" ", strip=True))
        if is_valid_current_price(price):
            return price
    return ""


def extract_unit_price_from_tile(tile: BeautifulSoup) -> str:
    """Extrai o preço unitário (€/kg, €/l, etc.) do tile, se presente.

    Args:
        tile: Elemento BeautifulSoup do product tile.

    Returns:
        Preço unitário normalizado (ex: ``"1.69€/kg"``), ou ``""`` se ausente.
    """
    selectors = [
        ".pwc-tile--price-secondary",
        '[class*="price-secondary"]',
        '[class*="unit-price"]',
    ]
    for sel in selectors:
        el = tile.select_one(sel)
        if not el:
            continue
        value = normalize_unit_price(el.get_text(" ", strip=True))
        if value:
            return value
    return ""


def extract_product_url_from_tile(tile: BeautifulSoup) -> str:
    """Extrai o URL absoluto da página de produto a partir do tile.

    Dá preferência a âncoras cujo ``href`` contenha ``/produto/``.

    Args:
        tile: Elemento BeautifulSoup do product tile.

    Returns:
        URL absoluto do produto, ou ``""`` se não encontrado.
    """
    for sel in ['a[href*="/produto/"]', "a.link", "a[href]"]:
        for el in tile.select(sel):
            href = el.get("href", "")
            if not href:
                continue
            if "/produto/" in href:
                return href if href.startswith("http") else urljoin("https://www.continente.pt", href)
    return ""


def extract_product_image_from_tile(tile: BeautifulSoup) -> str:
    """Extrai o URL da imagem principal do produto.

    Analisa atributos ``src``, ``data-src`` e ``srcset`` de todas as
    imagens no tile. Filtra badges promocionais (que aparecem como imagens
    sobrepostas ao produto) preferindo imagens reais.

    Args:
        tile: Elemento BeautifulSoup do product tile.

    Returns:
        URL absoluto da imagem, ou ``""`` se nenhuma imagem válida for encontrada.
    """
    candidates: list[str] = []
    base = "https://www.continente.pt"

    for img in tile.select("img"):
        for attr in ("src", "data-src"):
            val = img.get(attr)
            if val:
                candidates.append(val if val.startswith("http") else urljoin(base, val))

        srcset = img.get("srcset")
        if srcset:
            first = srcset.split(",")[0].strip().split(" ")[0]
            if first:
                candidates.append(first if first.startswith("http") else urljoin(base, first))

    # Descartar badges promocionais (caminhos que contenham "/badges/")
    real_images = [u for u in candidates if "/badges/" not in u.lower()]
    return real_images[0] if real_images else ""


# ---------------------------------------------------------------------------
# Parsing de produtos
# ---------------------------------------------------------------------------

def parse_product_tile(
    tile: BeautifulSoup,
    run_timestamp: str,
    debug: bool = False,
) -> dict | None:
    """Constrói o registo completo de um produto a partir de um tile HTML.

    Usa o JSON embutido como fonte primária e os seletores CSS como
    fontes secundárias para campos não disponíveis no JSON.

    Args:
        tile: Elemento BeautifulSoup do product tile.
        run_timestamp: Timestamp ISO da execução (preenchido em todos os registos).
        debug: Se ``True``, imprime motivos de rejeição de produtos.

    Returns:
        Dicionário com o registo do produto no schema do projeto, ou ``None``
        se o produto não tiver nome ou preço válido.
    """
    impression    = extract_tile_json(tile)
    product_id    = clean_text(impression.get("id"))
    nome          = clean_text(impression.get("name"))
    marca         = clean_text(impression.get("brand"))
    categoria     = normalize_category(impression.get("category"))
    preco         = normalize_price_string(impression.get("price"))
    preco_original = extract_old_price_from_tile(tile)
    preco_unitario = extract_unit_price_from_tile(tile)
    url           = extract_product_url_from_tile(tile)
    imagem        = extract_product_image_from_tile(tile)
    if not product_id and url:
        product_id = _stable_url_id("continente-url", url)

    if not nome:
        if debug:
            print("[DEBUG] Tile ignorado — sem nome.")
        return None

    if not is_valid_current_price(preco):
        if debug:
            print(f"[DEBUG] Tile ignorado — preço inválido. nome={nome!r} preco={preco!r}")
        return None

    # Se preço original == preço atual, não é desconto real — descartar
    if preco_original and preco_original == preco:
        preco_original = ""

    return build_product_record(
        id_externo=product_id,
        nome=nome,
        marca=marca,
        categoria=categoria,
        preco=preco,
        preco_original=preco_original,
        preco_unitario=preco_unitario,
        url=url,
        imagem=imagem,
        loja="Continente",
        data_recolha=run_timestamp,
    )


def parse_products_html(
    html: str,
    run_timestamp: str,
    debug: bool = False,
) -> list[dict]:
    """Extrai todos os produtos de uma página de listagem HTML.

    Args:
        html: Conteúdo HTML de uma página de resultados.
        run_timestamp: Timestamp ISO da execução.
        debug: Se ``True``, ativa logs de debug detalhados.

    Returns:
        Lista de registos de produtos (pode ser vazia).
    """
    soup     = BeautifulSoup(html, "html.parser")
    products = []
    errors   = 0

    for tile in soup.select("div.product-tile"):
        try:
            record = parse_product_tile(tile, run_timestamp=run_timestamp, debug=debug)
            if record:
                products.append(record)
        except Exception as exc:
            errors += 1
            if debug:
                print(f"  [ERRO] Falha no parsing de tile: {exc}")

    if errors and not debug:
        print(f"  [AVISO] {errors} tile(s) com erro de parsing (usa --debug para detalhes).")

    return products


# ---------------------------------------------------------------------------
# Scraping de listagens
# ---------------------------------------------------------------------------

def scrape_listing(
    session: requests.Session,
    task: ScrapeTask,
    max_pages: int,
    run_timestamp: str,
    debug: bool = False,
) -> list[dict]:
    """Percorre todas as páginas de uma listagem e devolve os produtos.

    A recolha para automaticamente quando uma página devolve 0 produtos
    (sinal de que o catálogo foi esgotado antes de atingir ``max_pages``).

    Args:
        session: Sessão HTTP reutilizável.
        task: Tarefa de scraping com label, URL e query opcional.
        max_pages: Limite máximo de páginas a recolher.
        run_timestamp: Timestamp ISO da execução.
        debug: Se ``True``, ativa logs detalhados.

    Returns:
        Lista de registos de produtos recolhidos em todas as páginas.
    """
    all_products: list[dict] = []
    separator = "=" * 60

    print(f"\n{separator}")
    print(f"  Tarefa: {task.label!r}")
    if task.query:
        print(f"  Query:  {task.query!r}")
    print(separator)

    for page in range(max_pages):
        start = page * PAGE_SIZE
        print(f"  Página {page + 1}/{max_pages} (offset={start})...", end=" ", flush=True)

        try:
            html     = fetch_listing_page(session, task.url, start=start, query=task.query)
            products = parse_products_html(html, run_timestamp=run_timestamp, debug=debug)

            if not products:
                print("sem produtos — fim da recolha.")
                break

            all_products.extend(products)
            print(f"{len(products)} produtos.")
            time.sleep(random.uniform(0.6, 1.2))

        except requests.RequestException as exc:
            print(f"\n  [ERRO] Página {page + 1} falhou após {MAX_RETRIES} tentativas: {exc}")
            break

    return all_products


def build_tasks(
    queries: str | None,
    categorias: list[str] | None,
) -> list[ScrapeTask]:
    """Constrói a lista de tarefas de scraping a partir dos argumentos CLI.

    Args:
        queries: String de queries separadas por vírgula (ex: ``"arroz,massa"``),
                 ou ``None`` se não forem especificadas.
        categorias: Lista de categorias a recolher (chaves de ``CATEGORIES``),
                    ou ``None`` se não forem especificadas.

    Returns:
        Lista de ``ScrapeTask`` prontas a ser executadas. Nunca contém duplicados.
    """
    tasks: list[ScrapeTask] = []

    if categorias:
        for cat in categorias:
            if cat not in CATEGORIES:
                print(f"  [AVISO] Categoria desconhecida ignorada: {cat!r}. Opções: {list(CATEGORIES)}")
                continue
            tasks.append(ScrapeTask(label=cat, url=CATEGORIES[cat], query=None))

    if queries:
        for q in (q.strip() for q in queries.split(",") if q.strip()):
            tasks.append(ScrapeTask(label=q, url=BASE_URL, query=q))

    return tasks


def deduplicate_by_id(products: list[dict]) -> list[dict]:
    """Remove produtos duplicados mantendo apenas o primeiro registo por ``id_externo``.

    Duplicados ocorrem quando o mesmo produto aparece em mais do que uma
    categoria scrapeada na mesma execução.

    Args:
        products: Lista de registos de produtos (pode ter duplicados).

    Returns:
        Lista sem duplicados, na ordem original de aparecimento.
    """
    seen: set[str] = set()
    unique: list[dict] = []
    for p in products:
        key = p.get("id_externo", "")
        if key and key not in seen:
            seen.add(key)
            unique.append(p)
        elif not key:
            unique.append(p)  # sem id, mantém por precaução
    return unique


# ---------------------------------------------------------------------------
# API pública para integração com agentes SPADE
# ---------------------------------------------------------------------------

def discover_scrape_tasks(
    queries: str | None,
    categorias: list[str] | None,
) -> list[ScrapeTask]:
    """Alias para build_tasks — interface usada pelo ContinenteScraper agent.

    Args:
        queries: String de queries separadas por vírgula, ou ``None``.
        categorias: Lista de categorias (chaves de ``CATEGORIES``), ou ``None``.

    Returns:
        Lista de ``ScrapeTask`` prontas a executar.
    """
    return build_tasks(queries, categorias)


def scrape_tasks(
    session: requests.Session,
    tasks: list[ScrapeTask],
    paginas: int,
    run_timestamp: str,
    debug: bool = False,
) -> list[dict]:
    """Executa uma lista de tarefas de scraping e devolve todos os produtos.

    Interface de alto nível usada pelo ContinenteScraper agent para
    desacoplar a descoberta de tarefas da execução do scraping.

    Args:
        session: Sessão HTTP reutilizável.
        tasks: Lista devolvida por ``discover_scrape_tasks``.
        paginas: Máximo de páginas por tarefa.
        run_timestamp: Timestamp ISO da execução.
        debug: Se ``True``, ativa logs detalhados.

    Returns:
        Lista de produtos recolhidos, deduplicada por ``id_externo``.
    """
    all_products: list[dict] = []
    for task in tasks:
        products = scrape_listing(
            session,
            task=task,
            max_pages=paginas,
            run_timestamp=run_timestamp,
            debug=debug,
        )
        all_products.extend(products)
    return deduplicate_by_id(all_products)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    """Ponto de entrada CLI: parseia argumentos e executa o scraping standalone."""
    parser = argparse.ArgumentParser(
        description="Scraper de preços do Continente Online",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  python continente_scraper.py --categorias mercearia
  python continente_scraper.py --categorias mercearia bebidas --paginas 10
  python continente_scraper.py --queries "arroz,massa,azeite" --formato ambos
  python continente_scraper.py --categorias mercearia --output-dir /dados/continente
        """,
    )
    parser.add_argument(
        "--queries", type=str, metavar="TERMOS",
        help="Termos de pesquisa separados por vírgula (ex: 'arroz,massa,azeite')",
    )
    parser.add_argument(
        "--categorias", nargs="+", choices=list(CATEGORIES.keys()),
        metavar="CAT",
        help=f"Categorias a recolher: {', '.join(CATEGORIES.keys())}",
    )
    parser.add_argument(
        "--paginas", type=int, default=3,
        help=f"Máximo de páginas por tarefa — cada página tem {PAGE_SIZE} produtos (default: %(default)s)",
    )
    parser.add_argument(
        "--formato", choices=["csv", "json", "ambos"], default="json",
        help="Formato do ficheiro de saída (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=OUTPUT_DIR, metavar="DIR",
        help=f"Diretório de saída (default: {OUTPUT_DIR})",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Ativar logs de debug detalhados (produtos ignorados, erros de parsing)",
    )
    args = parser.parse_args()

    if not args.queries and not args.categorias:
        parser.error("Indica pelo menos --queries ou --categorias.")

    tasks = build_tasks(args.queries, args.categorias)
    if not tasks:
        print("Nenhuma tarefa encontrada. Verifica os argumentos.")
        return

    # Um único timestamp para toda a execução — usado nos dados e no nome do ficheiro
    run_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    file_timestamp = run_timestamp.replace(":", "").replace("-", "").replace(" ", "_")

    session      = make_session()
    all_products: list[dict] = []
    task_stats:   list[tuple[str, int]] = []

    for task in tasks:
        products = scrape_listing(
            session,
            task=task,
            max_pages=args.paginas,
            run_timestamp=run_timestamp,
            debug=args.debug,
        )
        task_stats.append((task.label, len(products)))
        all_products.extend(products)

    # Desduplicar caso o mesmo produto apareça em várias categorias
    before = len(all_products)
    all_products = deduplicate_by_id(all_products)
    duplicates   = before - len(all_products)

    if not all_products:
        print("\nNenhum produto encontrado. Verifica a ligação ou os seletores CSS.")
        return

    output_dir = args.output_dir
    if args.formato in ("csv", "ambos"):
        save_to_csv(all_products, f"continente_{file_timestamp}.csv", output_dir)
    if args.formato in ("json", "ambos"):
        save_to_json(all_products, f"continente_{file_timestamp}.json", output_dir)

    print(f"\n{'=' * 60}")
    print(f"  Resumo da recolha ({run_timestamp})")
    print(f"{'=' * 60}")
    for label, count in task_stats:
        print(f"  {label:<20}: {count} produtos")
    if duplicates:
        print(f"  Duplicados removidos    : {duplicates}")
    print(f"  Total guardado          : {len(all_products)} produtos")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
