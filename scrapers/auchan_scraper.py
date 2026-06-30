"""
Scraper para recolher preços de produtos do Auchan Portugal.

Estratégia de recolha:
    1. Ler o sitemap index (sitemap_index.xml) para obter os sitemaps de produto.
    2. Ler cada sitemap de produto para agregar todos os URLs de produtos.
    3. Filtrar URLs por categoria e/ou subcategoria (inferidas do path da URL).
    4. Para cada produto, extrair o PID (identificador SFCC) a partir da URL.
    5. Tentar o endpoint AJAX SFCC (Product-Variation) que devolve JSON estruturado
       com nome, preço, preço por unidade, marca, stock e imagens.
    6. Em caso de falha do endpoint AJAX, fazer fallback ao parsing HTML da página
       de produto (SSR — preços presentes no HTML sem necessidade de JavaScript).
    7. Normalizar e validar todos os campos.
    8. Guardar no schema comum do projeto (JSON e/ou CSV).

Plataforma:
    Salesforce Commerce Cloud (SFCC / Demandware). URLs do tipo:
    ``/pt/{categoria}/{subcategoria}/{slug}/{pid}.html``
    Endpoint AJAX: ``/on/demandware.store/Sites-AuchanPT-Site/pt_PT/Product-Variation``

Fonte primária de dados (endpoint AJAX SFCC):
    GET /on/demandware.store/Sites-AuchanPT-Site/pt_PT/Product-Variation?pid={pid}&format=ajax
    Devolve JSON com: productName, id, brand, price.sales.value,
    price.list.value, unitPerPrice, images.large[0].url

Filtro por categoria:
    Baseia-se no path da URL de cada produto:
    ``/pt/{categoria}/{subcategoria}/.../{slug}/{pid}.html``
    Exemplos de categorias no auchan.pt: ``alimentacao`` (mercearia),
    ``bebidas-e-garrafeira``, ``produtos-frescos``, ``biologicos-e-alternativas``,
    ``limpeza-e-cuidados-do-lar``, ``saude-e-bem-estar``.

Uso:
    python auchan_scraper.py --categoria alimentacao --formato json
    python auchan_scraper.py --categoria alimentacao --subcategoria arroz --limite 50
    python auchan_scraper.py --categoria bebidas-e-garrafeira --limite 100 --debug
    python auchan_scraper.py --categoria alimentacao --output-dir /dados/auchan
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    HTTP_TIMEOUT,
    build_product_record,
    clean_text,
    fetch_text,
    is_valid_current_price,
    make_session,
    normalize_category,
    normalize_price_string,
    normalize_unit_price,
    parse_xml_urls,
    save_to_csv,
    save_to_json,
)

# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------

BASE_URL = "https://www.auchan.pt"

SITEMAP_INDEX_URL = "https://www.auchan.pt/sitemap_index.xml"
"""URL do sitemap index que lista todos os sub-sitemaps do site."""

AJAX_ENDPOINT = (
    "https://www.auchan.pt/on/demandware.store/Sites-AuchanPT-Site/pt_PT"
    "/Product-Variation?pid={pid}&format=ajax"
)
"""Endpoint SFCC para obter dados de produto em JSON dado um PID."""

OUTPUT_DIR = Path(__file__).parent / "output"


# ---------------------------------------------------------------------------
# Descoberta de URLs via Sitemap XML
# ---------------------------------------------------------------------------

def get_product_sitemaps(session: requests.Session) -> list[str]:
    """Lê o sitemap index e devolve apenas os URLs de sitemaps de produto.

    Os sitemaps de produto são identificados pela presença de ``"product"``
    no seu URL — convenção usada pelo Auchan (ex: ``sitemap_0-product.xml``).

    Args:
        session: Sessão HTTP reutilizável.

    Returns:
        Lista de URLs de sitemaps de produto.
    """
    xml_text     = fetch_text(session, SITEMAP_INDEX_URL)
    all_sitemaps = parse_xml_urls(xml_text)
    return [u for u in all_sitemaps if "product" in u]


def get_product_urls(session: requests.Session, sitemap_urls: list[str]) -> list[str]:
    """Lê cada sitemap de produto e agrega todos os URLs únicos.

    Sitemaps com erro são ignorados com aviso, permitindo que a recolha
    continue com os restantes.

    Args:
        session: Sessão HTTP reutilizável.
        sitemap_urls: Lista de URLs de sitemaps de produto a processar.

    Returns:
        Lista de URLs de produto únicos, pela ordem de descoberta.
    """
    seen: set[str] = set()
    urls: list[str] = []

    for sitemap_url in sitemap_urls:
        print(f"  [SITEMAP] {sitemap_url}")
        try:
            xml_text = fetch_text(session, sitemap_url)
            for url in parse_xml_urls(xml_text):
                if url not in seen:
                    seen.add(url)
                    urls.append(url)
        except Exception as exc:
            print(f"  [ERRO] Falha ao ler sitemap: {exc}")
        time.sleep(random.uniform(0.5, 1.0))

    return urls


def filter_urls(
    urls: list[str],
    categoria: str | None = None,
    subcategoria: str | None = None,
) -> list[str]:
    """Filtra URLs de produto por categoria e/ou subcategoria no path.

    A estrutura esperada dos paths é:
    ``/pt/{categoria}/{subcategoria}/.../{slug}/{pid}.html``

    Args:
        urls: Lista completa de URLs de produto.
        categoria: Segmento de categoria a filtrar (ex: ``"alimentacao"``).
                   ``None`` para não filtrar por categoria.
        subcategoria: Segmento de subcategoria a filtrar (ex: ``"arroz"``).
                      ``None`` para não filtrar por subcategoria.

    Returns:
        Lista filtrada de URLs. Imprime aviso se o resultado for vazio.
    """
    filtered = urls

    if categoria:
        cat      = categoria.strip("/").lower()
        filtered = [u for u in filtered if f"/pt/{cat}/" in u.lower()]

    if subcategoria:
        sub      = subcategoria.strip("/").lower()
        filtered = [u for u in filtered if f"/{sub}/" in u.lower()]

    if not filtered and urls:
        filtros = []
        if categoria:
            filtros.append(f"categoria={categoria!r}")
        if subcategoria:
            filtros.append(f"subcategoria={subcategoria!r}")
        print(f"  [AVISO] Nenhuma URL encontrada com {' e '.join(filtros)}.")
        print("          Verifica os valores — o path deve conter esses segmentos.")

    return filtered


def discover_product_sources(
    session: requests.Session,
    categoria: str | None = None,
    subcategoria: str | None = None,
) -> list[str]:
    """Executa o pipeline completo de descoberta de URLs de produto.

    Sequência: sitemap index → sitemaps de produto → URLs filtradas.

    Args:
        session: Sessão HTTP reutilizável.
        categoria: Filtro de categoria (ex: ``"alimentacao"``).
        subcategoria: Filtro de subcategoria (ex: ``"arroz"``).

    Returns:
        Lista de URLs de produto prontas para scraping.
    """
    print("[INFO] A obter sitemaps de produto...")
    sitemap_urls = get_product_sitemaps(session)
    print(f"       {len(sitemap_urls)} sitemaps encontrados.")

    print("[INFO] A ler URLs de produto...")
    product_urls = get_product_urls(session, sitemap_urls)
    print(f"       {len(product_urls)} URLs no total.")

    filtered = filter_urls(product_urls, categoria=categoria, subcategoria=subcategoria)
    print(f"       {len(filtered)} URLs após filtro.")
    return filtered


# ---------------------------------------------------------------------------
# Extração do PID (identificador SFCC) a partir da URL
# ---------------------------------------------------------------------------

def extract_pid_from_url(url: str) -> str:
    """Extrai o PID (product ID) SFCC a partir do path da URL.

    O PID é o segmento numérico final do path, imediatamente antes de ``.html``.
    Exemplos::

        /pt/alimentar/.../arroz-agulha-1kg/10.html     → "10"
        /pt/bebidas/.../sumo-laranja/3400089123.html    → "3400089123"

    Args:
        url: URL absoluto da página de produto.

    Returns:
        PID como string, ou ``""`` se o padrão não for encontrado.
    """
    match = re.search(r"/(\d+)\.html(?:\?.*)?$", url)
    return match.group(1) if match else ""


# ---------------------------------------------------------------------------
# Extração de categoria a partir da URL
# ---------------------------------------------------------------------------

def category_from_url(url: str) -> str:
    """Infere a hierarquia de categoria a partir do path da URL.

    Extrai os segmentos entre ``/pt/`` e o slug final (penúltimo segmento,
    que é o nome do produto) e o PID (último segmento ``{pid}.html``).

    Exemplo::

        /pt/alimentar/mercearia/arroz-massas-e-cereais/arroz/arroz-agulha/10.html
        → "alimentar/mercearia/arroz-massas-e-cereais/arroz"

    Args:
        url: URL absoluto da página de produto.

    Returns:
        Hierarquia de categorias com segmentos separados por ``/``,
        ou ``""`` se a estrutura esperada não for encontrada.
    """
    path  = urlparse(url).path.strip("/")
    parts = path.split("/")
    # Estrutura: pt / {categoria} / ... / {slug} / {pid}.html
    # Ignorar "pt" (índice 0), slug do produto (penúltimo) e pid.html (último)
    if len(parts) >= 4 and parts[0] == "pt":
        return "/".join(parts[1:-2])
    return ""


# ---------------------------------------------------------------------------
# Extração via endpoint AJAX SFCC (fonte primária)
# ---------------------------------------------------------------------------

def fetch_product_ajax(session: requests.Session, pid: str) -> dict | None:
    """Tenta obter os dados do produto via endpoint AJAX SFCC.

    Faz GET ao endpoint ``Product-Variation?pid={pid}&format=ajax``.
    Se a resposta for JSON válido com os campos esperados, devolve o dict.
    Se a resposta for HTML (fallback do servidor) ou inválida, devolve ``None``.

    Args:
        session: Sessão HTTP reutilizável.
        pid: Identificador SFCC do produto.

    Returns:
        Dicionário com os dados do produto, ou ``None`` se o endpoint
        não devolver JSON válido com os campos esperados.
    """
    if not pid:
        return None

    url = AJAX_ENDPOINT.format(pid=pid)
    try:
        response = session.get(url, timeout=HTTP_TIMEOUT)
        response.raise_for_status()
        # Verificar Content-Type antes de tentar parsear JSON
        content_type = response.headers.get("Content-Type", "")
        if "json" not in content_type and not response.text.strip().startswith("{"):
            return None
        data = response.json()
        # A resposta SFCC envolve o produto num campo "product" no topo:
        # {"action": "Product-Variation", "product": {...}}. Descer um nível.
        if isinstance(data, dict) and "product" in data:
            data = data["product"]
        # Validar que tem os campos mínimos esperados
        if isinstance(data, dict) and "productName" in data:
            return data
        return None
    except Exception:
        return None


def parse_product_ajax(data: dict, url: str, run_timestamp: str) -> dict | None:
    """Constrói o registo de produto a partir da resposta JSON SFCC.

    Extrai todos os campos do payload JSON devolvido pelo endpoint AJAX.
    O campo ``price.list`` é ``None`` quando o produto não está em promoção,
    pelo que é tratado com segurança.

    Args:
        data: Dicionário JSON devolvido pelo endpoint AJAX SFCC.
        url: URL da página de produto (para contexto e categoria).
        run_timestamp: Timestamp ISO da execução.

    Returns:
        Registo de produto no schema do projeto, ou ``None`` se o nome
        ou o preço estiverem ausentes/inválidos.
    """
    nome = clean_text(data.get("productName", ""))
    if not nome:
        return None

    # ``data.get("id")`` pode ser None — ``str(None)`` daria a string literal
    # "None" e contaminava o id_externo. ``or ""`` mantém limpo nesse caso.
    product_id = clean_text(str(data.get("id") or ""))
    if not product_id:
        product_id = extract_pid_from_url(url)
    if not product_id:
        return None

    # Marca — pode ser string, None, ou ausente.
    # Sem o ``or ""`` defensivo, ``str(None)`` daria "None" e essa string
    # aparecia depois no nome do produto (ex: "1/2 Pá De Borrego Kg None").
    marca = clean_text(data.get("brand") or "")

    # Preço atual (price.sales.value é numérico no SFCC)
    price_block = data.get("price") or {}
    sales_block = price_block.get("sales") or {}
    preco_raw   = sales_block.get("value") or sales_block.get("formatted", "")
    preco       = normalize_price_string(str(preco_raw))

    if not is_valid_current_price(preco):
        return None

    # Preço original — só faz sentido em promoção real.
    #
    # No SFCC do Auchan, ``price.list`` pode estar populado em duas situações
    # diferentes (verificadas via amostragem do AJAX em 2026-05-20):
    #
    #   (a) Promoção activa:  ``isPromotion`` é truthy (1) e ``list.value`` > ``sales.value``
    #   (b) Preço de referência (PVPR/best price recente): ``isPromotion`` é ``false``
    #       mas ``list.value`` continua diferente de ``sales.value`` — não é desconto activo.
    #
    # Sem este gate, o ingest marcava (b) como promoção (porque
    # ``preco_original > preco_atual``) e inflava a contagem de produtos em
    # promoção. Confiar em ``isPromotion`` evita esses falsos positivos.
    preco_original = ""
    if price_block.get("isPromotion"):
        list_block     = price_block.get("list") or {}
        preco_orig_raw = list_block.get("value") or list_block.get("formatted", "")
        preco_original = normalize_price_string(str(preco_orig_raw)) if preco_orig_raw else ""
        # Defesa adicional: se acidentalmente coincidir com o sales, descartar.
        if preco_original and preco_original == preco:
            preco_original = ""

    # Preço unitário (ex: "1.69 €/kg")
    preco_unitario = normalize_unit_price(str(data.get("unitPerPrice", "")))

    # Imagem — SFCC devolve lista por tamanho
    imagem = ""
    images = data.get("images") or {}
    for size in ("large", "medium", "small"):
        imgs = images.get(size, [])
        if imgs and isinstance(imgs, list):
            img_url = imgs[0].get("url", "") if isinstance(imgs[0], dict) else str(imgs[0])
            if img_url:
                imagem = img_url if img_url.startswith("http") else BASE_URL + img_url
                break

    # Categoria inferida do URL
    categoria = normalize_category(category_from_url(url))

    # EAN/GTIN — o SFCC devolve no campo "EAN" (maiúsculas) ao topo
    # ``or ""`` evita que um ``null`` JSON vire a string literal "None".
    ean = clean_text(str(data.get("EAN") or ""))

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
        loja="Auchan",
        data_recolha=run_timestamp,
        ean=ean,
    )


# ---------------------------------------------------------------------------
# Extração via HTML da página de produto (fallback)
# ---------------------------------------------------------------------------

def extract_json_ld(soup: BeautifulSoup) -> dict | None:
    """Extrai o bloco JSON-LD com ``@type: Product`` da página.

    O Auchan inclui JSON-LD schema.org/Product em todas as páginas de produto
    com: name, sku, gtin (EAN), brand.name, image e offers.price.

    Args:
        soup: BeautifulSoup da página de produto.

    Returns:
        Dicionário JSON-LD do bloco Product, ou ``None`` se não encontrado.
    """
    for script in soup.find_all("script", type="application/ld+json"):
        raw = script.get_text(strip=True)
        if not raw:
            continue
        try:
            data = json.loads(raw)
            if not isinstance(data, dict):
                continue
            t = data.get("@type")
            if t == "Product" or (isinstance(t, list) and "Product" in t):
                return data
        except json.JSONDecodeError:
            continue
    return None


def parse_product_html(
    html: str,
    url: str,
    run_timestamp: str,
    debug: bool = False,
) -> dict | None:
    """Constrói o registo de produto a partir do HTML SSR da página.

    Usado como fallback quando o endpoint AJAX não devolve JSON válido.
    O Auchan usa SFCC com SSR — todos os dados estão no HTML inicial
    sem necessidade de JavaScript.

    Hierarquia de fontes por campo:
    - **Nome**: JSON-LD (name) → ``h1``
    - **Marca**: JSON-LD (brand.name)
    - **EAN**: JSON-LD (gtin)
    - **SKU/ID**: JSON-LD (sku) → sufixo numérico da URL
    - **Preço atual**: JSON-LD (offers.price) → ``[itemprop="price"]`` → seletores SFCC
    - **Preço original**: seletores SFCC (.list .strike-through)
    - **Preço unitário**: regex no texto da página
    - **Categoria**: URL path
    - **Imagem**: JSON-LD (image) → ``.primary-image img`` → ``og:image``

    Args:
        html: Conteúdo HTML da página de produto.
        url: URL absoluto da página (usado para categoria e ID fallback).
        run_timestamp: Timestamp ISO da execução.
        debug: Se ``True``, imprime detalhes de parsing para páginas problemáticas.

    Returns:
        Registo de produto no schema do projeto, ou ``None`` se o nome
        ou o preço estiverem ausentes/inválidos.
    """
    soup         = BeautifulSoup(html, "html.parser")
    body_text    = soup.get_text(" ", strip=True)
    product_json = extract_json_ld(soup)

    # --- Nome: JSON-LD → h1 ---
    nome = clean_text(product_json.get("name", "")) if product_json else ""
    if not nome:
        h1 = soup.select_one("h1")
        nome = clean_text(h1.get_text()) if h1 else ""

    if not nome:
        if debug:
            print(f"  [DEBUG] Ignorado — sem nome: {url}")
        return None

    # --- Marca: JSON-LD brand.name ---
    marca = ""
    if product_json:
        brand = product_json.get("brand", {})
        if isinstance(brand, dict):
            marca = clean_text(brand.get("name", ""))
        elif isinstance(brand, str):
            marca = clean_text(brand)

    # --- EAN: JSON-LD gtin ---
    ean = clean_text(str(product_json.get("gtin", ""))) if product_json else ""

    # --- SKU / ID: JSON-LD sku → URL ---
    product_id = clean_text(str(product_json.get("sku", ""))) if product_json else ""
    if not product_id:
        product_id = extract_pid_from_url(url)

    # --- Preço atual: JSON-LD offers.price → itemprop → seletores SFCC ---
    preco = ""
    if product_json:
        offers = product_json.get("offers", {})
        if isinstance(offers, list):
            offers = offers[0] if offers else {}
        preco = normalize_price_string(str(offers.get("price", "")))

    if not is_valid_current_price(preco):
        el = soup.select_one("[itemprop='price']")
        if el:
            preco = normalize_price_string(el.get("content", "") or el.get_text())

    if not is_valid_current_price(preco):
        for sel in [".sales .value", ".price-sales .value", "[data-sales-price]"]:
            el = soup.select_one(sel)
            if el:
                preco = normalize_price_string(el.get("content", "") or el.get_text(" ", strip=True))
                if is_valid_current_price(preco):
                    break

    if not is_valid_current_price(preco):
        if debug:
            print(f"  [DEBUG] Ignorado — preço inválido: {url}")
        return None

    # --- ID: rejeitar produto se não conseguir extrair identificador único ---
    if not product_id:
        if debug:
            print(f"  [DEBUG] Ignorado — sem id_externo: {url}")
        return None

    # --- Preço original: seletores SFCC ---
    preco_original = ""
    for sel in [".list .strike-through .value", ".price-list .value", ".old-price", "[data-list-price]"]:
        el = soup.select_one(sel)
        if el:
            preco_original = normalize_price_string(el.get("content", "") or el.get_text(" ", strip=True))
            if is_valid_current_price(preco_original):
                break

    if preco_original and preco_original == preco:
        preco_original = ""

    # --- Preço unitário: regex no texto da página ---
    preco_unitario = ""
    match = re.search(
        r"(\d+(?:[.,]\d+)?)\s*€\s*/\s*(kg|g|l|lt|ml|cl|un)",
        body_text,
        flags=re.IGNORECASE,
    )
    if match:
        preco_unitario = normalize_unit_price(f"{match.group(1)}€/{match.group(2)}")

    # --- Categoria: URL path ---
    categoria = normalize_category(category_from_url(url))

    # --- Imagem: JSON-LD image → .primary-image img → og:image ---
    imagem = ""
    if product_json:
        img = product_json.get("image", [])
        if isinstance(img, list) and img:
            imagem = img[0]
        elif isinstance(img, str):
            imagem = img
    if not imagem:
        for sel in [".primary-image img", ".product-image-container img", "picture img"]:
            el = soup.select_one(sel)
            if el:
                src = el.get("src") or el.get("data-src", "")
                if src and "placeholder" not in src.lower():
                    imagem = src if src.startswith("http") else BASE_URL + src
                    break
    if not imagem:
        el = soup.select_one("meta[property='og:image']")
        if el:
            imagem = el.get("content", "")

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
        loja="Auchan",
        data_recolha=run_timestamp,
        ean=ean,
    )


# ---------------------------------------------------------------------------
# Extração de um produto (AJAX → fallback HTML)
# ---------------------------------------------------------------------------

def extract_product(
    session: requests.Session,
    url: str,
    run_timestamp: str,
    debug: bool = False,
) -> dict | None:
    """Extrai os dados de um produto tentando primeiro o AJAX e depois o HTML.

    Estratégia de dois níveis:
    1. Extrair o PID da URL e chamar o endpoint AJAX SFCC → parse JSON.
    2. Se o AJAX falhar ou devolver HTML, fazer GET à página e parsear o HTML.

    Args:
        session: Sessão HTTP reutilizável.
        url: URL absoluto da página de produto.
        run_timestamp: Timestamp ISO da execução.
        debug: Se ``True``, ativa logs detalhados.

    Returns:
        Registo de produto no schema do projeto, ou ``None`` se falhar.
    """
    pid = extract_pid_from_url(url)

    # Tentativa 1: endpoint AJAX SFCC (mais eficiente — JSON direto)
    if pid:
        ajax_data = fetch_product_ajax(session, pid)
        if ajax_data:
            record = parse_product_ajax(ajax_data, url, run_timestamp)
            if record:
                return record
            if debug:
                print(f"  [DEBUG] AJAX devolveu dados mas parse falhou: {url}")

    # Tentativa 2: HTML da página de produto (SSR)
    if debug:
        print(f"  [DEBUG] A usar fallback HTML para: {url}")
    html   = fetch_text(session, url)
    return parse_product_html(html, url, run_timestamp, debug=debug)


# ---------------------------------------------------------------------------
# Scraping de produtos
# ---------------------------------------------------------------------------

def scrape_products(
    session: requests.Session,
    *,
    categoria: str | None = None,
    subcategoria: str | None = None,
    limite: int | None = None,
    run_timestamp: str,
    debug: bool = False,
) -> list[dict]:
    """Descobre URLs via sitemap, filtra e extrai cada produto.

    Args:
        session: Sessão HTTP reutilizável.
        categoria: Filtro de categoria (ex: ``"alimentacao"``). ``None`` sem filtro.
        subcategoria: Filtro de subcategoria (ex: ``"arroz"``). ``None`` sem filtro.
        limite: Número máximo de produtos a recolher. ``None`` para sem limite.
        run_timestamp: Timestamp ISO da execução.
        debug: Se ``True``, ativa logs detalhados para produtos problemáticos.

    Returns:
        Lista de registos de produtos recolhidos com sucesso.
    """
    product_urls = discover_product_sources(session, categoria=categoria, subcategoria=subcategoria)

    if limite:
        random.shuffle(product_urls)
        product_urls = product_urls[:limite]
        print(f"[INFO] Limite aplicado: {limite} produto(s).")

    results:   list[dict] = []
    ignorados: int        = 0
    erros:     int        = 0
    total:     int        = len(product_urls)

    print(f"\n[INFO] A recolher {total} produto(s)...")
    print("=" * 60)

    for i, url in enumerate(product_urls, start=1):
        print(f"  [{i:>{len(str(total))}}/{total}] {url}", flush=True)
        try:
            product = extract_product(session, url, run_timestamp=run_timestamp, debug=debug)
            if product:
                results.append(product)
            else:
                ignorados += 1
                if debug:
                    print("         → ignorado (sem nome/preço válido)")
            time.sleep(random.uniform(0.8, 1.5))
        except requests.RequestException as exc:
            erros += 1
            print(f"         → [ERRO] Falha no pedido: {exc}")
        except Exception as exc:
            erros += 1
            print(f"         → [ERRO] Falha no parsing: {exc}")

    print(f"\n  Total recolhido: {len(results)} | Ignorados: {ignorados} | Erros: {erros}")
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    """Ponto de entrada CLI: parseia argumentos e executa o scraping standalone."""
    parser = argparse.ArgumentParser(
        description="Scraper de preços do Auchan Portugal",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  python auchan_scraper.py --categoria alimentacao
  python auchan_scraper.py --categoria alimentacao --subcategoria arroz --limite 50
  python auchan_scraper.py --categoria bebidas-e-garrafeira --limite 100 --debug
  python auchan_scraper.py --categoria alimentacao --output-dir /dados/auchan

Categorias disponíveis no auchan.pt:
  alimentacao (10248 produtos), bebidas-e-garrafeira (4018), produtos-frescos (3486),
  biologicos-e-alternativas (1405), limpeza-e-cuidados-do-lar (1993),
  saude-e-bem-estar (7366), animais (3482), o-mundo-do-bebe (1454)
        """,
    )
    parser.add_argument(
        "--categoria", type=str, default="alimentacao",
        help="Categoria principal no path do produto (default: %(default)s)",
    )
    parser.add_argument(
        "--subcategoria", type=str,
        help="Subcategoria opcional (ex: arroz, massas, conservas)",
    )
    parser.add_argument(
        "--limite", type=int,
        help="Limitar número de produtos a recolher (útil para testes)",
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
        help="Logs de debug detalhados para produtos sem nome/preço e páginas com erro",
    )
    args = parser.parse_args()

    run_timestamp  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    file_timestamp = run_timestamp.replace(":", "").replace("-", "").replace(" ", "_")

    session  = make_session()
    products = scrape_products(
        session,
        categoria=args.categoria,
        subcategoria=args.subcategoria,
        limite=args.limite,
        run_timestamp=run_timestamp,
        debug=args.debug,
    )

    if not products:
        print("\nNenhum produto recolhido. Verifica a ligação ou os filtros.")
        return

    output_dir = args.output_dir
    if args.formato in ("csv", "ambos"):
        save_to_csv(products, f"auchan_{file_timestamp}.csv", output_dir)
    if args.formato in ("json", "ambos"):
        save_to_json(products, f"auchan_{file_timestamp}.json", output_dir)

    print(f"\n{'=' * 60}")
    print(f"  Resumo da recolha ({run_timestamp})")
    print(f"{'=' * 60}")
    filtros = f"categoria={args.categoria!r}"
    if args.subcategoria:
        filtros += f", subcategoria={args.subcategoria!r}"
    print(f"  Filtros            : {filtros}")
    print(f"  Produtos guardados : {len(products)}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
