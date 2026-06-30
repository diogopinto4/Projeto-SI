"""
Scraper para recolher produtos do Pingo Doce via sitemap XML.

Estratégia de recolha:
    1. Ler o sitemap index para obter os URLs dos sitemaps de produto.
    2. Ler cada sitemap de produto para agregar todos os URLs individuais.
    3. Filtrar URLs por categoria e/ou subcategoria (inferidas do path).
    4. Visitar cada página de produto e extrair campos com JSON-LD como
       fonte primária e múltiplos fallbacks (meta tags, HTML, regex).
    5. Normalizar e validar todos os campos.
    6. Guardar no schema comum do projeto (JSON e/ou CSV).

Fonte primária de dados:
    Os dados estruturados estão disponíveis em blocos ``<script type="application/ld+json">``
    com ``@type: Product``. Os campos mais importantes (nome, marca, preço, SKU, imagem)
    vêm daqui. Os fallbacks só são usados quando o JSON-LD está incompleto.

Filtro por categoria:
    Baseia-se no path da URL de cada produto, que segue o padrão
    ``/home/produtos/{categoria}/{subcategoria}/{nome}.html``.
    Passa ``--categoria mercearia`` para restringir a essa secção do site.

Uso:
    python pingo_doce_scraper.py --categoria mercearia --formato json
    python pingo_doce_scraper.py --categoria mercearia --subcategoria arroz --limite 50
    python pingo_doce_scraper.py --categoria bebidas --limite 100 --debug
    python pingo_doce_scraper.py --categoria mercearia --output-dir /dados/pingo_doce
"""

from __future__ import annotations

import argparse
import hashlib
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
    build_product_record,
    clean_text,
    fetch_text,
    is_valid_current_price,
    make_session,
    normalize_category,
    normalize_price_string,
    normalize_unit,
    normalize_unit_price,
    parse_xml_urls,
    save_to_csv,
    save_to_json,
)

# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------

SITEMAP_INDEX_URL = "https://www.pingodoce.pt/home/sitemap_index.xml"
"""URL do sitemap index que lista todos os sitemaps do site."""

OUTPUT_DIR = Path(__file__).parent / "output"


def _stable_url_id(prefix: str, url: str) -> str:
    """Gera um identificador determinístico quando a página não expõe SKU."""
    if not url:
        return ""
    digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}-{digest}"


# ---------------------------------------------------------------------------
# Descoberta de URLs via Sitemap XML
# ---------------------------------------------------------------------------

def get_product_sitemaps(session: requests.Session) -> list[str]:
    """Lê o sitemap index e devolve apenas os URLs de sitemaps de produto.

    Os sitemaps de produto são identificados pela presença de ``"product"``
    no seu URL — convenção usada pelo Pingo Doce.

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
        time.sleep(random.uniform(0.5, 1.2))

    return urls


def _url_norm(url: str) -> str:
    """Normaliza um URL do PD para comparação de categoria.

    A PD inclui um zero-width space (``%E2%80%8B`` URL-encoded, ``U+200B``
    em Unicode) em alguns slugs de categoria no sitemap, ex:
    ``/home/produtos/leite-e-bebidas-vegetais%E2%80%8B/leite/...``. Sem esta
    normalização, o filtro nunca encontra correspondência porque o utilizador
    põe ``"leite-e-bebidas-vegetais"`` no config (sem o caracter invisível).

    Strip ZWS em ambas as formas (URL-encoded e Unicode) para robustez.
    """
    return url.lower().replace("%e2%80%8b", "").replace("​", "")


def filter_urls(
    urls: list[str],
    categoria: str | None = None,
    subcategoria: str | None = None,
) -> list[str]:
    """Filtra URLs de produto por categoria e/ou subcategoria no path.

    A estrutura esperada dos paths é:
    ``/home/produtos/{categoria}/{subcategoria}/{slug}.html``

    Args:
        urls: Lista completa de URLs de produto.
        categoria: Segmento de categoria a filtrar (ex: ``"mercearia"``).
                   ``None`` para não filtrar por categoria.
        subcategoria: Segmento de subcategoria a filtrar (ex: ``"arroz"``).
                      ``None`` para não filtrar por subcategoria.

    Returns:
        Lista filtrada de URLs. Imprime aviso se o resultado for vazio.
    """
    filtered = urls

    if categoria:
        cat      = categoria.strip("/").lower()
        filtered = [u for u in filtered if f"/home/produtos/{cat}/" in _url_norm(u)]

    if subcategoria:
        sub      = subcategoria.strip("/").lower()
        filtered = [u for u in filtered if f"/{sub}/" in _url_norm(u)]

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
        categoria: Filtro de categoria (ex: ``"mercearia"``).
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
# Extração de dados de cada página de produto
# ---------------------------------------------------------------------------

def extract_json_ld(soup: BeautifulSoup) -> list[dict]:
    """Extrai todos os blocos JSON-LD presentes na página.

    Suporta blocos em que o JSON raiz é um objeto ou uma lista de objetos.

    Args:
        soup: BeautifulSoup da página de produto.

    Returns:
        Lista de dicionários JSON-LD. Blocos inválidos são ignorados.
    """
    blocks: list[dict] = []
    for script in soup.find_all("script", type="application/ld+json"):
        raw = script.get_text(strip=True)
        if not raw:
            continue
        try:
            data = json.loads(raw)
            if isinstance(data, list):
                blocks.extend(x for x in data if isinstance(x, dict))
            elif isinstance(data, dict):
                blocks.append(data)
        except json.JSONDecodeError:
            continue
    return blocks


def find_product_jsonld(blocks: list[dict]) -> dict | None:
    """Encontra o primeiro bloco JSON-LD com ``@type: Product``.

    Args:
        blocks: Lista de blocos JSON-LD extraídos da página.

    Returns:
        Dicionário do bloco Product, ou ``None`` se não encontrado.
    """
    for block in blocks:
        t = block.get("@type")
        if t == "Product" or (isinstance(t, list) and "Product" in t):
            return block
    return None


def category_from_url(url: str) -> str:
    """Infere a categoria a partir do path da URL do produto.

    Extrai os segmentos entre ``/produtos/`` e o slug final.
    Exemplo::

        /home/produtos/mercearia/temperos/sal/sal-fino-1805.html
        → "mercearia/temperos/sal"

    Args:
        url: URL absoluto da página de produto.

    Returns:
        String de categoria com segmentos separados por ``/``,
        ou ``""`` se a estrutura esperada não for encontrada.
    """
    path  = urlparse(url).path.strip("/")
    parts = path.split("/")
    # Estrutura: home / produtos / {categoria...} / {slug}.html
    if len(parts) >= 5 and parts[0] == "home" and parts[1] == "produtos":
        return "/".join(parts[2:-1])
    return ""


def parse_breadcrumb_category(soup: BeautifulSoup) -> str:
    """Extrai a categoria a partir do breadcrumb de navegação da página.

    Tenta vários seletores CSS para encontrar o breadcrumb e filtra
    os itens genéricos ("home", "produtos"). Desduplicado preservando ordem.

    Args:
        soup: BeautifulSoup da página de produto.

    Returns:
        Segmentos do breadcrumb unidos por ``/``, ou ``""`` se não encontrado.
    """
    items: list[str] = []
    for sel in [
        'nav[aria-label*="breadcrumb"] a',
        ".breadcrumb a",
        '[class*="breadcrumb"] a',
    ]:
        for el in soup.select(sel):
            text = clean_text(el.get_text())
            if text and text.lower() not in {"home", "produtos", "/"}:
                items.append(text)

    # Desduplicar preservando ordem de aparecimento
    seen:   set[str]  = set()
    unique: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            unique.append(item)

    return "/".join(unique)


def extract_price_from_jsonld(product_json: dict) -> str:
    """Extrai o preço atual a partir das offers do JSON-LD.

    Tenta as chaves ``price`` (preço unitário) e ``lowPrice`` (preço
    mínimo numa gama de preços), pela ordem indicada.
    ``highPrice`` é intencionalmente excluído — representa o preço
    máximo de uma gama, não o preço a pagar.

    Args:
        product_json: Bloco JSON-LD com ``@type: Product``.

    Returns:
        Preço como string decimal (ex: ``"1.29"``), ou ``""`` se não encontrado.
    """
    offers  = product_json.get("offers", {})
    sources = offers if isinstance(offers, list) else [offers]
    for offer in sources:
        if not isinstance(offer, dict):
            continue
        for key in ("price", "lowPrice"):
            price = normalize_price_string(offer.get(key))
            if is_valid_current_price(price):
                return price
    return ""


def extract_old_price_from_jsonld(product_json: dict) -> str:
    """Extrai o preço original (antes de desconto) das offers do JSON-LD.

    Tenta várias chaves não-standard usadas por diferentes implementações
    do schema.org para preço antes de desconto.

    Args:
        product_json: Bloco JSON-LD com ``@type: Product``.

    Returns:
        Preço original como string decimal, ou ``""`` se não encontrado.
    """
    offers  = product_json.get("offers", {})
    sources = offers if isinstance(offers, list) else [offers]
    keys    = ("priceBeforeDiscount", "price_before_discount", "listPrice", "oldPrice", "regularPrice")
    for offer in sources:
        if not isinstance(offer, dict):
            continue
        for key in keys:
            price = normalize_price_string(offer.get(key))
            if is_valid_current_price(price):
                return price
    return ""


def extract_price_fallbacks(soup: BeautifulSoup, body_text: str) -> str:
    """Extrai o preço atual quando o JSON-LD não tem esse campo.

    Percorre três camadas de fallback pela ordem: meta tags OpenGraph/itemprop,
    elementos HTML com classes/atributos indicativos de preço, e por fim
    padrões de texto via regex.

    Args:
        soup: BeautifulSoup da página de produto.
        body_text: Texto completo da página (``soup.get_text()``).

    Returns:
        Preço como string decimal, ou ``""`` se nenhum fallback tiver êxito.
    """
    # Camada 1: meta tags
    for attrs, attr_name in [
        ({"property": "product:price:amount"}, "content"),
        ({"name":     "product:price:amount"}, "content"),
        ({"itemprop": "price"},                "content"),
    ]:
        el = soup.find("meta", attrs=attrs)
        if el and el.get(attr_name):
            price = normalize_price_string(el.get(attr_name))
            if is_valid_current_price(price):
                return price

    # Camada 2: elementos HTML com seletores indicativos de preço.
    #
    # IMPORTANTE: o atributo ``content`` (usado em itemprop="price") é estruturado
    # e confiável. Já o texto visível precisa de regex que exija o número
    # **adjacente** ao símbolo de moeda — caso contrário um seletor amplo como
    # [class*="price"] pode apanhar quantidades ("180g") em elementos que misturem
    # quantidade e preço (ex: "180g por 1.79€" — sem o regex, o primeiro número
    # capturado seria 180, não 1.79).
    for sel in [
        '[itemprop="price"]',
        '[class*="sales"] [class*="value"]',
        '[class*="current-price"]',
        '[class*="price"]',
    ]:
        for el in soup.select(sel):
            content_val = el.get("content", "")
            if content_val:
                price = normalize_price_string(content_val)
                if is_valid_current_price(price):
                    return price
            text_val = el.get_text(" ", strip=True)
            if text_val:
                match = re.search(r"(\d+(?:[.,]\d+)?)\s*€", text_val)
                if match:
                    price = normalize_price_string(match.group(1))
                    if is_valid_current_price(price):
                        return price

    # Camada 3: padrões de texto via regex
    for pattern in [
        r"(?i)(?:preço|pvp)\s*[:\-]?\s*(\d+(?:[.,]\d+)?)\s*€",
        r"(\d+(?:[.,]\d+)?)\s*€\s*(?:/\s*un)?",
    ]:
        for match in re.finditer(pattern, body_text):
            price = normalize_price_string(match.group(1))
            if is_valid_current_price(price):
                return price

    return ""


def extract_old_price_fallback(soup: BeautifulSoup, body_text: str) -> str:
    """Extrai o preço original usando HTML e padrões de texto como fallback.

    Args:
        soup: BeautifulSoup da página de produto.
        body_text: Texto completo da página (``soup.get_text()``).

    Returns:
        Preço original como string decimal, ou ``""`` se não encontrado.
    """
    for sel in [
        '[class*="old-price"]',
        '[class*="list-price"]',
        '[class*="strike"]',
        '[class*="was-price"]',
    ]:
        for el in soup.select(sel):
            price = normalize_price_string(el.get_text(" ", strip=True))
            if is_valid_current_price(price):
                return price

    match = re.search(
        r"(?i)(?:antes|preço anterior)\s*[:\-]?\s*(\d+(?:[.,]\d+)?)\s*€",
        body_text,
    )
    if match:
        price = normalize_price_string(match.group(1))
        if is_valid_current_price(price):
            return price

    return ""


def extract_unit_price_from_text(body_text: str) -> str:
    """Extrai o preço unitário (€/kg, €/l, etc.) do texto da página.

    Procura o padrão ``{valor}€/{unidade}`` em qualquer parte do texto.

    Args:
        body_text: Texto completo da página (``soup.get_text()``).

    Returns:
        Preço unitário normalizado (ex: ``"2.50€/kg"``), ou ``""`` se não encontrado.
    """
    match = re.search(
        r"(\d+(?:[.,]\d+)?)\s*€\s*/\s*(kg|g|l|lt|ml|cl|un)",
        body_text,
        flags=re.IGNORECASE,
    )
    if match:
        preco = normalize_price_string(match.group(1))
        unit  = normalize_unit(match.group(2))
        return f"{preco}€/{unit}" if preco else ""
    return ""


def extract_product_image(product_json: dict | None, soup: BeautifulSoup) -> str:
    """Extrai o URL da imagem principal do produto.

    Tenta primeiro o campo ``image`` do JSON-LD (suporta lista e string),
    e recorre à meta tag ``og:image`` como fallback.

    Args:
        product_json: Bloco JSON-LD com ``@type: Product``, ou ``None``.
        soup: BeautifulSoup da página de produto.

    Returns:
        URL da imagem, ou ``""`` se não encontrado.
    """
    if product_json:
        img = product_json.get("image")
        if isinstance(img, list) and img:
            return img[0]
        if isinstance(img, str) and img:
            return img

    el = soup.select_one("meta[property='og:image']")
    return el["content"] if el and el.get("content") else ""


def extract_product_id(product_json: dict | None, url: str) -> str:
    """Extrai o identificador único do produto.

    Tenta primeiro os campos ``sku`` e ``productID`` do JSON-LD.
    Como fallback, extrai o sufixo numérico do slug da URL
    (ex: ``sal-fino-1805.html`` → ``"1805"``).

    Args:
        product_json: Bloco JSON-LD com ``@type: Product``, ou ``None``.
        url: URL absoluto da página de produto.

    Returns:
        ID do produto como string. Se o site não expuser SKU, devolve um ID
        determinístico derivado da URL.
    """
    if product_json:
        pid = str(product_json.get("sku") or product_json.get("productID") or "")
        if pid:
            return clean_text(pid)

    match = re.search(r"-(\d+)\.html$", url)
    return match.group(1) if match else _stable_url_id("pingo-url", url)


def parse_product_html(
    html: str,
    url: str,
    run_timestamp: str,
    debug: bool = False,
) -> dict | None:
    """Constrói o registo completo de um produto a partir do HTML da sua página.

    Hierarquia de fontes por campo:

    - **Nome**: JSON-LD → ``<h1>``
    - **Marca**: JSON-LD (``brand.name`` ou string)
    - **Preço atual**: JSON-LD → meta tags → HTML → regex
    - **Preço original**: JSON-LD (chaves não-standard) → HTML → regex
    - **Preço unitário**: regex no texto da página
    - **Categoria**: breadcrumb → URL path
    - **Imagem**: JSON-LD → og:image
    - **ID**: JSON-LD (sku/productID) → sufixo numérico da URL

    Args:
        html: Conteúdo HTML da página de produto.
        url: URL absoluto da página (usado como fallback e identificador).
        run_timestamp: Timestamp ISO da execução.
        debug: Se ``True``, imprime detalhes de parsing para páginas com problemas.

    Returns:
        Registo do produto no schema do projeto, ou ``None`` se o produto
        não tiver nome ou preço válido.
    """
    soup         = BeautifulSoup(html, "html.parser")
    json_ld      = extract_json_ld(soup)
    product_json = find_product_jsonld(json_ld)
    body_text    = soup.get_text(" ", strip=True)

    # --- Nome ---
    nome = clean_text(product_json.get("name")) if product_json else ""
    if not nome:
        h1   = soup.select_one("h1")
        nome = clean_text(h1.get_text()) if h1 else ""

    # --- Marca ---
    marca = ""
    if product_json:
        brand_data = product_json.get("brand")
        if isinstance(brand_data, dict):
            marca = clean_text(brand_data.get("name", ""))
        elif isinstance(brand_data, str):
            marca = clean_text(brand_data)

    # --- Categoria ---
    categoria = parse_breadcrumb_category(soup) or category_from_url(url)
    categoria = normalize_category(categoria)

    # --- Preços ---
    preco = extract_price_from_jsonld(product_json) if product_json else ""
    if not is_valid_current_price(preco):
        preco = extract_price_fallbacks(soup, body_text)

    preco_original = extract_old_price_from_jsonld(product_json) if product_json else ""
    if not is_valid_current_price(preco_original):
        preco_original = extract_old_price_fallback(soup, body_text)

    preco_unitario = normalize_unit_price(extract_unit_price_from_text(body_text))

    # --- Outros campos ---
    product_id = extract_product_id(product_json, url)
    imagem     = extract_product_image(product_json, soup)

    # --- Validação obrigatória ---
    if not nome:
        if debug:
            print(f"  [DEBUG] Ignorado — sem nome: {url}")
        return None

    if not is_valid_current_price(preco):
        if debug:
            print(f"  [DEBUG] Ignorado — preço inválido: {url}")
            if product_json:
                print(json.dumps(product_json, ensure_ascii=False, indent=2)[:1500])
            else:
                print(body_text[:800])
        return None

    # Sanity check final: rejeita preços que são integers >=20€ e cujo valor
    # aparece literalmente como token no nome do produto. Este é o padrão
    # repetido do bug #30 (180g→180€, 200g→200€, "60 Minutos"→60€). Mesmo com
    # o regex mais estrito na Camada 2, páginas com markup invulgar podem
    # voltar a apanhar a quantidade como preço — esta verificação é a última
    # defesa antes de gravar.
    try:
        preco_float = float(preco)
        if preco_float >= 20 and preco_float == int(preco_float):
            preco_int_str = str(int(preco_float))
            # Mesma heurística da cleanup SQL: número como token isolado ou
            # imediatamente seguido por sufixo de quantidade/percentagem.
            padrao = (
                rf"\b{re.escape(preco_int_str)}\s*(g|gr|ml|kg|l|cl|un|x|min|minutos|%)\b"
                rf"|\b{re.escape(preco_int_str)}\b"
            )
            if re.search(padrao, nome, flags=re.IGNORECASE):
                if debug:
                    print(f"  [DEBUG] Rejeitado — preço {preco}€ coincide com número no nome '{nome}': {url}")
                return None
    except (ValueError, TypeError):
        pass

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
        loja="Pingo Doce",
        data_recolha=run_timestamp,
    )


# ---------------------------------------------------------------------------
# Scraping de produtos
# ---------------------------------------------------------------------------

def scrape_products(
    session: requests.Session,
    *,
    categoria: str | None = None,
    subcategoria: str | None = None,
    categorias: list[str] | None = None,
    limite: int | None = None,
    run_timestamp: str,
    debug: bool = False,
) -> list[dict]:
    """Descobre URLs via sitemap, filtra e visita cada página de produto.

    Args:
        session: Sessão HTTP reutilizável.
        categoria: Filtro de categoria única (ex: ``"mercearia"``). ``None``
            sem filtro. **Use ``categorias`` quando precisar de várias.**
        subcategoria: Filtro de subcategoria (ex: ``"arroz"``). ``None`` sem filtro.
            Só faz sentido com ``categoria`` singular.
        categorias: Lista de categorias a recolher (ex: ``["mercearia",
            "congelados", "iogurtes-e-sobremesas"]``). Quando fornecida, a
            função descobre URLs do sitemap **uma vez** e filtra para a união
            das categorias indicadas — mais eficiente que múltiplas chamadas.
            Tem precedência sobre ``categoria``.
        limite: Número máximo de produtos a recolher. ``None`` para sem limite
                (útil em produção; reduz para testes).
        run_timestamp: Timestamp ISO da execução.
        debug: Se ``True``, ativa logs detalhados para produtos problemáticos.

    Returns:
        Lista de registos de produtos recolhidos com sucesso.
    """
    if categorias:
        # União: descobre o sitemap uma vez, depois filtra para várias categorias
        print(f"[INFO] Modo multi-categoria: {categorias}")
        sitemap_urls = get_product_sitemaps(session)
        all_urls = get_product_urls(session, sitemap_urls)
        urls_set: set[str] = set()
        for cat in categorias:
            cat_norm = cat.strip("/").lower()
            # Usa ``_url_norm`` para tolerar zero-width spaces que a PD inclui
            # em alguns slugs do sitemap (ex: leite-e-bebidas-vegetais).
            urls_set.update(
                u for u in all_urls
                if f"/home/produtos/{cat_norm}/" in _url_norm(u)
            )
        product_urls = sorted(urls_set)
        print(f"       {len(product_urls)} URLs após união de {len(categorias)} categorias.")
    else:
        product_urls = discover_product_sources(
            session, categoria=categoria, subcategoria=subcategoria,
        )

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
            html    = fetch_text(session, url)
            product = parse_product_html(html, url, run_timestamp=run_timestamp, debug=debug)
            if product:
                results.append(product)
            else:
                ignorados += 1
                if debug:
                    print("         → ignorado (sem nome/preço válido)")
            time.sleep(random.uniform(0.8, 1.6))
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
        description="Scraper do Pingo Doce via sitemap XML",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  python pingo_doce_scraper.py --categoria mercearia
  python pingo_doce_scraper.py --categoria mercearia --subcategoria arroz --limite 50
  python pingo_doce_scraper.py --categoria bebidas --limite 100 --debug
  python pingo_doce_scraper.py --categoria mercearia --output-dir /dados/pingo_doce
        """,
    )
    parser.add_argument(
        "--categoria", type=str, default="mercearia",
        help="Categoria principal no path do produto (default: %(default)s)",
    )
    parser.add_argument(
        "--subcategoria", type=str,
        help="Subcategoria opcional (ex: arroz, conservas, cafe-cha)",
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

    # Um único timestamp para toda a execução — usado nos dados e no nome do ficheiro
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
        save_to_csv(products, f"pingo_doce_{file_timestamp}.csv", output_dir)
    if args.formato in ("json", "ambos"):
        save_to_json(products, f"pingo_doce_{file_timestamp}.json", output_dir)

    print(f"\n{'=' * 60}")
    print(f"  Resumo da recolha ({run_timestamp})")
    print(f"{'=' * 60}")
    filtros = f"categoria={args.categoria!r}"
    if args.subcategoria:
        filtros += f", subcategoria={args.subcategoria!r}"
    print(f"  Filtros                 : {filtros}")
    print(f"  Produtos guardados      : {len(products)}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
