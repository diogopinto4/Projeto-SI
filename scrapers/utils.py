"""
Utilitários partilhados pelos scrapers do projeto.

Este módulo é a única dependência cruzada entre os scrapers — ambos importam
daqui as funções de normalização, a fábrica de sessão HTTP e as funções de I/O.
Não deve importar código específico de nenhum scraper.

Importar com::

    from utils import clean_text, normalize_price_string, make_session, ...
"""

from __future__ import annotations

import csv
import json
import random
import re
import time
from pathlib import Path
from xml.etree import ElementTree as ET

import requests

# ---------------------------------------------------------------------------
# Normalização de texto
# ---------------------------------------------------------------------------

def clean_text(value: str | None) -> str:
    """Normaliza espaços e elimina espaços nas extremidades.

    Colapsa múltiplos espaços/tabs/newlines num único espaço.
    Devolve string vazia para ``None`` ou string vazia.

    Args:
        value: Texto a normalizar, ou ``None``.

    Returns:
        Texto limpo sem espaços desnecessários.
    """
    if not value:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


# ---------------------------------------------------------------------------
# Normalização de preços
# ---------------------------------------------------------------------------

def normalize_price_string(value: str | None) -> str:
    """Converte texto de preço para decimal com ponto como separador.

    Exemplos de entrada suportados::

        "1,69 €"        → "1.69"
        "1.69€"         → "1.69"
        "1.234,56 €"    → "1234.56"   (separador de milhares europeu)
        "PVPR 3,15€"    → "3.15"

    Estratégia:
        1. Remover símbolo de moeda e espaços não-separáveis.
        2. Se coexistirem ``.`` e ``,``, o ``.`` é separador de milhares — remover.
        3. Substituir ``,`` restante por ``.`` (separador decimal PT).
        4. Extrair o primeiro número decimal válido.

    Args:
        value: Texto com preço em formato livre.

    Returns:
        String decimal com ponto (ex: ``"1.69"``), ou ``""`` se não for possível
        extrair um número válido.
    """
    if not value:
        return ""

    text = str(value).replace("\xa0", " ")
    text = text.replace("€", "").replace("EUR", "").replace("eur", "").strip()

    # Separador de milhares europeu: "1.234,56" → "1234,56"
    if "," in text and "." in text:
        text = text.replace(".", "")

    text = text.replace(",", ".")
    match = re.search(r"\d+(?:\.\d+)?", text)
    return match.group(0) if match else ""


#: Limite máximo plausível de preço unitário de produto de supermercado.
#: Defesa contra extração errada que apanha quantidades ("125g" → 125€) ou
#: anos ("Mundial 2026" → 2026€) do nome do produto em vez do preço real.
#: Mesmo produtos premium (presunto, vinhos topo, packs grandes) ficam abaixo
#: deste limite com folga.
MAX_PLAUSIBLE_PRICE = 1000.0


def is_valid_current_price(price: str | None) -> bool:
    """Verifica se uma string de preço representa um valor positivo plausível.

    Aceita apenas valores estritamente entre 0 e :data:`MAX_PLAUSIBLE_PRICE`.
    O limite superior é defesa contra extração errada — ver constante para detalhes.

    Args:
        price: String de preço normalizada (ex: ``"1.69"``).

    Returns:
        ``True`` se o preço for um número entre 0 e MAX_PLAUSIBLE_PRICE.
    """
    if not price:
        return False
    try:
        value = float(price)
        return 0 < value < MAX_PLAUSIBLE_PRICE
    except ValueError:
        return False


# ---------------------------------------------------------------------------
# Normalização de unidades e preços unitários
# ---------------------------------------------------------------------------

def normalize_unit(value: str) -> str:
    """Normaliza a designação de uma unidade de medida para forma canónica.

    Mapeia variantes e abreviaturas para as formas normalizadas usadas
    no schema do projeto.

    Args:
        value: Designação de unidade em formato livre (ex: ``"lt"``, ``"Kg"``).

    Returns:
        Forma canónica da unidade (ex: ``"l"``, ``"kg"``), ou o valor
        original em minúsculas se não houver mapeamento.
    """
    unit = value.lower().strip()
    unit_map: dict[str, str] = {
        # Volume
        "l":      "l",
        "lt":     "l",
        "ltr":    "l",
        "lts":    "l",
        "litro":  "l",
        "litros": "l",
        "cl":     "cl",
        "dl":     "dl",
        "ml":     "ml",
        # Peso
        "kg":     "kg",
        "g":      "g",
        "gr":     "g",
        "grs":    "g",
        # Unidade / embalagem
        "un":     "un",
        "uni":    "un",
        "unid":   "un",
        "unids":  "un",
        "caps":   "un",
        "cáps":   "un",
        "cap":    "un",
        "emb":    "un",
        "dose":   "un",
        "doses":  "un",
        "pack":   "un",
    }
    return unit_map.get(unit, unit)


def normalize_unit_price(value: str | None) -> str:
    """Normaliza preço unitário para formato ``{valor}€/{unidade}``.

    Exemplos::

        "22,05€/kg"  → "22.05€/kg"
        "2.14€/lt"   → "2.14€/l"
        "1,69 € / l" → "1.69€/l"

    Extrai valor numérico e unidade via regex, normalizando ambos.
    Devolve ``""`` se o padrão não for encontrado.

    Args:
        value: Texto de preço unitário em formato livre.

    Returns:
        Preço unitário normalizado, ou ``""`` se o padrão não for reconhecido.
    """
    text = clean_text(value)
    if not text:
        return ""

    match = re.search(
        r"(\d+(?:[.,]\d+)?)\s*€\s*/\s*([A-Za-z]+)",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        return ""

    num = normalize_price_string(match.group(1))
    unit = normalize_unit(match.group(2))
    if num and unit:
        return f"{num}€/{unit}"
    return ""


# ---------------------------------------------------------------------------
# Normalização de categorias
# ---------------------------------------------------------------------------

def slugify_category_part(value: str) -> str:
    """Converte um segmento de categoria para slug ASCII em minúsculas.

    Remove acentos e caracteres especiais, substitui espaços e pontuação
    por hífenes, e colapsa hífenes consecutivos.

    Exemplos::

        "Mercearia & Bebidas" → "mercearia-e-bebidas"
        "Arroz/Massas"        → "arroz-massas"
        "Café & Chá"          → "cafe-e-cha"

    Args:
        value: Texto do segmento de categoria.

    Returns:
        Slug ASCII em minúsculas.
    """
    value = clean_text(value).lower()

    # Substituições de caracteres acentuados e especiais
    replacements: dict[str, str] = {
        "á": "a", "à": "a", "ã": "a", "â": "a", "ä": "a",
        "é": "e", "è": "e", "ê": "e", "ë": "e",
        "í": "i", "ì": "i", "î": "i", "ï": "i",
        "ó": "o", "ò": "o", "õ": "o", "ô": "o", "ö": "o",
        "ú": "u", "ù": "u", "û": "u", "ü": "u",
        "ç": "c", "ñ": "n",
    }
    for char, replacement in replacements.items():
        value = value.replace(char, replacement)

    value = value.replace("&", " e ")
    value = re.sub(r"[^a-z0-9]+", "-", value)
    value = re.sub(r"-+", "-", value).strip("-")
    return value


def normalize_category(value: str | None) -> str:
    """Normaliza um caminho de categoria completo para slug hierárquico.

    Divide por ``/``, aplica ``slugify_category_part`` a cada segmento
    e rejunta com ``/``. Segmentos vazios são ignorados.

    Exemplos::

        "Mercearia/Arroz & Massas" → "mercearia/arroz-e-massas"
        "Bebidas/Sumos"            → "bebidas/sumos"

    Args:
        value: Caminho de categoria em formato livre, com segmentos separados
               por ``/``.

    Returns:
        Caminho normalizado em minúsculas com slugs por segmento.
    """
    text = clean_text(value)
    if not text:
        return ""
    parts = [p.strip() for p in text.split("/") if p.strip()]
    return "/".join(slugify_category_part(part) for part in parts)


# ---------------------------------------------------------------------------
# Sessão HTTP
# ---------------------------------------------------------------------------

def make_session() -> requests.Session:
    """Cria uma sessão HTTP com headers de browser para evitar bloqueios.

    Os headers simulam um browser Chrome em Windows, que é o user-agent
    mais comum e menos propenso a ser bloqueado por WAFs dos supermercados.

    Returns:
        Sessão ``requests.Session`` configurada e pronta a usar.
    """
    session = requests.Session()
    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0.0.0 Safari/537.36"
        ),
        "Accept":          "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "pt-PT,pt;q=0.9,en;q=0.8",
        "Accept-Encoding": "gzip, deflate",
    })
    return session


# ---------------------------------------------------------------------------
# Fetch HTTP com retry e backoff exponencial
# ---------------------------------------------------------------------------

#: Tentativas por pedido HTTP antes de desistir (partilhado entre scrapers).
HTTP_MAX_RETRIES = 3

#: Timeout em segundos por pedido HTTP (partilhado entre scrapers).
HTTP_TIMEOUT = 30


def fetch_text(
    session: requests.Session,
    url: str,
    *,
    max_retries: int = HTTP_MAX_RETRIES,
    timeout: int = HTTP_TIMEOUT,
) -> str:
    """Faz GET a uma URL com retry e backoff exponencial.

    Tenta até ``max_retries`` vezes. Entre tentativas, aguarda um tempo
    crescente com jitter aleatório para evitar sobrecarregar o servidor.
    Na última tentativa, propaga a exceção diretamente.

    Args:
        session: Sessão HTTP reutilizável (com headers de browser).
        url: URL a obter.
        max_retries: Número máximo de tentativas. Default: :data:`HTTP_MAX_RETRIES`.
        timeout: Timeout em segundos por tentativa. Default: :data:`HTTP_TIMEOUT`.

    Returns:
        Conteúdo da resposta como string (HTML, XML ou JSON).

    Raises:
        requests.RequestException: Se todas as tentativas falharem.
    """
    for attempt in range(1, max_retries + 1):
        try:
            response = session.get(url, timeout=timeout)
            response.raise_for_status()
            return response.text
        except requests.RequestException as exc:
            if attempt < max_retries:
                wait = 1.5 * attempt + random.uniform(0.2, 0.8)
                print(f"    [RETRY {attempt}/{max_retries}] {exc} — a aguardar {wait:.1f}s")
                time.sleep(wait)
            else:
                raise


# ---------------------------------------------------------------------------
# Parsing de sitemaps XML
# ---------------------------------------------------------------------------

def parse_xml_urls(xml_text: str) -> list[str]:
    """Extrai todas as URLs de um documento XML de sitemap.

    Suporta tanto sitemap index (``<sitemapindex>``) como sitemap de URLs
    (``<urlset>``), pois ambos usam ``<loc>`` para os endereços.

    Args:
        xml_text: Conteúdo XML do sitemap como string.

    Returns:
        Lista de URLs encontradas, sem espaços nas extremidades.
    """
    root = ET.fromstring(xml_text)
    ns   = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
    return [loc.text.strip() for loc in root.findall(".//sm:loc", ns) if loc.text]


# ---------------------------------------------------------------------------
# Schema de produto
# ---------------------------------------------------------------------------

#: Ordem canónica dos campos no schema do projeto.
#: Usada como cabeçalho CSV e para garantir consistência entre scrapers.
OUTPUT_FIELDNAMES: list[str] = [
    "id_externo",
    "nome",
    "marca",
    "categoria",
    "preco",
    "preco_original",
    "preco_unitario",
    "url",
    "imagem",
    "loja",
    "data_recolha",
    "ean",
]


def build_product_record(
    *,
    id_externo: str,
    nome: str,
    marca: str,
    categoria: str,
    preco: str,
    preco_original: str,
    preco_unitario: str,
    url: str,
    imagem: str,
    loja: str,
    data_recolha: str,
    ean: str = "",
) -> dict:
    """Constrói um registo de produto no schema comum do projeto.

    Todos os parâmetros são keyword-only para evitar erros de ordem de
    argumentos, dado o elevado número de campos.

    Args:
        id_externo: Identificador do produto no site de origem (SKU, id numérico, etc.).
        nome: Nome completo do produto tal como aparece no site.
        marca: Marca do produto (pode ser ``""`` se não disponível).
        categoria: Caminho de categoria normalizado (ex: ``"mercearia/arroz"``).
        preco: Preço atual como string decimal (ex: ``"1.69"``).
        preco_original: Preço antes de desconto (``""`` se não em promoção).
        preco_unitario: Preço por unidade de medida (ex: ``"1.69€/kg"``; ``""`` se ausente).
        url: URL absoluto da página do produto.
        imagem: URL da imagem principal do produto (``""`` se não disponível).
        loja: Nome da cadeia de supermercado (ex: ``"Continente"``, ``"Pingo Doce"``).
        data_recolha: Timestamp da execução no formato ``"YYYY-MM-DD HH:MM:SS"``.
        ean: Código de barras EAN/GTIN (``""`` se não disponível).

    Returns:
        Dicionário com todos os campos no schema do projeto, na ordem de
        ``OUTPUT_FIELDNAMES``.
    """
    return {
        "id_externo":     id_externo,
        "nome":           nome,
        "marca":          marca,
        "categoria":      categoria,
        "preco":          preco,
        "preco_original": preco_original,
        "preco_unitario": preco_unitario,
        "url":            url,
        "imagem":         imagem,
        "loja":           loja,
        "data_recolha":   data_recolha,
        "ean":            ean,
    }


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def save_to_csv(products: list[dict], filename: str, output_dir: Path) -> Path | None:
    """Guarda uma lista de produtos em ficheiro CSV.

    Usa ``OUTPUT_FIELDNAMES`` como cabeçalho e como filtro de colunas
    (campos extra nos dicionários são ignorados silenciosamente).

    Args:
        products: Lista de registos de produto (dicionários no schema do projeto).
        filename: Nome do ficheiro de saída (ex: ``"continente_20260101_120000.csv"``).
        output_dir: Diretório onde guardar o ficheiro. Criado se não existir.

    Returns:
        ``Path`` do ficheiro criado, ou ``None`` se ``products`` estiver vazio.
    """
    if not products:
        print("Nenhum produto para guardar (CSV).")
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    filepath = output_dir / filename

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(products)

    print(f"CSV guardado: {filepath} ({len(products)} produtos)")
    return filepath


def save_to_json(
    records: list[dict],
    filename: str,
    output_dir: Path,
    *,
    label: str = "produtos",
) -> Path | None:
    """Guarda uma lista de registos genéricos em ficheiro JSON com indentação.

    Usada por:
    - Scrapers de produtos (Auchan, Continente, Pingo Doce) — ``label="produtos"``.
    - Scraper de lojas físicas — ``label="lojas"``.

    Args:
        records: Lista de registos (dicionários) a serializar.
        filename: Nome do ficheiro de saída (ex: ``"continente_20260101_120000.json"``).
        output_dir: Diretório onde guardar o ficheiro. Criado se não existir.
        label: Designação usada nas mensagens de log (default: ``"produtos"``).

    Returns:
        ``Path`` do ficheiro criado, ou ``None`` se ``records`` estiver vazio.
    """
    if not records:
        print(f"Nenhum/a {label} para guardar (JSON).")
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    filepath = output_dir / filename

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"JSON guardado: {filepath} ({len(records)} {label})")
    return filepath
