"""
Pipeline de ingestão de dados dos scrapers para PostgreSQL.

Transforma os registos JSON produzidos pelos scrapers no schema normalizado
de 5 tabelas da base de dados, seguindo esta sequência para cada registo:

    1. Desduplicar registos por chave estável dentro do mesmo batch.
    2. Garantir a existência da loja (``lojas``).
    3. Construir a chave mestre do produto (por EAN ou hash semântico) e
       garantir a existência do produto canónico (``produtos_mestre``).
    4. Garantir a existência do produto específico desta loja (``produtos_loja``).
    5. Inserir preço no histórico (``historico_precos``, idempotente por
       constraint UNIQUE em id_produto_loja + data_recolha + agente_origem).
    6. Atualizar o preço atual (``precos_atuais``, só sobrescreve se o registo
       for mais recente ou se algum campo de preço tiver mudado).

Cada produto é uma transação independente: falhas individuais não afetam
os restantes registos do batch.

Uso:
    python scripts/ingest.py --input scrapers/output/*.json
    python scripts/ingest.py --input scrapers/output/continente_*.json --dry-run
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Imports — stdlib → third-party → local
# ---------------------------------------------------------------------------

import argparse
import glob
import json
import re
import unicodedata
from collections import OrderedDict
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import NamedTuple
from zoneinfo import ZoneInfo
import sys

import psycopg2

sys.path.insert(0, str(Path(__file__).parent))
from db_config import DB_CONFIG

# ---------------------------------------------------------------------------
# Constantes e expressões regulares
# ---------------------------------------------------------------------------

#: Unidades de medida reconhecidas pelo parser de quantidades.
UNIDADES_BASE = (
    r"kg|g|mg|l|lt|ml|cl|dl|un|uni|unid|unids|caps|capsulas|cápsulas|saq|saquetas|doses"
)

#: Padrão para packs com multiplicador: "3x250g", "6 x 1l".
PACK_RE = re.compile(
    rf"(\d+)\s*[xX]\s*(\d+(?:[\.,]\d+)?)\s*({UNIDADES_BASE})\b",
    re.IGNORECASE,
)

#: Padrão para quantidade simples no nome: "500g", "1.5l".
QUANTIDADE_RE = re.compile(
    rf"(\d+(?:[\.,]\d+)?)\s*({UNIDADES_BASE})\b",
    re.IGNORECASE,
)

#: Padrão para preço unitário: "2.50€/kg", "1,69 € / l".
PRECO_UNITARIO_RE = re.compile(
    r"(\d+(?:[\.,]\d+)?)\s*€\s*/\s*([A-Za-zÀ-ÿ]+)",
    re.IGNORECASE,
)

TZ_LOCAL = ZoneInfo("Europe/Lisbon")
"""Fuso horário local usado para interpretar timestamps sem offset."""


# ---------------------------------------------------------------------------
# NamedTuple para campos de preço
# ---------------------------------------------------------------------------

class CamposPreco(NamedTuple):
    """Campos de preço extraídos de um registo de scraper.

    Substitui o tuplo anónimo de 7 elementos que era devolvido por
    ``_extrair_campos_preco()``, tornando o acesso por nome em vez de
    por posição e eliminando o risco de unpacking errado.
    """

    preco_atual:            Decimal
    preco_original:         Decimal | None
    preco_unitario_valor:   Decimal | None
    preco_unitario_unidade: str | None
    em_promocao:            bool
    data_recolha:           datetime
    agente_origem:          str


# ---------------------------------------------------------------------------
# Utilitários de texto e decimal
# ---------------------------------------------------------------------------

def get_connection() -> psycopg2.extensions.connection:
    """Abre e devolve uma nova ligação à base de dados PostgreSQL."""
    return psycopg2.connect(**DB_CONFIG)


def limpar_texto(valor: str | None) -> str:
    """Normaliza espaços e elimina espaços nas extremidades.

    Args:
        valor: Texto a limpar, ou ``None``.

    Returns:
        Texto com espaços colapsados, ou ``""`` para ``None`` ou vazio.
    """
    if valor is None:
        return ""
    return re.sub(r"\s+", " ", str(valor)).strip()


def normalizar_marca(valor: str | None) -> str:
    """Normaliza a capitalização do nome da marca para apresentação consistente.

    Diferentes scrapers (e diferentes JSON-LD) devolvem marca em casing diverso:
    "FERBAR", "Ferbar", "ferbar" — o mesmo conceito comercial mas com chaves
    diferentes na BD. Esta função converte para Title Case quando o input é
    todo maiúsculas e tem mais de 3 caracteres, preservando siglas curtas
    como "EA", "PD", "AB".

    Args:
        valor: Marca em formato livre.

    Returns:
        Marca normalizada (Title Case se aplicável), ou ``""`` se vazio/None.
    """
    texto = limpar_texto(valor)
    if not texto:
        return ""
    # Preservar siglas curtas (EA, PD, AB, etc.). Aplicar Title Case quando
    # o texto é todo maiúsculas E suficientemente longo para não ser uma sigla.
    if texto.isupper() and len(texto) > 3:
        return texto.title()
    # Casing misto ou minúsculas → Title Case canónico para uniformidade.
    return texto.title()


def normalizar_decimal(valor: str | None) -> Decimal | None:
    """Converte texto de preço ou quantidade para ``Decimal``.

    Suporta separadores europeus de milhares e decimais::

        "1,69"      → Decimal("1.69")
        "1.69"      → Decimal("1.69")
        "1.234,56"  → Decimal("1234.56")   ← separador de milhares

    Args:
        valor: Texto numérico em formato livre.

    Returns:
        ``Decimal`` positivo ou zero, ou ``None`` se a conversão falhar.
    """
    texto = limpar_texto(valor)
    if not texto:
        return None

    texto = texto.replace("€", "").strip()

    # Separador de milhares europeu: "1.234,56" → "1234,56"
    if "," in texto and "." in texto:
        texto = texto.replace(".", "")

    texto = texto.replace(",", ".")
    match = re.search(r"\d+(?:\.\d+)?", texto)
    if not match:
        return None

    try:
        return Decimal(match.group(0))
    except InvalidOperation:
        return None


def normalizar_texto_chave(valor: str | None) -> str:
    """Converte texto para slug ASCII normalizado, usado na construção da chave mestre.

    Remove acentos via decomposição Unicode, substitui caracteres não
    alfanuméricos por hífenes e colapsa hífenes consecutivos.

    Exemplos::

        "Arroz Agulha"  → "arroz-agulha"
        "Café & Chá"    → "cafe-cha"     (& colapsa como qualquer char especial; não é expandido para "e" como em slugify_category_part)
        "50% Desconto"  → "50-pct-desconto"

    Args:
        valor: Texto a converter.

    Returns:
        Slug ASCII em minúsculas.
    """
    valor = limpar_texto(valor).lower()
    valor = unicodedata.normalize("NFKD", valor)
    valor = "".join(c for c in valor if not unicodedata.combining(c))
    valor = valor.replace("%", " pct ")
    valor = re.sub(r"[^a-z0-9]+", "-", valor)
    valor = re.sub(r"-+", "-", valor)
    return valor.strip("-")


def decimal_para_str_sem_zeros(valor: Decimal | None) -> str:
    """Converte ``Decimal`` para string sem zeros decimais desnecessários.

    Exemplos::

        Decimal("500.000") → "500"
        Decimal("1.500")   → "1.5"
        Decimal("1.250")   → "1.25"

    Args:
        valor: Valor decimal a converter.

    Returns:
        String sem zeros à direita, ou ``""`` para ``None``.
    """
    if valor is None:
        return ""
    texto = format(valor.normalize(), "f")
    if "." in texto:
        texto = texto.rstrip("0").rstrip(".")
    return texto


def decimal_abs(valor: Decimal | None) -> Decimal | None:
    """Devolve o valor absoluto de um ``Decimal``, ou ``None`` para ``None``."""
    return valor.copy_abs() if valor is not None else None


# ---------------------------------------------------------------------------
# Normalização de unidades
# ---------------------------------------------------------------------------

def normalizar_unidade(unidade: str) -> str:
    """Normaliza a designação de uma unidade de medida para forma canónica.

    Args:
        unidade: Designação em formato livre (ex: ``"lt"``, ``"capsulas"``).

    Returns:
        Forma canónica (ex: ``"l"``, ``"un"``), ou o valor original em minúsculas
        se não houver mapeamento.
    """
    unidade = limpar_texto(unidade).lower()
    mapa: dict[str, str] = {
        "lt":        "l",
        "uni":       "un",
        "unid":      "un",
        "unids":     "un",
        "caps":      "un",
        "capsulas":  "un",
        "cápsulas":  "un",
        "saq":       "un",
        "saquetas":  "un",
        "doses":     "un",
    }
    return mapa.get(unidade, unidade)


# ---------------------------------------------------------------------------
# Extração e validação de quantidades
# ---------------------------------------------------------------------------

def aproximar_quantidade_plausivel(
    quantidade: Decimal,
    unidade: str,
) -> tuple[Decimal | None, str | None]:
    """Arredonda uma quantidade para o tamanho comercial mais próximo plausível.

    Usa uma lista de tamanhos comuns por unidade. Aceita o arredondamento se
    o erro relativo for ≤ 8% (evita valores como 499g quando é claramente 500g).
    Para g e ml sem correspondência exata, aceita arredondamento inteiro com
    tolerância de 3%. Para un, aceita qualquer inteiro com tolerância de 8%.

    Args:
        quantidade: Valor em bruto (pode ter decimais por erros de divisão).
        unidade: Unidade canónica (``"g"``, ``"ml"``, ``"un"``).

    Returns:
        Tuplo (quantidade_arredondada, unidade), ou (None, None) se nenhum
        arredondamento for considerado plausível.
    """
    tamanhos_comuns: dict[str, list[Decimal]] = {
        "g": [
            Decimal("10"), Decimal("20"), Decimal("25"), Decimal("30"), Decimal("40"),
            Decimal("42"), Decimal("50"), Decimal("75"), Decimal("80"), Decimal("85"),
            Decimal("90"), Decimal("100"), Decimal("125"), Decimal("150"), Decimal("160"),
            Decimal("180"), Decimal("200"), Decimal("225"), Decimal("250"), Decimal("300"),
            Decimal("330"), Decimal("350"), Decimal("375"), Decimal("400"), Decimal("450"),
            Decimal("500"), Decimal("550"), Decimal("600"), Decimal("750"), Decimal("800"),
            Decimal("900"), Decimal("1000"), Decimal("1250"), Decimal("1500"), Decimal("1750"),
            Decimal("2000"), Decimal("2500"), Decimal("3000"), Decimal("4000"), Decimal("5000"),
        ],
        "ml": [
            Decimal("100"), Decimal("125"), Decimal("150"), Decimal("200"), Decimal("250"),
            Decimal("330"), Decimal("500"), Decimal("750"), Decimal("1000"), Decimal("1500"),
            Decimal("2000"), Decimal("3000"), Decimal("5000"),
        ],
        "un": [
            Decimal("1"), Decimal("2"), Decimal("3"), Decimal("4"), Decimal("5"),
            Decimal("6"), Decimal("8"), Decimal("10"), Decimal("12"), Decimal("16"),
            Decimal("18"), Decimal("20"), Decimal("24"), Decimal("30"), Decimal("32"),
            Decimal("36"), Decimal("40"), Decimal("48"), Decimal("60"),
        ],
    }

    candidatos = tamanhos_comuns.get(unidade, [])
    if candidatos:
        melhor = min(candidatos, key=lambda x: decimal_abs(x - quantidade))
        erro   = decimal_abs(melhor - quantidade) / quantidade if quantidade != 0 else Decimal("999")
        if erro <= Decimal("0.08"):
            return melhor, unidade

    if unidade in {"g", "ml"}:
        arredondada = quantidade.quantize(Decimal("1"))
        erro = decimal_abs(arredondada - quantidade) / quantidade if quantidade != 0 else Decimal("999")
        if erro <= Decimal("0.03"):
            return arredondada, unidade

    if unidade == "un":
        arredondada = quantidade.quantize(Decimal("1"))
        if arredondada >= 1:
            erro = decimal_abs(arredondada - quantidade) / quantidade if quantidade != 0 else Decimal("999")
            if erro <= Decimal("0.08"):
                return arredondada, unidade

    return None, None


def converter_para_unidade_canonica(
    quantidade: Decimal | None,
    unidade: str,
) -> tuple[Decimal | None, str]:
    """Converte quantidade para a unidade base canónica do projeto.

    Converte para g (peso) ou ml (volume) de forma a que produtos com
    unidades diferentes sejam comparáveis::

        (1, "kg")  → (1000, "g")
        (500, "mg") → (0.5, "g")
        (1, "l")   → (1000, "ml")
        (33, "cl") → (330, "ml")
        (2, "dl")  → (200, "ml")

    Unidades já canónicas (g, ml, un) são devolvidas sem alteração.

    Args:
        quantidade: Valor numérico a converter.
        unidade: Unidade original (antes de normalizar).

    Returns:
        Tuplo (quantidade_convertida, unidade_canónica).
    """
    if quantidade is None or not unidade:
        return quantidade, unidade

    unidade = normalizar_unidade(unidade)

    conversoes: dict[str, tuple[Decimal, str]] = {
        "kg": (Decimal("1000"), "g"),
        "mg": (Decimal("0.001"), "g"),
        "l":  (Decimal("1000"), "ml"),
        "cl": (Decimal("10"),   "ml"),
        "dl": (Decimal("100"),  "ml"),
    }
    if unidade in conversoes:
        fator, unidade_dest = conversoes[unidade]
        return quantidade * fator, unidade_dest

    return quantidade, unidade


# ---------------------------------------------------------------------------
# Extração de informação de quantidade do nome do produto
# ---------------------------------------------------------------------------

def extrair_pack_quantidade(nome: str) -> dict | None:
    """Extrai informação de pack com multiplicador do nome do produto.

    Reconhece padrões do tipo ``"3x250g"``, ``"6 x 1l"``, ``"12 X 33cl"``.

    Args:
        nome: Nome completo do produto.

    Returns:
        Dicionário com campos ``tipo``, ``multiplicador``,
        ``quantidade_por_unidade``, ``unidade_por_unidade``,
        ``quantidade_total``, ``unidade_total`` e ``fonte``,
        ou ``None`` se o padrão não for encontrado.
    """
    match = PACK_RE.search(limpar_texto(nome))
    if not match:
        return None

    multiplicador = normalizar_decimal(match.group(1))
    quantidade    = normalizar_decimal(match.group(2))
    unidade       = normalizar_unidade(match.group(3))

    if multiplicador is None or quantidade is None or not unidade:
        return None

    qt_total, un_total = converter_para_unidade_canonica(multiplicador * quantidade, unidade)
    qpu,      upu      = converter_para_unidade_canonica(quantidade, unidade)

    return {
        "tipo":                  "pack",
        "multiplicador":         int(multiplicador),
        "quantidade_por_unidade": qpu,
        "unidade_por_unidade":    upu,
        "quantidade_total":       qt_total,
        "unidade_total":          un_total,
        "fonte":                  "nome_pack",
    }


def extrair_quantidade_simples(nome: str) -> dict | None:
    """Extrai quantidade simples (sem multiplicador) do nome do produto.

    Usa a última ocorrência do padrão no nome, pois frequentemente
    o nome tem um número no início (ex: "3 Queijos") mas a quantidade
    relevante está no final (ex: "3 Queijos Barreado 200g").

    Args:
        nome: Nome completo do produto.

    Returns:
        Dicionário com campos de quantidade (ver ``extrair_pack_quantidade``),
        ou ``None`` se não for encontrada nenhuma quantidade.
    """
    matches = list(QUANTIDADE_RE.finditer(limpar_texto(nome)))
    if not matches:
        return None

    match     = matches[-1]   # última ocorrência
    quantidade = normalizar_decimal(match.group(1))
    unidade    = normalizar_unidade(match.group(2))

    if quantidade is None or not unidade:
        return None

    quantidade, unidade = converter_para_unidade_canonica(quantidade, unidade)

    return {
        "tipo":                   "simples",
        "multiplicador":          None,
        "quantidade_por_unidade": quantidade,
        "unidade_por_unidade":    unidade,
        "quantidade_total":       quantidade,
        "unidade_total":          unidade,
        "fonte":                  "nome_simples",
    }


def extrair_preco_unitario(valor: str | None) -> tuple[Decimal | None, str | None]:
    """Extrai valor e unidade de um preço unitário normalizado.

    Reconhece padrões como ``"2.50€/kg"``, ``"1,69 € / l"``.

    Args:
        valor: String de preço unitário (ex: ``"2.50€/kg"``).

    Returns:
        Tuplo (valor_decimal, unidade_canónica), ou (None, None) se o padrão
        não for reconhecido.
    """
    texto = limpar_texto(valor)
    if not texto:
        return None, None

    match = PRECO_UNITARIO_RE.search(texto.replace(",", "."))
    if not match:
        return None, None

    preco  = normalizar_decimal(match.group(1))
    unidade = normalizar_unidade(match.group(2))
    return preco, unidade


def inferir_quantidade_por_preco_unitario(item: dict) -> dict | None:
    """Infere a quantidade de um produto dividindo preço total por preço unitário.

    Usado como último recurso quando o nome não contém quantidade explícita.
    A quantidade inferida é validada contra tamanhos comerciais plausíveis
    (tolerância de 8% para g/ml, 8% para un).

    Exemplo: preço=1.69€, preço_unitário=1.69€/kg → quantidade=1000g

    Args:
        item: Registo de produto com campos ``preco`` e ``preco_unitario``.

    Returns:
        Dicionário de quantidade (ver ``extrair_pack_quantidade``), ou ``None``
        se a inferência não produzir um resultado plausível.
    """
    preco                    = normalizar_decimal(item.get("preco"))
    preco_unitario, unidade  = extrair_preco_unitario(item.get("preco_unitario"))

    if preco is None or preco_unitario is None or preco_unitario <= 0 or not unidade:
        return None

    try:
        quantidade = preco / preco_unitario
    except InvalidOperation:
        return None

    if quantidade <= 0:
        return None

    quantidade, unidade = converter_para_unidade_canonica(quantidade, unidade)
    q_aprox, u_aprox   = aproximar_quantidade_plausivel(quantidade, unidade)

    if q_aprox is None or u_aprox is None:
        return None

    return {
        "tipo":                   "inferida_preco_unitario",
        "multiplicador":          None,
        "quantidade_por_unidade": q_aprox,
        "unidade_por_unidade":    u_aprox,
        "quantidade_total":       q_aprox,
        "unidade_total":          u_aprox,
        "fonte":                  "preco_unitario",
    }


def extrair_info_quantidade(item: dict) -> dict | None:
    """Ponto de entrada unificado para extração de quantidade de um produto.

    Tenta as três estratégias pela ordem de fiabilidade:

    1. Pack com multiplicador no nome (ex: ``"3x250g"``).
    2. Quantidade simples no nome (ex: ``"500g"``).
    3. Inferência por divisão preço / preço_unitário (fallback).

    Args:
        item: Registo de produto com campos ``nome``, ``preco`` e ``preco_unitario``.

    Returns:
        Dicionário com informação de quantidade, ou ``None`` se nenhuma
        estratégia tiver êxito.
    """
    nome = item.get("nome")
    return (
        extrair_pack_quantidade(nome)
        or extrair_quantidade_simples(nome)
        or inferir_quantidade_por_preco_unitario(item)
    )


# ---------------------------------------------------------------------------
# Parsing de timestamps
# ---------------------------------------------------------------------------

def parse_data_recolha(valor: str | None) -> datetime:
    """Converte texto de timestamp para ``datetime`` UTC.

    Suporta os formatos produzidos pelos scrapers::

        "2026-04-08 05:22:01"   → datetime UTC
        "2026-04-08T05:22:01"   → datetime UTC
        "2026-04-08"            → datetime UTC (meia-noite em Lisboa)
        "2026-04-08T05:22:01Z"  → datetime UTC (já com offset)

    Timestamps sem informação de fuso são tratados como hora de Lisboa
    e convertidos para UTC.

    Args:
        valor: String de timestamp.

    Returns:
        ``datetime`` com tzinfo UTC.

    Raises:
        ValueError: Se o valor for vazio ou não corresponder a nenhum formato.
    """
    texto = limpar_texto(valor)
    if not texto:
        raise ValueError("data_recolha em falta.")

    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(texto, fmt).replace(tzinfo=TZ_LOCAL)
            return dt.astimezone(timezone.utc)
        except ValueError:
            continue

    try:
        dt = datetime.fromisoformat(texto.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=TZ_LOCAL)
        return dt.astimezone(timezone.utc)
    except ValueError:
        raise ValueError(f"Formato de data_recolha não reconhecido: {texto!r}")


# ---------------------------------------------------------------------------
# Construção de chave mestre e nome padronizado
# ---------------------------------------------------------------------------

def limpar_nome_base_para_chave(nome: str) -> str:
    """Remove padrões de quantidade, pack e promoção do nome do produto.

    Usado antes de construir a chave mestre semântica para que o mesmo
    produto com embalagens diferentes (ex: 500g vs 1000g) tenha bases
    de chave similares — a quantidade entra separadamente no descriptor.

    Args:
        nome: Nome completo do produto.

    Returns:
        Nome sem padrões de quantidade, em minúsculas.
    """
    texto = limpar_texto(nome).lower()
    texto = PACK_RE.sub(" ", texto)
    texto = QUANTIDADE_RE.sub(" ", texto)
    texto = re.sub(r"\bemb\.?\b", " ", texto)
    texto = re.sub(r"\bpack\b",   " ", texto)
    texto = re.sub(r"\bpromo(?:cao)?\b", " ", texto)
    return re.sub(r"\s+", " ", texto).strip()


def construir_descriptor_quantidade(info_quantidade: dict | None) -> str:
    """Constrói a componente de quantidade para a chave mestre.

    Exemplos::

        pack 3x250g total 750g  → "q:pack-3x250g-total-750g"
        simples 500g            → "q:500g"
        sem quantidade          → "q:desconhecida"

    Args:
        info_quantidade: Dicionário devolvido por ``extrair_info_quantidade``,
                         ou ``None``.

    Returns:
        String de descriptor de quantidade para inclusão na chave mestre.
    """
    if not info_quantidade:
        return "q:desconhecida"

    total       = decimal_para_str_sem_zeros(info_quantidade["quantidade_total"])
    un_total    = info_quantidade["unidade_total"]

    if info_quantidade["tipo"] == "pack":
        mult = info_quantidade["multiplicador"]
        qpu  = decimal_para_str_sem_zeros(info_quantidade["quantidade_por_unidade"])
        upu  = info_quantidade["unidade_por_unidade"]
        return f"q:pack-{mult}x{qpu}{upu}-total-{total}{un_total}"

    return f"q:{total}{un_total}"


def construir_chave_mestre(item: dict) -> str:
    """Constrói a chave mestre determinística do produto.

    A chave mestre é o identificador canónico que permite reconhecer o mesmo
    produto independentemente da loja onde é vendido.

    Estratégia:
        - **Se EAN disponível**: ``"ean:{ean}"`` — chave inequívoca.
        - **Caso contrário**: hash semântico com 4 componentes::

            "n:{nome_base}|m:{marca}|c:{categoria}|q:{descriptor_quantidade}"

          onde ``nome_base`` é o nome sem quantidades e em slug ASCII, e
          ``descriptor_quantidade`` codifica a embalagem (pack ou simples).

    A chave é completamente determinística: o mesmo produto produz sempre
    a mesma chave, independentemente da ordem de ingestão ou da loja.

    Args:
        item: Registo de produto com campos ``ean``, ``nome``, ``marca``,
              ``categoria``, ``preco`` e ``preco_unitario``.

    Returns:
        String de chave mestre (máx. ~512 chars na prática).
    """
    ean = limpar_texto(item.get("ean"))
    if ean:
        return f"ean:{ean}"

    nome_original = limpar_texto(item.get("nome"))
    nome_base     = normalizar_texto_chave(limpar_nome_base_para_chave(nome_original))
    marca         = normalizar_texto_chave(item.get("marca"))
    categoria     = normalizar_texto_chave((item.get("categoria") or "").split("/")[0])

    info_qt       = extrair_info_quantidade(item)
    descriptor    = construir_descriptor_quantidade(info_qt)

    return "|".join([
        f"n:{nome_base or normalizar_texto_chave(nome_original)}",
        f"m:{marca}",
        f"c:{categoria}",
        descriptor,
    ])


def construir_nome_padronizado(item: dict) -> str:
    """Constrói o nome canónico do produto para apresentação ao utilizador.

    Limpa o nome (remove padrões de quantidade/pack/promo), capitaliza
    cada palavra, adiciona a marca se não estiver já no nome, e adiciona
    o sufixo de quantidade se não estiver presente.

    Exemplos::

        "arroz agulha cigala 1000g" (Cigala) → "Arroz Agulha Cigala 1000g"
        "atum ao natural 85g" (Ramirez)      → "Atum Ao Natural Ramirez 85g"

    Args:
        item: Registo de produto com campos ``nome``, ``marca``, ``preco``
              e ``preco_unitario``.

    Returns:
        Nome padronizado como string.
    """
    nome_original = limpar_texto(item.get("nome"))
    marca         = normalizar_marca(item.get("marca"))
    info_qt       = extrair_info_quantidade(item)

    nome_base = limpar_texto(limpar_nome_base_para_chave(nome_original)) or nome_original
    nome_base = " ".join(p.capitalize() if not p.isupper() else p for p in nome_base.split())

    if marca and marca.lower() not in nome_base.lower():
        nome_base = f"{nome_base} {marca}".strip()

    if info_qt and info_qt["quantidade_total"] is not None and info_qt["unidade_total"]:
        if info_qt["tipo"] == "pack" and info_qt["multiplicador"] and info_qt["quantidade_por_unidade"]:
            sufixo = (
                f"{info_qt['multiplicador']}x"
                f"{decimal_para_str_sem_zeros(info_qt['quantidade_por_unidade'])}"
                f"{info_qt['unidade_por_unidade']}"
            )
        else:
            sufixo = (
                f"{decimal_para_str_sem_zeros(info_qt['quantidade_total'])}"
                f"{info_qt['unidade_total']}"
            )
        if sufixo.lower() not in nome_base.lower():
            nome_base = f"{nome_base} {sufixo}".strip()

    return nome_base


# ---------------------------------------------------------------------------
# Carregamento e deduplicação de ficheiros
# ---------------------------------------------------------------------------

def carregar_ficheiros(padroes: list[str]) -> tuple[list[dict], list[str]]:
    """Carrega e agrega registos de um ou mais ficheiros JSON.

    Suporta padrões glob e caminhos diretos. Ordena os ficheiros pelo nome
    para processamento determinístico.

    Args:
        padroes: Lista de padrões glob ou caminhos de ficheiro.

    Returns:
        Tuplo (lista_de_registos, lista_de_caminhos_carregados).

    Raises:
        FileNotFoundError: Se nenhum ficheiro corresponder aos padrões.
        ValueError: Se um ficheiro não contiver uma lista JSON.
    """
    caminhos: list[str] = []
    for padrao in padroes:
        encontrados = glob.glob(padrao)
        if encontrados:
            caminhos.extend(encontrados)
        elif Path(padrao).is_file():
            caminhos.append(padrao)

    caminhos = sorted(set(caminhos))
    if not caminhos:
        raise FileNotFoundError("Nenhum ficheiro encontrado para ingestão.")

    registos: list[dict] = []
    for caminho in caminhos:
        with open(caminho, "r", encoding="utf-8") as f:
            dados = json.load(f)
            if not isinstance(dados, list):
                raise ValueError(f"{caminho} não contém uma lista JSON.")
            registos.extend(dados)

    return registos, caminhos


def _chave_deduplicacao(item: dict) -> tuple[str, str]:
    """Constrói uma chave estável para deduplicação dentro do batch.

    O SKU/id externo é a fonte preferencial. Quando o scraper não consegue
    extrair esse campo, usa-se a URL do produto; se também não existir URL,
    cai-se para uma assinatura semântica simples.
    """
    loja = limpar_texto(item.get("loja"))
    sku = limpar_texto(item.get("id_externo"))
    if sku:
        return loja, f"sku:{sku}"

    url = limpar_texto(item.get("url"))
    if url:
        return loja, f"url:{normalizar_texto_chave(url)}"

    nome = limpar_texto(item.get("nome"))
    categoria = limpar_texto(item.get("categoria"))
    preco = limpar_texto(item.get("preco"))
    assinatura = normalizar_texto_chave("|".join([nome, categoria, preco]))
    return loja, f"sem-sku:{assinatura}"


def deduplicar_registos(registos: list[dict]) -> list[dict]:
    """Remove duplicados dentro do batch, mantendo um registo por produto/loja.

    A chave de deduplicação **não inclui** ``data_recolha`` porque CSV e JSON
    gerados no mesmo run têm timestamps ligeiramente diferentes, o que criaria
    duplicados que depois produziriam chaves_mestre distintas na BD. Quando
    ``id_externo`` está vazio, a função usa URL/nome/categoria/preço como
    fallback para não colapsar produtos diferentes da mesma loja.

    Em caso de duplicado, mantém o registo com a ``data_recolha`` mais recente
    (comparação lexicográfica dos timestamps ISO é correta).

    Args:
        registos: Lista bruta de registos carregados dos ficheiros.

    Returns:
        Lista deduplicada, na ordem de primeira aparição.
    """
    dedup: OrderedDict[tuple, dict] = OrderedDict()
    for item in registos:
        chave = _chave_deduplicacao(item)
        existente = dedup.get(chave)
        if existente is None or (
            limpar_texto(item.get("data_recolha", ""))
            >= limpar_texto(existente.get("data_recolha", ""))
        ):
            dedup[chave] = item
    return list(dedup.values())


# ---------------------------------------------------------------------------
# Operações na base de dados
# ---------------------------------------------------------------------------

def obter_ou_criar_loja(
    cur: psycopg2.extensions.cursor,
    insignia: str,
    formato_loja: str = "Online",
    localizacao: str  = "Nacional",
    canal: str        = "online",
) -> int:
    """Garante a existência da loja e devolve o seu ``id_loja``.

    Usa ``ON CONFLICT DO UPDATE`` para ser idempotente.

    Args:
        cur: Cursor de base de dados ativo.
        insignia: Nome da cadeia (ex: ``"Continente"``, ``"Pingo Doce"``).
        formato_loja: Formato da loja (ex: ``"Online"``, ``"Física"``).
        localizacao: Localização geográfica (ex: ``"Nacional"``).
        canal: Canal de venda (ex: ``"online"``).

    Returns:
        ``id_loja`` (inteiro).
    """
    cur.execute(
        """
        INSERT INTO lojas (insignia, formato_loja, localizacao, canal)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (insignia, formato_loja, localizacao, canal)
        DO UPDATE SET data_atualizacao = CURRENT_TIMESTAMP
        RETURNING id_loja;
        """,
        (insignia, formato_loja, localizacao, canal),
    )
    return cur.fetchone()[0]


def obter_ou_criar_produto_mestre(
    cur: psycopg2.extensions.cursor,
    item: dict,
) -> int:
    """Garante a existência do produto canónico e devolve o seu ``id_produto_mestre``.

    Em caso de conflito na ``chave_mestre`` (produto já existe), atualiza
    apenas os campos com valor ``NULL`` na BD — preserva dados existentes e
    enriquece com novos (ex: EAN descoberto numa ingestão posterior).

    Args:
        cur: Cursor de base de dados ativo.
        item: Registo de produto do scraper.

    Returns:
        ``id_produto_mestre`` (inteiro).
    """
    chave_mestre     = construir_chave_mestre(item)
    info_qt          = extrair_info_quantidade(item)
    nome_padronizado = construir_nome_padronizado(item)

    cur.execute(
        """
        INSERT INTO produtos_mestre (
            chave_mestre, ean, nome_padronizado, marca, categoria_geral,
            quantidade_valor, quantidade_unidade
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (chave_mestre) DO UPDATE SET
            ean              = COALESCE(produtos_mestre.ean,              EXCLUDED.ean),
            nome_padronizado = EXCLUDED.nome_padronizado,
            marca            = COALESCE(EXCLUDED.marca,            produtos_mestre.marca),
            categoria_geral  = COALESCE(EXCLUDED.categoria_geral,  produtos_mestre.categoria_geral),
            quantidade_valor = COALESCE(EXCLUDED.quantidade_valor, produtos_mestre.quantidade_valor),
            quantidade_unidade = COALESCE(EXCLUDED.quantidade_unidade, produtos_mestre.quantidade_unidade)
        RETURNING id_produto_mestre;
        """,
        (
            chave_mestre,
            limpar_texto(item.get("ean")) or None,
            nome_padronizado,
            normalizar_marca(item.get("marca")) or None,
            limpar_texto((item.get("categoria") or "").split("/")[0]) or None,
            info_qt["quantidade_total"]  if info_qt else None,
            info_qt["unidade_total"]     if info_qt else None,
        ),
    )
    return cur.fetchone()[0]


def obter_ou_criar_produto_loja(
    cur: psycopg2.extensions.cursor,
    id_produto_mestre: int,
    id_loja: int,
    item: dict,
) -> int:
    """Garante a existência do produto desta loja específica.

    Liga o ``produto_mestre`` à ``loja`` através de ``produtos_loja``.
    Em caso de conflito (mesmo SKU na mesma loja), atualiza todos os campos
    e regista a data da última observação.

    Args:
        cur: Cursor de base de dados ativo.
        id_produto_mestre: FK para ``produtos_mestre``.
        id_loja: FK para ``lojas``.
        item: Registo de produto do scraper.

    Returns:
        ``id_produto_loja`` (inteiro).
    """
    info_qt = extrair_info_quantidade(item)

    cur.execute(
        """
        INSERT INTO produtos_loja (
            id_produto_mestre, id_loja, sku_loja, nome_na_loja,
            categoria_loja, url_produto, url_imagem,
            quantidade_valor, quantidade_unidade,
            multiplicador_pack, unidade_base_pack, unidade_medida_pack
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (id_loja, sku_loja) DO UPDATE SET
            id_produto_mestre   = EXCLUDED.id_produto_mestre,
            nome_na_loja        = EXCLUDED.nome_na_loja,
            categoria_loja      = EXCLUDED.categoria_loja,
            url_produto         = EXCLUDED.url_produto,
            url_imagem          = EXCLUDED.url_imagem,
            quantidade_valor    = EXCLUDED.quantidade_valor,
            quantidade_unidade  = EXCLUDED.quantidade_unidade,
            multiplicador_pack  = EXCLUDED.multiplicador_pack,
            unidade_base_pack   = EXCLUDED.unidade_base_pack,
            unidade_medida_pack = EXCLUDED.unidade_medida_pack,
            data_ultima_observacao = CURRENT_TIMESTAMP
        RETURNING id_produto_loja;
        """,
        (
            id_produto_mestre,
            id_loja,
            limpar_texto(item.get("id_externo")),
            limpar_texto(item.get("nome")),
            limpar_texto(item.get("categoria"))  or None,
            limpar_texto(item.get("url"))        or None,
            limpar_texto(item.get("imagem"))     or None,
            info_qt["quantidade_total"]          if info_qt else None,
            info_qt["unidade_total"]             if info_qt else None,
            info_qt["multiplicador"]             if info_qt and info_qt["tipo"] == "pack" else None,
            info_qt["quantidade_por_unidade"]    if info_qt and info_qt["tipo"] == "pack" else None,
            info_qt["unidade_por_unidade"]       if info_qt and info_qt["tipo"] == "pack" else None,
        ),
    )
    return cur.fetchone()[0]


def _extrair_campos_preco(item: dict) -> CamposPreco:
    """Extrai e valida todos os campos de preço de um registo.

    Função interna partilhada por ``inserir_historico_preco`` e
    ``upsert_preco_atual`` para evitar duplicação de lógica.

    Args:
        item: Registo de produto do scraper.

    Returns:
        ``CamposPreco`` com todos os campos validados.

    Raises:
        ValueError: Se o preço atual for inválido (None ou ≤ 0).
    """
    preco_atual = normalizar_decimal(item.get("preco"))
    if preco_atual is None or preco_atual <= 0:
        raise ValueError(f"Preço atual inválido para SKU {item.get('id_externo')!r}")

    preco_original               = normalizar_decimal(item.get("preco_original"))
    preco_unitario_valor, pu_un  = extrair_preco_unitario(item.get("preco_unitario"))
    em_promocao                  = bool(preco_original and preco_original > preco_atual)
    data_recolha                 = parse_data_recolha(item.get("data_recolha"))
    agente_origem                = f"scraper_{normalizar_texto_chave(item.get('loja', ''))}"

    return CamposPreco(
        preco_atual=preco_atual,
        preco_original=preco_original,
        preco_unitario_valor=preco_unitario_valor,
        preco_unitario_unidade=pu_un,
        em_promocao=em_promocao,
        data_recolha=data_recolha,
        agente_origem=agente_origem,
    )


def inserir_historico_preco(
    cur: psycopg2.extensions.cursor,
    id_produto_loja: int,
    item: dict,
) -> None:
    """Insere um registo de preço no histórico (idempotente).

    O ``ON CONFLICT DO NOTHING`` garante que o mesmo registo (mesmo produto,
    mesma data, mesmo agente) não é inserido duas vezes, tornando a operação
    segura para re-execuções.

    Args:
        cur: Cursor de base de dados ativo.
        id_produto_loja: FK para ``produtos_loja``.
        item: Registo de produto do scraper.
    """
    c = _extrair_campos_preco(item)
    cur.execute(
        """
        INSERT INTO historico_precos (
            id_produto_loja, preco_atual, preco_original,
            preco_unitario_valor, preco_unitario_unidade,
            em_promocao, data_recolha, agente_origem
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT DO NOTHING;
        """,
        (
            id_produto_loja,
            c.preco_atual,
            c.preco_original,
            c.preco_unitario_valor,
            c.preco_unitario_unidade,
            c.em_promocao,
            c.data_recolha,
            c.agente_origem,
        ),
    )


def upsert_preco_atual(
    cur: psycopg2.extensions.cursor,
    id_produto_loja: int,
    item: dict,
) -> None:
    """Atualiza o preço atual do produto, mas só se necessário.

    A cláusula ``WHERE`` garante que só sobrescreve quando o registo é
    mais recente **ou** algum campo de preço mudou — evitando escritas
    desnecessárias quando o scraper re-coleta o mesmo preço.

    Args:
        cur: Cursor de base de dados ativo.
        id_produto_loja: FK para ``produtos_loja``.
        item: Registo de produto do scraper.
    """
    c = _extrair_campos_preco(item)
    cur.execute(
        """
        INSERT INTO precos_atuais (
            id_produto_loja, preco_atual, preco_original,
            preco_unitario_valor, preco_unitario_unidade,
            em_promocao, data_recolha, agente_origem
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (id_produto_loja) DO UPDATE SET
            preco_atual              = EXCLUDED.preco_atual,
            preco_original           = EXCLUDED.preco_original,
            preco_unitario_valor     = EXCLUDED.preco_unitario_valor,
            preco_unitario_unidade   = EXCLUDED.preco_unitario_unidade,
            em_promocao              = EXCLUDED.em_promocao,
            data_recolha             = EXCLUDED.data_recolha,
            agente_origem            = EXCLUDED.agente_origem,
            data_ultima_atualizacao  = CURRENT_TIMESTAMP
        WHERE
            EXCLUDED.data_recolha >= precos_atuais.data_recolha
            AND (
                EXCLUDED.data_recolha > precos_atuais.data_recolha
                OR precos_atuais.preco_atual            IS DISTINCT FROM EXCLUDED.preco_atual
                OR precos_atuais.preco_original         IS DISTINCT FROM EXCLUDED.preco_original
                OR precos_atuais.preco_unitario_valor   IS DISTINCT FROM EXCLUDED.preco_unitario_valor
                OR precos_atuais.preco_unitario_unidade IS DISTINCT FROM EXCLUDED.preco_unitario_unidade
                OR precos_atuais.em_promocao            IS DISTINCT FROM EXCLUDED.em_promocao
                OR precos_atuais.agente_origem          IS DISTINCT FROM EXCLUDED.agente_origem
            );
        """,
        (
            id_produto_loja,
            c.preco_atual,
            c.preco_original,
            c.preco_unitario_valor,
            c.preco_unitario_unidade,
            c.em_promocao,
            c.data_recolha,
            c.agente_origem,
        ),
    )


# ---------------------------------------------------------------------------
# Orquestração da ingestão
# ---------------------------------------------------------------------------

def ingestao(registos: list[dict], dry_run: bool = False) -> tuple[int, list[dict]]:
    """Ingere uma lista de registos na base de dados.

    Cada registo é processado numa transação independente: uma falha num
    produto não afeta os restantes. Esta estratégia é deliberada — permite
    ingestão parcial com relatório de falhas, em vez de rollback total.

    Sequência por registo:
        1. Garantir loja.
        2. Garantir produto mestre (chave ou EAN).
        3. Garantir produto loja (SKU + loja).
        4. Inserir no histórico de preços (idempotente).
        5. Atualizar preço atual (só se necessário).

    Args:
        registos: Lista de registos já deduplicados.
        dry_run: Se ``True``, valida os registos sem escrever na BD.
                 Útil para detetar problemas antes de uma ingestão real.

    Returns:
        Tuplo (total_ok, lista_de_falhas).
        Cada falha é um dicionário com ``sku``, ``nome`` e ``erro``.
    """
    registos  = deduplicar_registos(registos)
    total     = len(registos)
    total_ok  = 0
    falhas:   list[dict] = []
    intervalo = max(1, total // 10)   # reportar progresso a cada ~10%

    if dry_run:
        print(f"[DRY-RUN] {total} registos a validar (sem escrita na BD)...")
        for item in registos:
            sku  = limpar_texto(item.get("id_externo", "?"))
            nome = limpar_texto(item.get("nome", "?"))
            try:
                construir_chave_mestre(item)
                _extrair_campos_preco(item)
                total_ok += 1
            except Exception as exc:
                falhas.append({"sku": sku, "nome": nome, "erro": str(exc)})
                print(f"  [FALHA] SKU={sku} | {nome[:60]} | {exc}")
        return total_ok, falhas

    with get_connection() as conn:
        for i, item in enumerate(registos, start=1):
            sku  = limpar_texto(item.get("id_externo", "?"))
            nome = limpar_texto(item.get("nome", "?"))
            try:
                with conn.cursor() as cur:
                    insignia = limpar_texto(item.get("loja"))
                    if not insignia:
                        raise ValueError("Registo sem campo 'loja'.")

                    id_loja            = obter_ou_criar_loja(cur, insignia=insignia)
                    id_produto_mestre  = obter_ou_criar_produto_mestre(cur, item)
                    id_produto_loja    = obter_ou_criar_produto_loja(cur, id_produto_mestre, id_loja, item)
                    inserir_historico_preco(cur, id_produto_loja, item)
                    upsert_preco_atual(cur, id_produto_loja, item)

                conn.commit()
                total_ok += 1

            except Exception as exc:
                conn.rollback()
                falhas.append({"sku": sku, "nome": nome, "erro": str(exc)})
                print(f"  [FALHA] SKU={sku} | {nome[:60]} | {exc}")

            if i % intervalo == 0 or i == total:
                pct = i / total * 100
                print(f"  [{i:>{len(str(total))}}/{total}] {pct:.0f}% — ok={total_ok} falhas={len(falhas)}")

    return total_ok, falhas


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    """Ponto de entrada CLI: carrega, deduplica e ingere registos de scrapers."""
    parser = argparse.ArgumentParser(
        description="Ingestão de ficheiros JSON dos scrapers para PostgreSQL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  python scripts/ingest.py --input scrapers/output/*.json
  python scripts/ingest.py --input scrapers/output/continente_*.json scrapers/output/pingo_doce_*.json
  python scripts/ingest.py --input scrapers/output/*.json --dry-run
        """,
    )
    parser.add_argument(
        "--input", nargs="+", required=True, metavar="FICHEIRO",
        help="Ficheiros JSON ou padrões glob (ex: scrapers/output/*.json)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Validar registos sem escrever na BD (útil para detetar problemas)",
    )
    args = parser.parse_args()

    print(f"[INGEST] A carregar ficheiros: {args.input}")
    registos, caminhos = carregar_ficheiros(args.input)
    print(f"[INGEST] {len(registos)} registos em {len(caminhos)} ficheiro(s).")

    if args.dry_run:
        print("[INGEST] Modo dry-run — nenhum dado será escrito na BD.")

    total_ok, falhas = ingestao(registos, dry_run=args.dry_run)

    separador = "=" * 50
    print(f"\n{separador}")
    print(f"  Ficheiros processados : {len(caminhos)}")
    print(f"  Registos processados  : {len(registos)}")
    print(f"  Ingeridos com sucesso : {total_ok}")
    print(f"  Com falha             : {len(falhas)}")
    if args.dry_run:
        print("  [DRY-RUN: nada foi escrito na BD]")
    print(separador)

    if falhas:
        print("\nDetalhes das falhas:")
        for f in falhas:
            print(f"  SKU {f['sku']!r}: {f['erro']}")


if __name__ == "__main__":
    main()
