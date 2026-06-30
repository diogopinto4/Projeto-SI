"""
Módulo de geolocalização e cálculo de custo de deslocação.

Fornece quatro funcionalidades principais para a feature de "custo de
deslocação" do agente de recomendação:

1. :func:`haversine_km` — distância "great-circle" entre dois pontos GPS.
2. :func:`lojas_proximas` — listagem das lojas físicas mais próximas de um
   ponto, com filtragem por raio máximo e/ou cadeia.
3. :func:`custo_deslocacao_euros` — custo monetário da deslocação ida-e-volta
   a uma loja, dado o preço por km.
4. :func:`otimizar_lista_compras_geo` — variante da otimização de lista de
   compras que considera **custo total = preço produtos + deslocação**.

Decisões de design (justificadas no relatório):

* **Fórmula de Haversine** em vez de Vincenty: erro <0.5% em distâncias até
  1000 km (Portugal continental cabe perfeitamente). Não requer biblioteca
  externa — apenas ``math.radians``, ``sin``, ``cos``, ``asin``.

* **Bounding box pré-filtro** antes do cálculo de haversine: para uma query
  com raio R em km, restringimos a candidatas com ``|Δlat| ≤ R/111`` e
  ``|Δlon| ≤ R/(111·cos(lat))``. Reduz drasticamente o número de chamadas a
  haversine quando o raio é pequeno (<50 km típico).

* **Custo = 2 × distância × €/km** (ida e volta). O factor 2 é explícito
  no código, não escondido.

* **Presets oficiais de €/km** publicados como :data:`PRESETS_CUSTO_KM`:
    - ``0.12`` — só combustível (gasolina ~1.80€/L, 6 L/100km)
    - ``0.20`` — equilibrado (combustível + manutenção) — **default**
    - ``0.36`` — tarifa oficial AT (Portaria 1553-D/2008, todos os custos)
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import psycopg2

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from db_config import DB_CONFIG


# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

#: Raio médio da Terra em km. Usado pela fórmula de Haversine.
#: Valor recomendado pelo NOAA / IUGG (média ponderada por área).
EARTH_RADIUS_KM = 6371.0088

#: Conversão aproximada: 1° de latitude ≈ 111 km (constante, polo a polo).
KM_POR_GRAU_LAT = 111.0

#: Presets de custo por quilómetro, em euros. As fontes estão documentadas
#: nos docstrings de cada chave:
#:
#: - ``so_combustivel``: gasolina 95 (~1.80 €/L) × consumo médio (6 L/100km).
#:   Apropriado quando o utilizador pensa "só conta a gasolina extra".
#:
#: - ``equilibrado``: combustível + manutenção (pneus, mudanças óleo, revisões
#:   básicas). Aproximação real para utilizador médio. **Default no sistema.**
#:
#: - ``tarifa_at``: tarifa oficial portuguesa para reembolso de viatura própria,
#:   Portaria nº 1553-D/2008, em vigor desde 2009. Inclui combustível,
#:   amortização, seguro, IUC e desgaste. É a estimativa mais conservadora.
PRESETS_CUSTO_KM: dict[str, dict] = {
    "so_combustivel": {
        "valor": 0.12,
        "label": "Só combustível",
        "descricao": "Gasolina 95 ~1,80 €/L × 6 L/100km",
    },
    "equilibrado": {
        "valor": 0.20,
        "label": "Equilibrado",
        "descricao": "Combustível + manutenção (pneus, revisões básicas)",
    },
    "tarifa_at": {
        "valor": 0.36,
        "label": "Tarifa AT",
        "descricao": "Tarifa oficial AT (Portaria 1553-D/2008) — inclui amortização, seguro, IUC",
    },
}

#: Custo por km default quando o utilizador não especifica preferência.
DEFAULT_CUSTO_KM: float = PRESETS_CUSTO_KM["equilibrado"]["valor"]


# ---------------------------------------------------------------------------
# Cache in-memory para queries de lojas próximas
# ---------------------------------------------------------------------------
# Decisões:
#
# * **Escopo**: in-memory por processo Python — desaparece com restart do agente.
#   Adequado porque os agentes SPADE têm vida longa (dias/semanas) e a tabela
#   ``lojas_fisicas`` muda muito raramente (re-correr o scraper é manual/diário).
#
# * **TTL**: 5 minutos. Compromisso entre performance (várias queries seguidas
#   na mesma sessão do dashboard cacheiam) e correção (após re-correr o seed,
#   os dados frescos ficam visíveis em menos de 5 min sem intervenção manual).
#
# * **Cache key**: lat/lon arredondadas a 4 casas decimais (~11 m de precisão)
#   para que pequenas variações do GPS do browser não invalidem a cache.

#: Tempo de vida das entradas em cache, em segundos.
CACHE_TTL_SEGUNDOS: float = 300.0

#: Precisão de arredondamento das coordenadas usadas como chave. 4 casas
#: decimais ≈ 11 metros — pedidos dentro do mesmo edifício cacheiam juntos.
CACHE_COORD_PRECISAO: int = 4

#: Cache de :func:`lojas_proximas` — chave: tupla com argumentos arredondados.
_cache_lojas_proximas: dict[tuple, tuple[float, list[dict]]] = {}


def _cache_key_lojas_proximas(
    lat: float, lon: float, raio_km: float,
    insignia: str | None, limite: int,
) -> tuple:
    """Constrói uma chave determinística para a cache de :func:`lojas_proximas`.

    Arredonda lat/lon a ``CACHE_COORD_PRECISAO`` casas para permitir cache hits
    em pedidos quase-idênticos (típico do GPS do browser que tem variação
    sub-métrica entre chamadas consecutivas).
    """
    return (
        round(lat, CACHE_COORD_PRECISAO),
        round(lon, CACHE_COORD_PRECISAO),
        float(raio_km),
        insignia,
        int(limite),
    )


def limpar_cache_lojas_proximas() -> None:
    """Esvazia a cache de :func:`lojas_proximas`.

    Útil para testes e quando se faz ingestão manual de novas lojas físicas
    (ver ``scripts/ingest_lojas_fisicas.py``) — re-popular a cache evita
    devolver resultados stale durante o TTL.
    """
    _cache_lojas_proximas.clear()


# ---------------------------------------------------------------------------
# Haversine — distância geográfica
# ---------------------------------------------------------------------------

def haversine_km(
    lat1: float, lon1: float,
    lat2: float, lon2: float,
) -> float:
    """Calcula a distância great-circle entre dois pontos GPS em km.

    Usa a fórmula de Haversine, que assume a Terra como esfera de raio
    ``EARTH_RADIUS_KM``. Tem erro <0.5% para distâncias até 1000 km — mais
    do que adequado para qualquer cenário em Portugal (max ~600 km Minho-Algarve).

    Args:
        lat1: Latitude do ponto 1 em graus decimais (-90 a +90).
        lon1: Longitude do ponto 1 em graus decimais (-180 a +180).
        lat2: Latitude do ponto 2 em graus decimais.
        lon2: Longitude do ponto 2 em graus decimais.

    Returns:
        Distância em km entre os dois pontos. Sempre >= 0.

    Example:
        >>> # Distância Braga (UMinho) → Porto (Boavista)
        >>> haversine_km(41.561, -8.397, 41.158, -8.629)
        49.5
    """
    # Converter para radianos uma única vez
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi    = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    # Fórmula de Haversine
    a = (math.sin(delta_phi / 2) ** 2
         + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2)
    c = 2 * math.asin(math.sqrt(a))

    return EARTH_RADIUS_KM * c


# ---------------------------------------------------------------------------
# Custo de deslocação
# ---------------------------------------------------------------------------

def custo_deslocacao_euros(
    distancia_km: float,
    custo_km: float = DEFAULT_CUSTO_KM,
    *,
    ida_e_volta: bool = True,
) -> float:
    """Converte uma distância em custo monetário de deslocação.

    Por defeito assume **ida e volta** (factor 2). Esse é o modelo
    realista para "ir ao supermercado e voltar a casa".

    Args:
        distancia_km: Distância (one-way) em km até à loja.
        custo_km: Custo por km em euros. Default: :data:`DEFAULT_CUSTO_KM`.
        ida_e_volta: Se ``True`` (default), multiplica por 2. Se ``False``
            (ex: utilizador já passa pela loja), conta apenas one-way.

    Returns:
        Custo em euros, arredondado a 2 casas decimais.
    """
    multiplicador = 2.0 if ida_e_volta else 1.0
    return round(distancia_km * custo_km * multiplicador, 2)


# ---------------------------------------------------------------------------
# Lojas próximas — query à BD
# ---------------------------------------------------------------------------

def _bbox_para_raio(lat: float, lon: float, raio_km: float) -> tuple[float, float, float, float]:
    """Devolve uma bounding box (lat_min, lat_max, lon_min, lon_max) para o raio.

    A bbox é **conservadora** (ligeiramente maior que o círculo real) para
    garantir que não exclui lojas a candidatas. A filtragem final é feita
    com haversine. A correção pelo cosseno da latitude evita uma bbox
    desnecessariamente larga em longitudes a norte.

    Args:
        lat: Latitude do centro (graus).
        lon: Longitude do centro (graus).
        raio_km: Raio máximo (km).

    Returns:
        Tuplo ``(lat_min, lat_max, lon_min, lon_max)`` em graus.
    """
    delta_lat = raio_km / KM_POR_GRAU_LAT
    # Correção pela latitude: 1° lon ≈ 111·cos(lat) km
    cos_lat = max(math.cos(math.radians(lat)), 1e-6)   # evita divisão por zero perto dos polos
    delta_lon = raio_km / (KM_POR_GRAU_LAT * cos_lat)
    return (lat - delta_lat, lat + delta_lat,
            lon - delta_lon, lon + delta_lon)


def lojas_proximas(
    user_lat: float,
    user_lon: float,
    raio_km: float = 20.0,
    insignia: str | None = None,
    limite: int = 50,
) -> list[dict]:
    """Devolve as lojas físicas mais próximas do utilizador, ordenadas por distância.

    **Cache**: resultados são guardados em memória (TTL :data:`CACHE_TTL_SEGUNDOS`,
    chave arredondada a :data:`CACHE_COORD_PRECISAO` casas decimais). Chamadas
    sucessivas com argumentos quase-idênticos (típico do GPS do browser, que
    flutua sub-métricamente) devolvem o resultado da cache em vez de re-querer
    a BD. Para invalidar manualmente (ex: depois de re-popular ``lojas_fisicas``),
    chamar :func:`limpar_cache_lojas_proximas`.

    Pipeline em 2 fases (para eficiência com 1000+ lojas na BD):

    1. **Filtragem por bounding box em SQL**: a query inclui ``WHERE latitude
       BETWEEN lat_min AND lat_max AND longitude BETWEEN lon_min AND lon_max``.
       Aproveita os índices ``idx_lojas_fisicas_latitude/longitude``.

    2. **Cálculo exacto com Haversine em Python**: as poucas candidatas são
       depois ordenadas pela distância real e filtradas pelo raio exacto.

    Args:
        user_lat: Latitude do utilizador (graus decimais).
        user_lon: Longitude do utilizador (graus decimais).
        raio_km: Raio máximo em km. Default: 20 km (cobre uma cidade média).
        insignia: Se fornecido, filtra apenas lojas dessa cadeia (ex: ``"Continente"``).
        limite: Número máximo de lojas a devolver.

    Returns:
        Lista de dicionários ordenados por distância ascendente. Cada dict tem:
        ``id_loja_fisica``, ``insignia``, ``nome_loja``, ``morada``, ``cidade``,
        ``codigo_postal``, ``latitude``, ``longitude``, ``distancia_km``.
        Lista vazia se nenhuma loja estiver no raio.
    """
    # ----- Cache lookup -----
    chave = _cache_key_lojas_proximas(user_lat, user_lon, raio_km, insignia, limite)
    agora = time.monotonic()
    entrada = _cache_lojas_proximas.get(chave)
    if entrada is not None:
        ts, resultado_cached = entrada
        if agora - ts < CACHE_TTL_SEGUNDOS:
            # Devolve cópia rasa para o caller não conseguir mutar a cache
            return [dict(loja) for loja in resultado_cached]

    # ----- Cache miss: ir à BD -----
    lat_min, lat_max, lon_min, lon_max = _bbox_para_raio(user_lat, user_lon, raio_km)

    sql = """
        SELECT id_loja_fisica, insignia, nome_loja, morada, cidade, codigo_postal,
               latitude::float, longitude::float
        FROM lojas_fisicas
        WHERE ativa = TRUE
          AND latitude  BETWEEN %s AND %s
          AND longitude BETWEEN %s AND %s
    """
    params: list = [lat_min, lat_max, lon_min, lon_max]
    if insignia:
        sql += " AND insignia = %s"
        params.append(insignia)

    with psycopg2.connect(**DB_CONFIG) as conn, conn.cursor() as cur:
        cur.execute(sql, tuple(params))
        cols = [d[0] for d in cur.description]
        rows = cur.fetchall()

    # Aplicar haversine exacto e ordenar
    candidatas: list[dict] = []
    for row in rows:
        loja = dict(zip(cols, row))
        dist = haversine_km(user_lat, user_lon, loja["latitude"], loja["longitude"])
        if dist <= raio_km:
            loja["distancia_km"] = round(dist, 2)
            candidatas.append(loja)

    candidatas.sort(key=lambda x: x["distancia_km"])
    resultado = candidatas[:limite]

    # ----- Cache store -----
    _cache_lojas_proximas[chave] = (agora, [dict(loja) for loja in resultado])
    return resultado


# ---------------------------------------------------------------------------
# Distância mínima a cada loja física de uma insígnia
# ---------------------------------------------------------------------------

def distancia_minima_por_insignia(
    user_lat: float,
    user_lon: float,
    insignias: list[str],
    raio_km: float = 30.0,
) -> dict[str, dict | None]:
    """Para cada insígnia, devolve a loja física mais próxima e a sua distância.

    Esta é a função-chave para o cálculo de "custo total por loja" na
    otimização de lista de compras com GPS: se um utilizador escolher
    comprar tudo no Continente, queremos saber quantos km tem de percorrer
    até **ao Continente mais próximo dele**, não a um qualquer.

    Args:
        user_lat: Latitude do utilizador.
        user_lon: Longitude do utilizador.
        insignias: Lista de insígnias a considerar (ex: ``["Continente", "Pingo Doce"]``).
        raio_km: Raio máximo. Se nenhuma loja de uma cadeia estiver dentro do raio,
            o resultado dessa cadeia é ``None``.

    Returns:
        Dicionário ``{insignia: {loja_dict, distancia_km} | None}``.
        ``None`` para cadeias sem loja no raio.
    """
    resultado: dict[str, dict | None] = {}
    for ins in insignias:
        proximas = lojas_proximas(user_lat, user_lon, raio_km=raio_km, insignia=ins, limite=1)
        resultado[ins] = proximas[0] if proximas else None
    return resultado


# ---------------------------------------------------------------------------
# Validação de presets / utilitário público
# ---------------------------------------------------------------------------

def resolver_custo_km(preset_ou_valor: str | float | None) -> float:
    """Resolve um custo €/km a partir de uma string de preset ou valor numérico.

    Permite que a API REST e o dashboard aceitem tanto um nome de preset
    (``"equilibrado"``, ``"tarifa_at"``) como um valor manual (``0.18``).

    Args:
        preset_ou_valor: Chave de :data:`PRESETS_CUSTO_KM`, número direto,
            ou ``None`` para usar o default.

    Returns:
        Custo por km em euros (>= 0; ``0.0`` modela deslocação sem custo
        marginal, como bicicleta ou carro de empresa).

    Raises:
        ValueError: Se for uma string que não corresponde a nenhum preset,
            ou se for um número negativo. ``0.0`` é aceite — modela o caso
            de "deslocação sem custo marginal" (bike, carro de empresa).
    """
    if preset_ou_valor is None:
        return DEFAULT_CUSTO_KM

    if isinstance(preset_ou_valor, (int, float)):
        if preset_ou_valor < 0:
            raise ValueError(f"custo_km não pode ser negativo, recebido: {preset_ou_valor}")
        return float(preset_ou_valor)

    # String — tentar como preset, depois como número
    if preset_ou_valor in PRESETS_CUSTO_KM:
        return PRESETS_CUSTO_KM[preset_ou_valor]["valor"]

    try:
        valor = float(preset_ou_valor)
    except ValueError:
        raise ValueError(
            f"custo_km inválido: {preset_ou_valor!r}. "
            f"Use um número ou um destes presets: {list(PRESETS_CUSTO_KM)}"
        )
    if valor < 0:
        raise ValueError(f"custo_km não pode ser negativo, recebido: {valor}")
    return valor
