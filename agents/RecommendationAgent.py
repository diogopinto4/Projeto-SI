"""
Agente SPADE de recomendação de compras de supermercado.

Encapsula o módulo ``models/recommender.py`` no paradigma de agentes,
expondo as suas quatro funcionalidades via mensagens FIPA-ACL.

Comportamentos:
    - ServeBehaviour (CyclicBehaviour): responde a pedidos de recomendação
      de qualquer agente ou cliente externo.

Protocolo de mensagens:
    Recebe (pesquisa de produtos):
        performative: query
        body: {"type": "pesquisar", "params": {"termo": str, "limite": int}}

    Recebe (melhor loja para produto):
        performative: query
        body: {"type": "melhor_loja", "params": {"produto": str, "top_n": int}}

    Recebe (otimizar lista de compras):
        performative: query
        body: {"type": "otimizar_lista", "params": {"lista": ["item1", "item2", ...]}}

    Recebe (recomendar momento de compra — usa LSTM internamente):
        performative: query
        body: {"type": "momento_compra",
               "params": {"produto_id":  int,
                          "caminho_csv": str,  (opcional)
                          "horizonte":   int,  (opcional)
                          "limiar":      float,(opcional)
                          "amostras":    int}} (opcional)

    Envia (sucesso):
        performative: inform
        body: resultado serializado via jsonpickle (dict ou lista de dicts)

    Envia (erro):
        performative: failure
        body: {"erro": str}
"""

from __future__ import annotations

import asyncio
import math
import sys
from pathlib import Path

import jsonpickle
import pandas as pd
from spade import agent
from spade.behaviour import CyclicBehaviour
from spade.message import Message

from agents.messaging import construir_reply


def _sanitizar_para_json(obj):
    """Substitui ``NaN``/``Infinity`` por ``None`` recursivamente.

    O JSON estrito não suporta NaN/Infinity. Embora a stack (jsonpickle +
    FastAPI) tolere alguns destes, em casos raros propagam-se como excepção
    e o utilizador vê o 500 default do FastAPI ("Internal Server Error")
    em vez de uma mensagem útil. Esta sanitização é a última defesa antes
    da serialização do payload.
    """
    if isinstance(obj, list):
        return [_sanitizar_para_json(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _sanitizar_para_json(v) for k, v in obj.items()}
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.recommender import (
    otimizar_lista_compras,
    otimizar_lista_compras_geo,
    otimizar_lista_compras_geo_multi_loja,
    pesquisar_produtos,
    produto_perto_de_mim,
    recomendar_melhor_loja,
    recomendar_momento_compra,
)

#: Caminho absoluto por omissão do dataset usado pela previsão LSTM interna.
DATASET_PATH_DEFAULT = str(Path(__file__).parent.parent / "data/generated/forecasting_dataset.csv")


class RecommendationAgent(agent.Agent):
    """Agente de recomendação de compras.

    Centraliza pesquisa de produtos, comparação de preços entre lojas,
    otimização de listas de compras e recomendação do momento ideal de compra.
    """

    DEFAULT_CONFIG: dict = {
        "dataset": DATASET_PATH_DEFAULT,
        "horizonte": 7,    # horizonte de previsão para momento_compra
        "limiar": 2.0,     # descida mínima (%) para recomendar espera
        "amostras": 50,    # simulações Monte Carlo para momento_compra
        "limite": 20,      # máximo de resultados por pesquisa
        "ativo": True,
    }

    def __init__(self, jid: str, password: str) -> None:
        super().__init__(jid, password)
        self.config = dict(self.DEFAULT_CONFIG)

    # ------------------------------------------------------------------
    # Comportamento de serviço
    # ------------------------------------------------------------------

    class ServeBehaviour(CyclicBehaviour):
        """Responde a pedidos de recomendação via FIPA-ACL."""

        # Tabela de despacho: tipo → método handler
        _HANDLERS: dict[str, str] = {
            "pesquisar": "_handle_pesquisar",
            "melhor_loja": "_handle_melhor_loja",
            "otimizar_lista": "_handle_otimizar_lista",
            "otimizar_lista_geo": "_handle_otimizar_lista_geo",
            "produto_perto_de_mim": "_handle_produto_perto_de_mim",
            "momento_compra": "_handle_momento_compra",
        }

        async def run(self) -> None:
            msg = await self.receive(timeout=10)
            if not msg:
                return

            if msg.get_metadata("performative") != "query":
                print(f"[RecommendationAgent] Performative inesperado: "
                      f"{msg.get_metadata('performative')!r}")
                return

            try:
                data = jsonpickle.decode(msg.body)
                query_type = data.get("type", "")
                handler = self._HANDLERS.get(query_type)

                if handler is None:
                    raise ValueError(f"Tipo de query desconhecido: {query_type!r}")

                await getattr(self, handler)(msg, data.get("params", {}))

            except Exception as exc:
                print(f"[RecommendationAgent] Erro ao processar mensagem: {exc}")
                await self._enviar_falha(msg, str(exc))

        # --------------------------------------------------------------
        # Handlers individuais
        # --------------------------------------------------------------

        async def _handle_pesquisar(self, msg: Message, params: dict) -> None:
            """Pesquisa produtos por palavra-chave."""
            config = self.agent.config
            termo = params.get("termo", "")
            if not termo.strip():
                raise ValueError("Parâmetro 'termo' em falta ou vazio.")

            limite = params.get("limite", config.get("limite", 20))
            # pesquisar_produtos é bloqueante (psycopg2) — corre em thread separada
            # para não bloquear o event loop do SPADE.
            df = await asyncio.to_thread(pesquisar_produtos, termo, limite=limite)

            if df is None:
                resultado = []
            else:
                # F1: substituir NaN por None — `to_dict` preserva NaN como
                # float('nan'), o que partia a serialização downstream em
                # alguns pares de pesquisa (ex: produtos com `precos_atuais`
                # vazio após cleanup do bug #30).
                df = df.where(pd.notna(df), None)
                resultado = _sanitizar_para_json(df.to_dict(orient="records"))

            await self._enviar_resposta(msg, resultado)

        async def _handle_melhor_loja(self, msg: Message, params: dict) -> None:
            """Devolve os melhores preços para um produto por loja."""
            produto = params.get("produto", "")
            if not produto.strip():
                raise ValueError("Parâmetro 'produto' em falta ou vazio.")

            top_n = params.get("top_n", 5)
            # recomendar_melhor_loja é bloqueante (psycopg2) — corre em thread separada.
            df = await asyncio.to_thread(recomendar_melhor_loja, produto, top_n=top_n)

            if df is None:
                resultado = []
            else:
                # Converter colunas datetime para string para serialização via jsonpickle
                for col in df.select_dtypes(include=["datetime64[ns]", "datetimetz"]).columns:
                    df[col] = df[col].astype(str)
                # F1: substituir NaN por None (ver _handle_pesquisar).
                df = df.where(pd.notna(df), None)
                resultado = _sanitizar_para_json(df.to_dict(orient="records"))

            await self._enviar_resposta(msg, resultado)

        async def _handle_otimizar_lista(self, msg: Message, params: dict) -> None:
            """Otimiza o custo de uma lista de compras entre lojas."""
            lista = params.get("lista", [])
            if not lista:
                raise ValueError("Parâmetro 'lista' em falta ou vazio.")

            # otimizar_lista_compras é bloqueante (psycopg2) — corre em thread separada.
            resultado = await asyncio.to_thread(otimizar_lista_compras, lista)

            if resultado is None:
                resultado = {"erro": "Nenhum item encontrado na BD."}

            await self._enviar_resposta(msg, resultado)

        async def _handle_otimizar_lista_geo(self, msg: Message, params: dict) -> None:
            """Otimiza a lista incluindo custo de deslocação à loja física mais próxima.

            Esta query estende ``otimizar_lista`` com geolocalização: requer
            coordenadas GPS do utilizador e calcula ``custo_total = produtos +
            2 × distância × €/km`` para cada cadeia com lista completa.

            Quando ``params['multi_loja'] is True``, avalia também a divisão
            da lista entre 2 cadeias (rota triangular ``user → A → B → user``),
            recomendando split apenas se for estritamente melhor que a melhor
            opção single-store.

            Nota: a implementação actual chama ``otimizar_lista_compras_geo``
            diretamente, que por sua vez usa ``models.geolocation`` para o
            cálculo de distâncias. A versão totalmente delegada ao
            ``LocationAgent`` via XMPP seria uma alternativa (mais "puramente
            multi-agente") mas adiciona latência de mensagens sem ganho
            funcional — o LocationAgent existe sobretudo para servir queries
            externas (clientes HTTP a perguntar "lojas próximas").
            """
            lista = params.get("lista", [])
            lat = params.get("lat")
            lon = params.get("lon")

            if not lista:
                raise ValueError("Parâmetro 'lista' em falta ou vazio.")
            if lat is None or lon is None:
                raise ValueError("Parâmetros 'lat' e 'lon' obrigatórios.")

            custo_km = params.get("custo_km")            # preset ou número
            raio_km = params.get("raio_km", 30.0)
            multi_loja = bool(params.get("multi_loja", False))

            # Função e respetivos kwargs — bloqueante (psycopg2 + cálculos),
            # corre em thread separada para não bloquear o event loop.
            fn = (otimizar_lista_compras_geo_multi_loja if multi_loja
                  else otimizar_lista_compras_geo)
            resultado = await asyncio.to_thread(
                fn,
                lista,
                user_lat=float(lat),
                user_lon=float(lon),
                custo_km=custo_km,
                raio_km=float(raio_km),
            )

            if resultado is None:
                resultado = {"erro": "Nenhum item encontrado na BD."}

            await self._enviar_resposta(msg, resultado)

        async def _handle_produto_perto_de_mim(self, msg: Message, params: dict) -> None:
            """Encontra um único produto nas cadeias e cruza com a loja física mais próxima.

            Diferente de ``otimizar_lista_geo`` que recomenda **uma** cadeia para
            uma lista — esta query devolve **todas** as cadeias alcançáveis com
            o produto, ordenadas por custo total (preço + 2 × distância × €/km).
            """
            termo = params.get("termo") or params.get("nome")
            lat = params.get("lat")
            lon = params.get("lon")

            if not termo or not str(termo).strip():
                raise ValueError("Parâmetro 'termo' (ou 'nome') em falta ou vazio.")
            if lat is None or lon is None:
                raise ValueError("Parâmetros 'lat' e 'lon' obrigatórios.")

            custo_km = params.get("custo_km")
            raio_km = params.get("raio_km", 30.0)
            top_n = params.get("top_n", 5)

            # produto_perto_de_mim é bloqueante (psycopg2 + cálculos) — thread separada.
            resultado = await asyncio.to_thread(
                produto_perto_de_mim,
                termo,
                user_lat=float(lat),
                user_lon=float(lon),
                custo_km=custo_km,
                raio_km=float(raio_km),
                top_n=int(top_n),
            )

            if resultado is None:
                resultado = {"erro": f"Produto '{termo}' não encontrado na BD."}
            elif not resultado:
                resultado = {"erro": f"Nenhuma cadeia tem '{termo}' no raio de {raio_km} km."}

            await self._enviar_resposta(msg, resultado)

        async def _handle_momento_compra(self, msg: Message, params: dict) -> None:
            """Recomenda o momento de compra com base em previsão LSTM."""
            config = self.agent.config
            produto_id = params.get("produto_id")
            if produto_id is None:
                raise ValueError("Parâmetro 'produto_id' em falta.")

            # recomendar_momento_compra é bloqueante (psycopg2 + PyTorch) — corre em thread separada.
            resultado = await asyncio.to_thread(
                recomendar_momento_compra,
                produto_id,
                caminho_csv=params.get("caminho_csv", config.get("dataset", DATASET_PATH_DEFAULT)),
                horizonte=params.get("horizonte", config.get("horizonte", 7)),
                limiar_descida_pct=params.get("limiar", config.get("limiar", 2.0)),
                n_amostras_mc=params.get("amostras", config.get("amostras", 50)),
            )

            if resultado is None:
                resultado = {
                    "erro": (
                        f"Produto {produto_id} sem modelo ou histórico disponível. "
                        "Verifica se o modelo está treinado: "
                        "python models/price_predictor.py --treinar"
                    )
                }

            await self._enviar_resposta(msg, resultado)

        # --------------------------------------------------------------
        # Utilitários de resposta (delegam para agents.messaging)
        # --------------------------------------------------------------

        async def _enviar_resposta(self, original: Message, resultado: object) -> None:
            """Envia reply ``inform`` ao remetente, propagando ``correlation_id``."""
            await self.send(construir_reply(original, "inform", resultado))

        async def _enviar_falha(self, original: Message, erro: str) -> None:
            """Envia reply ``failure`` ao remetente, propagando ``correlation_id``."""
            await self.send(construir_reply(original, "failure", {"erro": erro}))

    # ------------------------------------------------------------------
    # Setup do agente
    # ------------------------------------------------------------------

    async def setup(self) -> None:
        print("[RecommendationAgent] A iniciar...")
        self.add_behaviour(self.ServeBehaviour())
