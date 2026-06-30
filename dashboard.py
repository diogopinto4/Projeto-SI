"""
dashboard.py — Interface visual do sistema de análise de preços de supermercado.

Cobre todas as funcionalidades do projeto:
  - Pesquisa de produtos por palavra-chave (pg_trgm)
  - Comparação de preços entre lojas (melhor loja)
  - Otimização de lista de compras (custo mínimo por loja)
  - Previsão de preço LSTM (determinista + Monte Carlo Dropout)
  - Histórico de preços com gráfico temporal
  - Preços atuais por loja com distribuição

Arranque:
    streamlit run dashboard.py

Pré-requisito: API REST a correr em http://localhost:8000
    python main.py

Pode usar a variável de ambiente API_BASE para apontar o dashboard para outra API.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from streamlit_geolocation import streamlit_geolocation
from streamlit_local_storage import LocalStorage


# ---------------------------------------------------------------------------
# Configuração global
# ---------------------------------------------------------------------------

API_BASE = os.getenv("API_BASE", "http://localhost:8000").rstrip("/")
LOJAS = ["Continente", "Pingo Doce", "Auchan"]

st.set_page_config(
    page_title="Análise de Preços de Supermercado",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Utilitários HTTP
# ---------------------------------------------------------------------------

def api_get(path: str, params: dict | None = None, silent: bool = False) -> dict | list | None:
    """GET à API REST. Devolve None e exibe erro em caso de falha.

    Se ``silent=True``, não exibe mensagens de erro (útil para sondagens internas).
    """
    try:
        r = requests.get(f"{API_BASE}{path}", params=params, timeout=30)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        if not silent:
            st.error(
                "Não foi possível ligar à API. "
                "Verifica se `python main.py` está a correr."
            )
        return None
    except requests.exceptions.HTTPError:
        if not silent:
            try:
                detail = r.json().get("detail", r.text)
            except Exception:
                detail = r.text
            st.error(f"Erro da API ({r.status_code}): {detail}")
        return None
    except Exception as exc:
        if not silent:
            st.error(f"Erro inesperado: {exc}")
        return None


def api_post(path: str, json: dict, silent: bool = False) -> dict | list | None:
    """POST à API REST. Devolve None e exibe erro em caso de falha.

    Se ``silent=True``, não exibe mensagens de erro (útil para sondagens
    internas em que o erro técnico não acrescenta valor ao utilizador).
    """
    try:
        r = requests.post(f"{API_BASE}{path}", json=json, timeout=60)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        if not silent:
            st.error(
                "Não foi possível ligar à API. "
                "Verifica se `python main.py` está a correr."
            )
        return None
    except requests.exceptions.HTTPError:
        if not silent:
            try:
                detail = r.json().get("detail", r.text)
            except Exception:
                detail = r.text
            st.error(f"Erro da API ({r.status_code}): {detail}")
        return None
    except Exception as exc:
        if not silent:
            st.error(f"Erro inesperado: {exc}")
        return None


def _ids_candidatos_para_nome(nome: str, top_n: int = 10, silent: bool = False) -> list[dict]:
    """Devolve até top_n candidatos (id, nome, loja, preco) via /produtos/melhor-loja.

    Usado pelas tabs de Previsão e Histórico para desambiguar entre as várias
    variantes/lojas de um produto (ex: "arroz agulha" resolve para dezenas de
    produtos diferentes em 3 cadeias). O preço permite ao utilizador distinguir
    o mesmo produto entre lojas no dropdown de desambiguação.
    """
    resultado = api_get("/produtos/melhor-loja", {"nome": nome, "top_n": top_n}, silent=silent)
    if not resultado:
        return []
    return [
        {
            "id": r.get("id_produto_loja"),
            "nome": r.get("nome_padronizado", nome),
            "loja": r.get("loja", ""),
            "preco": r.get("preco_atual"),
        }
        for r in resultado
        if r.get("id_produto_loja") is not None
    ]


def _label_candidato(c: dict) -> str:
    """Formata um candidato como 'Nome (Loja) — 1.39€' para o dropdown de desambiguação."""
    loja = f" ({c['loja']})" if c.get("loja") else ""
    preco = f" — {float(c['preco']):.2f}€" if c.get("preco") is not None else ""
    return f"{c['nome']}{loja}{preco}"


def _fmt_preco(valor) -> str:
    """Formata um preço float como string com 2 casas decimais."""
    try:
        return f"{float(valor):.2f}"
    except (TypeError, ValueError):
        return "—"


# ---------------------------------------------------------------------------
# Sidebar — estado do sistema
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("🛒 SuperMarket AI")
    st.caption(
        "Universidade do Minho  \n"
        "Projeto em Sistemas Inteligentes  \n"
        "2025/2026"
    )
    st.divider()

    st.subheader("Estado do Sistema")
    health = api_get("/saude")

    if health and health.get("status") == "ok":
        st.success("API online")
        for nome, jid in health.get("agentes", {}).items():
            st.caption(f"• {nome}: `{jid}`")
    else:
        st.error("API offline")
        st.info("Arranca o sistema:\n```\npython main.py\n```")

    st.divider()
    st.caption("Documentação: [/docs](http://localhost:8000/docs)")


# ---------------------------------------------------------------------------
# Cabeçalho
# ---------------------------------------------------------------------------

st.title("Análise de Preços de Supermercado")
st.caption(
    "Pesquisa · Comparação entre lojas · Otimização de lista · "
    "Previsão LSTM · Histórico de preços"
)

(
    tab_pesquisa,
    tab_lista,
    tab_geo,
    tab_previsao,
    tab_historico,
    tab_loja,
    tab_validacao,
) = st.tabs([
    "🔍 Pesquisa & Comparação",
    "📋 Lista de Compras",
    "📍 Lista com Localização",
    "📈 Previsão de Preços",
    "📅 Histórico de Preços",
    "🏬 Preços por Loja",
    "✅ Validação",
])


# ===========================================================================
# Persistência via localStorage — partilhada entre tabs
# ===========================================================================

#: Instância singleton de LocalStorage (componente JS via Streamlit).
#: É criada uma única vez por sessão para evitar múltiplas montagens.
_local_storage = LocalStorage()

#: Caminho do ficheiro JSON com as listas de compras guardadas.
#:
#: Razão de não usar ``localStorage`` (como a primeira implementação tinha):
#: a library ``streamlit-local-storage`` 0.0.25 tem race conditions
#: reproduzíveis — na 1ª render após F5 o ``getItem`` devolve ``None`` mesmo
#: quando o valor está no browser. Isto fazia parecer que as listas
#: desapareciam. Persistência server-side em ficheiro JSON é instantânea,
#: não tem race conditions, e é trivialmente backupável.
#:
#: Trade-off assumido: as listas ficam **partilhadas entre todas as sessões**
#: que usem a mesma instância do dashboard (não são per-utilizador). Para o
#: contexto académico isto é aceitável — só uma pessoa usa o dashboard.
#: Para multi-utilizador no futuro, separar por session_id ou por utilizador
#: autenticado.
LISTAS_FILE_PATH = Path(__file__).parent / "data" / "user_data" / "listas_compras.json"


def _listas_guardadas_get_all() -> dict[str, str]:
    """Lê todas as listas de compras guardadas do ficheiro JSON server-side.

    Returns:
        Dict ``{nome_lista: texto_da_lista}``. Vazio se o ficheiro não
        existir ou estiver corrompido.
    """
    if not LISTAS_FILE_PATH.exists():
        return {}
    try:
        with open(LISTAS_FILE_PATH, encoding="utf-8") as f:
            d = json.load(f)
        if isinstance(d, dict):
            return {str(k): str(v) for k, v in d.items() if isinstance(v, str)}
    except (OSError, json.JSONDecodeError, ValueError):
        pass
    return {}


def _lista_guardada_save(nome: str, conteudo: str) -> None:
    """Adiciona ou actualiza uma lista guardada (overwrite por nome)."""
    listas = _listas_guardadas_get_all()
    listas[nome] = conteudo
    LISTAS_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LISTAS_FILE_PATH, "w", encoding="utf-8") as f:
        json.dump(listas, f, ensure_ascii=False, indent=2)


def _lista_guardada_delete(nome: str) -> None:
    """Remove uma lista guardada do ficheiro JSON."""
    listas = _listas_guardadas_get_all()
    if nome in listas:
        del listas[nome]
        LISTAS_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(LISTAS_FILE_PATH, "w", encoding="utf-8") as f:
            json.dump(listas, f, ensure_ascii=False, indent=2)


def _formatar_nome_loja(texto: str | None) -> str:
    """Title-case texto em CAPS LOCK preservando stop-words PT em minúscula.

    Cerca de 265 lojas Pingo Doce vêm da BD com nome, morada e cidade em
    maiúsculas ("LARGO DO RATO", "RUA DOS COMBATENTES — COIMBRA"), porque
    o scraping preserva o output do site oficial. Auchan e Continente
    vêm em Title Case nativo. Esta função uniformiza a apresentação no
    dashboard sem alterar a BD.

    Stop-words PT (``da``, ``de``, ``do``, ``das``, ``dos``, ``e``) ficam
    em minúscula excepto se forem a primeira palavra — produz
    "Rua dos Combatentes" em vez do "Rua Dos Combatentes" que ``.title()``
    daria por defeito.
    """
    if not texto or not isinstance(texto, str):
        return texto or ""
    # "Está em CAPS LOCK" = todas as letras com casing estão em maiúscula.
    # ``c.upper() != c.lower()`` filtra caracteres sem distinção
    # maiúscula/minúscula (``º``, ``ª``, ordinais, dígitos), evitando que
    # "CASTELO BRANCO - AV. 1º MAIO" passe intacto só por causa do ``º``
    # — que o Unicode classifica como ``.islower() == True`` apesar de
    # não ser letra de casing.
    letras_cased = [c for c in texto if c.isalpha() and c.upper() != c.lower()]
    if not letras_cased or any(c.islower() for c in letras_cased):
        return texto
    stop_words = {"da", "de", "do", "das", "dos", "e"}
    palavras = texto.title().split(" ")
    return " ".join(
        p.lower() if i > 0 and p.lower() in stop_words else p
        for i, p in enumerate(palavras)
    )


# ===========================================================================
# Tab 1 — Pesquisa & Comparação entre lojas
# ===========================================================================

with tab_pesquisa:

    # ---- Pesquisa de produtos ----
    st.subheader("Pesquisa de Produtos")
    st.caption("Procura produtos na base de dados por palavra-chave (similaridade pg_trgm).")

    col_t, col_l = st.columns([4, 1])
    with col_t:
        termo = st.text_input(
            "Termo de pesquisa",
            placeholder="ex: arroz agulha, leite meio-gordo, azeite…",
            key="p_termo",
        )
    with col_l:
        limite = st.number_input("Limite", min_value=5, max_value=100, value=20, key="p_limite")

    if st.button("Pesquisar", type="primary", key="btn_pesquisar"):
        if not termo.strip():
            st.warning("Introduz um termo de pesquisa.")
        elif len(termo.strip()) < 2:
            st.warning("O termo de pesquisa deve ter pelo menos 2 caracteres.")
        else:
            with st.spinner("A pesquisar na base de dados…"):
                resultado = api_get("/produtos/pesquisar", {"termo": termo, "limite": limite})

            if resultado is not None:
                if not resultado:
                    st.info(f"Nenhum produto encontrado para '{termo}'.")
                else:
                    st.success(f"**{len(resultado)}** produto(s) encontrado(s).")
                    df = pd.DataFrame(resultado)
                    col_rename = {
                        "nome_padronizado": "Produto",
                        "marca": "Marca",
                        "quantidade_valor": "Qtd.",
                        "quantidade_unidade": "Un.",
                        "num_lojas": "Lojas",
                        "preco_min": "Preço Mín. (€)",
                        "preco_max": "Preço Máx. (€)",
                    }
                    df_show = df.rename(columns={k: v for k, v in col_rename.items() if k in df.columns})
                    for col in ["Preço Mín. (€)", "Preço Máx. (€)"]:
                        if col in df_show.columns:
                            df_show[col] = df_show[col].apply(_fmt_preco)
                    st.dataframe(df_show, width="stretch", hide_index=True)

    st.divider()

    # ---- Melhor loja ----
    st.subheader("Melhor Loja para um Produto")
    st.caption(
        "Compara o preço do produto entre todas as lojas disponíveis na BD. "
        "Marca **'incluir deslocação'** para considerar também a loja física "
        "mais próxima de ti e ordenar por custo total."
    )

    col_p, col_n = st.columns([4, 1])
    with col_p:
        prod_ml = st.text_input(
            "Nome do produto",
            placeholder="ex: arroz agulha, azeite virgem extra…",
            key="ml_produto",
        )
    with col_n:
        top_n = st.number_input("Top N", min_value=1, max_value=20, value=5, key="ml_topn")

    # Toggle: incluir custo de deslocação
    incluir_geo = st.checkbox(
        "📍 Incluir custo de deslocação à loja física mais próxima",
        value=False,
        key="ml_incluir_geo",
        help=(
            "Quando activo, mostra também a distância à loja física mais próxima "
            "de ti em cada cadeia e ordena por custo total (preço + deslocação). "
            "Usa a localização definida na tab '📍 Lista com Localização' (GPS do "
            "browser ou manual)."
        ),
    )

    # Recuperar localização do session_state (definida pela tab GPS).
    # 'geo_lat' e 'geo_lon' são os keys dos number_input no expander manual,
    # mas o GPS do browser não tem key. Vamos derivar de forma robusta:
    #   - se geo_manual_loc estiver setado, usar.
    #   - senão, usar geo_lat/geo_lon dos inputs.
    user_loc_lat = None
    user_loc_lon = None
    manual_loc_state = st.session_state.get("geo_manual_loc")
    if manual_loc_state is not None:
        user_loc_lat, user_loc_lon = manual_loc_state
    elif "geo_lat" in st.session_state and "geo_lon" in st.session_state:
        # Inputs do expander manual têm um valor default; só os usamos se o
        # utilizador esteve na tab GPS pelo menos uma vez (i.e., os keys
        # existem no session_state).
        user_loc_lat = st.session_state.get("geo_lat")
        user_loc_lon = st.session_state.get("geo_lon")

    if incluir_geo and (user_loc_lat is None or user_loc_lon is None):
        st.warning(
            "Define a tua localização primeiro na tab **📍 Lista com Localização**. "
            "O sistema vai reutilizá-la aqui."
        )

    if st.button("Comparar Lojas", type="primary", key="btn_melhor_loja"):
        if not prod_ml.strip():
            st.warning("Introduz o nome do produto.")
        elif incluir_geo and (user_loc_lat is None or user_loc_lon is None):
            st.warning("Define a localização primeiro (tab 📍 Lista com Localização).")
        elif incluir_geo:
            # Caminho geo: /produtos/perto-de-mim
            with st.spinner(f"A comparar preços + deslocação a partir de ({user_loc_lat:.4f}, {user_loc_lon:.4f})…"):
                resultado = api_get(
                    "/produtos/perto-de-mim",
                    {"nome": prod_ml, "lat": user_loc_lat, "lon": user_loc_lon,
                     "top_n": top_n},
                )
            if resultado is None:
                st.info(f"Nenhum resultado para '{prod_ml}'.")
            elif isinstance(resultado, dict) and "erro" in resultado:
                st.error(resultado["erro"])
            elif not resultado:
                st.info(f"Nenhum resultado para '{prod_ml}'.")
            else:
                df = pd.DataFrame(resultado)
                # Tabela
                cols_disp = ["insignia", "produto", "preco_atual", "em_promocao",
                             "distancia_km", "custo_deslocacao", "custo_total"]
                df_show = df[cols_disp].rename(columns={
                    "insignia":          "Cadeia",
                    "produto":           "Produto",
                    "preco_atual":       "Preço (€)",
                    "em_promocao":       "Promoção",
                    "distancia_km":      "Distância (km)",
                    "custo_deslocacao":  "Deslocação (€)",
                    "custo_total":       "Total (€)",
                })
                st.dataframe(df_show, width="stretch", hide_index=True)

                # Gráfico de barras empilhadas: preço + deslocação por cadeia
                df_chart = df[["insignia", "preco_atual", "custo_deslocacao"]].rename(columns={
                    "insignia":         "Cadeia",
                    "preco_atual":      "Preço",
                    "custo_deslocacao": "Deslocação",
                })
                fig = px.bar(
                    df_chart, x="Cadeia", y=["Preço", "Deslocação"],
                    title=f"Custo total para '{prod_ml}' por cadeia",
                    labels={"value": "€", "variable": ""},
                    color_discrete_map={"Preço": "#1976D2", "Deslocação": "#FB8C00"},
                )
                fig.update_layout(barmode="stack", legend=dict(orientation="h", y=1.1))
                st.plotly_chart(fig, width="stretch")

                # Caixa de recomendação para a cadeia top
                top = resultado[0]
                st.success(
                    f"**Mais barato: {top['insignia']}** — total **{top['custo_total']:.2f}€** "
                    f"({top['preco_atual']:.2f}€ produto + {top['custo_deslocacao']:.2f}€ "
                    f"deslocação a {top['distancia_km']:.1f} km).  \n"
                    f"Loja: *{_formatar_nome_loja(top['loja_fisica']['nome_loja'])}*"
                )
        else:
            # Caminho sem deslocação (comportamento original)
            with st.spinner("A comparar preços entre lojas…"):
                resultado = api_get("/produtos/melhor-loja", {"nome": prod_ml, "top_n": top_n})

            if resultado is not None:
                if not resultado:
                    st.info(f"Nenhum resultado para '{prod_ml}'.")
                else:
                    df = pd.DataFrame(resultado)
                    col_tbl, col_chart = st.columns([1, 1])

                    with col_tbl:
                        cols_disp = [c for c in [
                            "nome_padronizado", "loja", "preco_atual", "em_promocao",
                            "poupanca_pct", "preco_unitario_valor",
                            "preco_unitario_unidade", "ultima_atualizacao",
                        ] if c in df.columns]
                        df_show = df[cols_disp].rename(columns={
                            "nome_padronizado": "Produto",
                            "loja": "Loja",
                            "preco_atual": "Preço (€)",
                            "em_promocao": "Em promoção (site)",
                            "poupanca_pct": "Desconto vs mais caro (%)",
                            "preco_unitario_valor": "P. Unit.",
                            "preco_unitario_unidade": "Un.",
                            "ultima_atualizacao": "Atualizado",
                        })
                        if "Preço (€)" in df_show.columns:
                            df_show["Preço (€)"] = df_show["Preço (€)"].apply(_fmt_preco)
                        st.dataframe(df_show, width="stretch", hide_index=True)
                        st.caption(
                            "**Em promoção (site)** = flag enviada pelo retailer (preço barrado). "
                            "**Desconto vs mais caro** = quanto este preço é mais baixo do que o "
                            "mais caro listado. São métricas independentes — uma promoção pequena "
                            "pode ter desconto baixo; um preço regular barato pode ter desconto "
                            "alto sem estar em promoção."
                        )

                    with col_chart:
                        if (
                            "loja" in df.columns
                            and "preco_atual" in df.columns
                            and "nome_padronizado" in df.columns
                        ):
                            # F3 — barras horizontais, **uma por produto**.
                            # O gráfico stacked-by-loja anterior era ilegível
                            # quando todos os resultados eram da mesma cadeia
                            # (ex: "café delta" → 5 produtos Auchan empilhados
                            # numa única barra). Esta visualização funciona em
                            # ambos os cenários (uma cadeia ou várias).
                            df_sorted = df.sort_values("preco_atual", ascending=True).copy()
                            df_sorted["_label"] = df_sorted["nome_padronizado"].apply(
                                lambda n: n if len(str(n)) <= 55 else str(n)[:52] + "..."
                            )
                            cores_por_loja = {
                                "Continente": "#E53935",
                                "Pingo Doce": "#43A047",
                                "Auchan":     "#FB8C00",
                            }
                            fig = px.bar(
                                df_sorted,
                                y="_label",
                                x="preco_atual",
                                color="loja",
                                orientation="h",
                                text="preco_atual",
                                labels={
                                    "_label":      "Produto",
                                    "preco_atual": "Preço (€)",
                                    "loja":        "Loja",
                                },
                                title=f"Preços para '{prod_ml}'",
                                color_discrete_map=cores_por_loja,
                            )
                            fig.update_traces(
                                texttemplate="%{text:.2f}€",
                                textposition="outside",
                            )
                            fig.update_layout(
                                yaxis=dict(autorange="reversed"),   # mais barato no topo
                                legend=dict(orientation="h", y=1.12),
                                margin=dict(t=80, b=10, l=10, r=30),
                                height=max(280, 60 * len(df_sorted) + 80),
                            )
                            st.plotly_chart(fig, width="stretch")
                            st.caption(
                                "Cada barra é um **SKU único** (combinação nome+marca+"
                                "tamanho). Produtos vendidos em múltiplas cadeias podem "
                                "aparecer como linhas distintas quando o sistema não tem "
                                "EAN partilhado — usa o filtro por marca/tamanho para "
                                "comparar SKUs equivalentes."
                            )


# ===========================================================================
# Tab 2 — Otimização de lista de compras
# ===========================================================================

with tab_lista:
    st.subheader("Otimização de Lista de Compras")
    st.caption(
        "Introduz os produtos (um por linha). "
        "O sistema encontra onde comprar cada item ao menor custo e calcula o "
        "custo total por loja."
    )

    # ---- Mensagens pendentes (sobrevivem ao st.rerun) ----
    # Flag de "operação concluída" definida pelos botões Guardar/Apagar antes
    # do rerun — mostramos a mensagem no topo da tab e limpamos. Sem isto, o
    # ``st.success`` chamado mesmo antes de ``st.rerun()`` desaparece antes do
    # utilizador o conseguir ler.
    if "_lista_msg" in st.session_state:
        msg_tipo, msg_texto = st.session_state.pop("_lista_msg")
        getattr(st, msg_tipo)(msg_texto)

    # ---- Listas guardadas (persistência server-side em ficheiro JSON) ----
    # Renderizamos sempre o selectbox, mesmo que ``listas_guardadas`` seja vazio,
    # para manter a UI estável e o controlo visível.
    listas_guardadas = _listas_guardadas_get_all()
    col_load, col_del = st.columns([3, 1])
    with col_load:
        opcoes_lista = ["—"] + sorted(listas_guardadas.keys())
        nome_a_carregar = st.selectbox(
            "Carregar lista guardada",
            options=opcoes_lista,
            key="lista_carregar_select",
            help=(
                "Listas guardadas em ficheiro JSON no servidor "
                "(`data/user_data/listas_compras.json`). Persistem entre "
                "sessões e F5."
            ),
        )
    with col_del:
        st.write("")  # alinhamento vertical com o selectbox
        st.write("")
        if nome_a_carregar != "—" and st.button(
            "Apagar", key="btn_apagar_lista",
            help=f"Remove a lista '{nome_a_carregar}' do ficheiro de listas guardadas.",
        ):
            _lista_guardada_delete(nome_a_carregar)
            # Persistir mensagem para sobreviver ao rerun.
            st.session_state["_lista_msg"] = ("success", f"Lista '{nome_a_carregar}' apagada.")
            st.session_state.pop("_lista_carregada", None)
            st.rerun()

    # Pré-preencher o text_area se o utilizador seleccionou uma lista guardada.
    if nome_a_carregar != "—" and st.session_state.get("_lista_carregada") != nome_a_carregar:
        st.session_state["lista_texto"] = listas_guardadas[nome_a_carregar]
        st.session_state["_lista_carregada"] = nome_a_carregar

    lista_texto = st.text_area(
        "Lista de compras",
        placeholder="arroz agulha\nazeite virgem extra\nleite meio-gordo\natum natural",
        height=180,
        key="lista_texto",
    )

    # ---- Guardar lista atual ----
    with st.expander("Guardar lista atual", expanded=False):
        col_n, col_b = st.columns([3, 1])
        with col_n:
            nome_a_guardar = st.text_input(
                "Nome",
                placeholder="ex: compra semanal",
                key="lista_guardar_nome",
                label_visibility="collapsed",
            )
        with col_b:
            if st.button("Guardar", key="btn_guardar_lista", type="secondary"):
                nome_limpo = nome_a_guardar.strip()
                if not nome_limpo:
                    st.warning("Dá um nome à lista antes de guardar.")
                elif not lista_texto.strip():
                    st.warning("A lista está vazia.")
                else:
                    _lista_guardada_save(nome_limpo, lista_texto)
                    # Persistir mensagem para sobreviver ao rerun.
                    st.session_state["_lista_msg"] = (
                        "success", f"Lista '{nome_limpo}' guardada."
                    )
                    st.rerun()

    if st.button("Otimizar Lista", type="primary", key="btn_otimizar"):
        itens = [i.strip().rstrip(",").strip() for i in lista_texto.strip().splitlines() if i.strip().rstrip(",").strip()]
        if not itens:
            st.warning("Introduz pelo menos um produto na lista.")
        else:
            with st.spinner(f"A otimizar {len(itens)} item(ns) entre lojas…"):
                resultado = api_post("/compras/otimizar", {"lista": itens})

            if resultado is not None:
                if "erro" in resultado:
                    st.error(resultado["erro"])
                else:
                    # Aviso: itens da lista que não foram encontrados na BD.
                    # Sem este aviso, o utilizador pode pensar que a lista está
                    # toda coberta quando na verdade só alguns itens entraram
                    # no cálculo do custo.
                    nao_encontrados = resultado.get("nao_encontrados") or []
                    n_total = len(itens)
                    n_encontrados = n_total - len(nao_encontrados)
                    if nao_encontrados:
                        st.warning(
                            f"**{n_encontrados} de {n_total} itens encontrados.** "
                            f"Os seguintes não têm correspondência na base de dados "
                            f"(o custo apresentado **não os inclui**):\n\n"
                            + "\n".join(f"- {it}" for it in nao_encontrados)
                            + "\n\nTenta reformular (ex: termos mais genéricos, "
                            "verificar acentuação) ou pesquisa-os na tab "
                            "**🔍 Pesquisa & Comparação**."
                        )

                    # Métricas de resumo — mostramos as DUAS estratégias
                    # (dividir vs loja única) e a recomendação. Anteriormente
                    # só aparecia "Custo mínimo" (clampado ao menor das duas)
                    # e a "Poupança ao dividir" estava quase sempre a 0, dando
                    # a impressão de que o cálculo estava partido. Agora cada
                    # estratégia tem o seu valor explícito.
                    melhor_loja = resultado.get("melhor_loja")
                    custo_loja_unica = resultado.get("custo_melhor_loja")
                    custo_dividido = resultado.get("custo_dividido", resultado.get("custo_minimo", 0))
                    melhor_estrategia = resultado.get("melhor_estrategia", "dividir")
                    poupanca = resultado.get("poupanca_split")

                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric(
                        "Custo a dividir",
                        f"{custo_dividido:.2f} €",
                        help=(
                            "Soma do preço mais baixo de cada item em qualquer "
                            "loja. Requer visitar várias cadeias."
                        ),
                    )
                    c2.metric(
                        "Custo loja única",
                        "—" if custo_loja_unica is None else f"{custo_loja_unica:.2f} €",
                        help=(
                            "Custo total comprando tudo na melhor loja que "
                            "tenha a lista completa."
                        ),
                    )
                    # NB: o valor da métrica é mantido curto (🏪/🔀 + nome de
                    # loja apenas, sem prefixos como "Loja única — X") para
                    # evitar truncamento em cadeias de nome longo como "Pingo
                    # Doce" ou "Continente". O contexto vai no label e help.
                    if melhor_estrategia == "dividir":
                        c3.metric(
                            "Estratégia recomendada",
                            "🔀 Dividir",
                            help="Dividir entre lojas (compra em várias cadeias) custa menos que comprar tudo numa só.",
                        )
                        c4.metric(
                            "Poupas se dividires",
                            f"{poupanca:.2f} €",
                            help="Diferença face à loja única.",
                        )
                    elif melhor_estrategia == "loja_unica":
                        c3.metric(
                            "Estratégia recomendada",
                            f"🏪 {melhor_loja}",
                            help=(
                                f"Comprar tudo em **{melhor_loja}** (loja única) "
                                "custa menos (ou igual) a dividir entre cadeias."
                            ),
                        )
                        diff = custo_dividido - (custo_loja_unica or 0)
                        c4.metric(
                            "Poupas vs dividir",
                            f"{diff:.2f} €" if diff > 0.005 else "0.00 €",
                            help=(
                                "Diferença entre comprar dividido vs na loja "
                                "única recomendada. 0€ quando empata."
                            ),
                        )
                    else:  # dividir_parcial — nenhuma loja tem lista completa
                        c3.metric(
                            "Estratégia recomendada",
                            "🔀 Dividir (forçado)",
                            help="Nenhuma loja tem todos os itens — tens de visitar várias.",
                        )
                        c4.metric(
                            "Poupas se dividires",
                            "—",
                            help="Não há loja única para comparar (nenhuma cobre a lista toda).",
                        )

                    # Gráfico de custo por loja
                    if "detalhe_por_loja" in resultado:
                        detalhe = resultado["detalhe_por_loja"]
                        n_itens_total = len(itens)
                        df_lojas = pd.DataFrame([
                            {
                                "Loja":            loja,
                                "Custo (€)":       info["total"],
                                "Itens em falta":  len(info.get("em_falta", [])),
                                "Cobertura":       n_itens_total - len(info.get("em_falta", [])),
                                "Completa":        "Sim" if not info.get("em_falta") else "Não",
                                "Em falta (lista)": ", ".join(info.get("em_falta", [])) or "—",
                            }
                            for loja, info in detalhe.items()
                        ]).sort_values("Custo (€)")

                        # Diferencia visualmente lojas com cobertura parcial — caso
                        # contrário o utilizador vê o custo mais baixo numa loja
                        # incompleta e pensa que é a melhor opção (bug #11).
                        df_lojas["Status"] = df_lojas["Completa"].map(
                            lambda c: "Lista completa" if c == "Sim" else "Cobertura parcial"
                        )
                        df_lojas["text_label"] = df_lojas.apply(
                            lambda r: (f"{r['Custo (€)']:.2f}€"
                                       if r["Completa"] == "Sim"
                                       else f"{r['Custo (€)']:.2f}€  ({r['Cobertura']}/{n_itens_total})"),
                            axis=1,
                        )
                        fig = px.bar(
                            df_lojas,
                            x="Loja",
                            y="Custo (€)",
                            color="Status",
                            text="text_label",
                            title="Custo total da lista por loja",
                            color_discrete_map={
                                "Lista completa":    "#43A047",
                                "Cobertura parcial": "#FB8C00",
                            },
                            hover_data={
                                "Cobertura":       True,
                                "Em falta (lista)": True,
                                "Status":          False,
                                "text_label":      False,
                            },
                        )
                        fig.update_traces(textposition="outside")
                        fig.update_layout(legend=dict(orientation="h", y=1.12))
                        st.plotly_chart(fig, width="stretch")
                        if (df_lojas["Completa"] == "Não").any():
                            st.caption(
                                "⚠️ Barras a **laranja** indicam lojas onde a lista "
                                "**não está completa** — o custo só conta os itens que "
                                "essa loja tem. Não compares diretamente com uma loja completa."
                            )

                    # ---- Tabela: detalhe por item (estratégia DIVIDIR) ----
                    if "itens" in resultado and resultado["itens"]:
                        st.subheader("Detalhe por item — estratégia dividida")
                        st.caption(
                            "Preço mais baixo encontrado para cada item em qualquer cadeia. "
                            "Para ver os preços específicos numa loja única, expande "
                            "**'Detalhe por loja'** abaixo."
                        )
                        df_itens = pd.DataFrame(resultado["itens"])
                        cols_disp = [c for c in [
                            "item_pesquisado", "nome_padronizado", "loja",
                            "preco_atual", "em_promocao", "id_produto_loja",
                        ] if c in df_itens.columns]
                        df_show = df_itens[cols_disp].rename(columns={
                            "item_pesquisado": "Pesquisado",
                            "nome_padronizado": "Produto na BD",
                            "loja": "Loja",
                            "preco_atual": "Preço (€)",
                            "em_promocao": "Promoção",
                            "id_produto_loja": "ID",
                        })
                        if "Preço (€)" in df_show.columns:
                            df_show["Preço (€)"] = df_show["Preço (€)"].apply(_fmt_preco)
                        st.dataframe(df_show, width="stretch", hide_index=True)
                        st.caption(
                            "O **ID** pode ser usado diretamente na aba "
                            "**Previsão de Preços** para ver a evolução futura "
                            "de cada produto."
                        )

                    # ---- Expander: detalhe por loja (estratégia LOJA ÚNICA) ----
                    # Mostra os itens TAL COMO seriam comprados em cada loja.
                    # Útil quando a recomendação é "loja única" e o utilizador
                    # quer saber quais SKUs específicos comprava nessa loja
                    # (que podem ser diferentes do "mais barato global por
                    # item" mostrado na tabela acima).
                    if "detalhe_por_loja" in resultado:
                        detalhe = resultado["detalhe_por_loja"]
                        # Só mostra se alguma loja tem o campo "itens" populado
                        # (compatibilidade com respostas mais antigas que não traziam).
                        tem_itens_por_loja = any(
                            isinstance(info.get("itens"), list) and info["itens"]
                            for info in detalhe.values()
                        )
                        if tem_itens_por_loja:
                            with st.expander(
                                "🏪 Ver detalhe por loja (estratégia loja única)",
                                expanded=False,
                            ):
                                st.caption(
                                    "Como seria a tua compra se fosses tudo a cada uma das "
                                    "lojas. Lojas com lista incompleta mostram apenas os itens "
                                    "que essa cadeia tem."
                                )
                                for loja in sorted(
                                    detalhe.keys(),
                                    key=lambda l: detalhe[l]["total"],
                                ):
                                    info = detalhe[loja]
                                    n_falta = len(info.get("em_falta", []))
                                    if n_falta == 0:
                                        st.markdown(
                                            f"#### ✅ {loja} — **{info['total']:.2f}€** "
                                            "_(lista completa)_"
                                        )
                                    else:
                                        st.markdown(
                                            f"#### ⚠️ {loja} — **{info['total']:.2f}€** "
                                            f"_({n_falta} item(s) em falta)_"
                                        )
                                    itens_loja = info.get("itens") or []
                                    if itens_loja:
                                        df_loja = pd.DataFrame(itens_loja)
                                        cols_l = [c for c in [
                                            "item_pesquisado", "nome_padronizado",
                                            "preco_atual", "em_promocao",
                                        ] if c in df_loja.columns]
                                        df_loja_show = df_loja[cols_l].rename(columns={
                                            "item_pesquisado":  "Pesquisado",
                                            "nome_padronizado": "Produto na BD",
                                            "preco_atual":      "Preço (€)",
                                            "em_promocao":      "Promoção",
                                        })
                                        if "Preço (€)" in df_loja_show.columns:
                                            df_loja_show["Preço (€)"] = (
                                                df_loja_show["Preço (€)"].apply(_fmt_preco)
                                            )
                                        st.dataframe(
                                            df_loja_show, width="stretch",
                                            hide_index=True,
                                        )
                                    if info.get("em_falta"):
                                        st.caption(
                                            "Em falta nesta loja: "
                                            + ", ".join(info["em_falta"])
                                        )


# ===========================================================================
# Tab 2b — Lista de compras com localização GPS (custo de deslocação)
# ===========================================================================

#: Cidades pré-definidas usadas como **fallback** (quando o utilizador não pode/
#: não quer partilhar a localização do browser). Úteis também para demonstração
#: na apresentação académica — permitem testar a feature de qualquer cidade do país.
LOCALIZACOES_PRESET = {
    "Personalizar...":  (None, None),
    "UMinho (Gualtar)": (41.5610, -8.3970),
    "Braga (centro)":   (41.5454, -8.4265),
    "Porto (Boavista)": (41.1579, -8.6291),
    "Lisboa (Marquês)": (38.7259, -9.1500),
    "Coimbra (centro)": (40.2056, -8.4196),
    "Faro (centro)":    (37.0179, -7.9304),
}

#: Caminho do ficheiro JSON com a última localização do utilizador.
#:
#: Mesma razão da [[LISTAS_FILE_PATH]]: ``streamlit-local-storage`` 0.0.25 tem
#: race conditions reproduzíveis em F5 — a primeira render após reload devolve
#: ``None`` mesmo quando o valor está no browser, e a localização "desaparece".
#: Persistência server-side resolve isso de forma determinística.
LOCALIZACAO_FILE_PATH = Path(__file__).parent / "data" / "user_data" / "localizacao.json"


def _persistir_localizacao(lat: float, lon: float, fonte: str) -> None:
    """Guarda a localização em ficheiro JSON server-side.

    Persistência entre sessões do dashboard — o utilizador não precisa de
    autorizar o GPS ou seleccionar a cidade outra vez quando volta.
    """
    LOCALIZACAO_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {"lat": float(lat), "lon": float(lon), "fonte": fonte}
    with open(LOCALIZACAO_FILE_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _carregar_localizacao_persistida() -> tuple[float, float, str] | None:
    """Lê a localização guardada no ficheiro. ``None`` se ausente/inválida."""
    if not LOCALIZACAO_FILE_PATH.exists():
        return None
    try:
        with open(LOCALIZACAO_FILE_PATH, encoding="utf-8") as f:
            d = json.load(f)
        return float(d["lat"]), float(d["lon"]), str(d.get("fonte", "manual"))
    except (OSError, KeyError, ValueError, TypeError, json.JSONDecodeError):
        return None


def _apagar_localizacao_persistida() -> None:
    """Remove o ficheiro de localização e o estado de sessão associado."""
    try:
        LOCALIZACAO_FILE_PATH.unlink(missing_ok=True)
    except OSError:
        pass
    st.session_state.pop("geo_manual_loc", None)


with tab_geo:
    st.subheader("Lista de Compras com Custo de Deslocação")
    st.caption(
        "Otimiza a lista de compras tendo em conta a tua localização GPS e o "
        "custo de deslocação ao supermercado. Para cada cadeia, calculamos "
        "**custo total = preço dos produtos + deslocação ida-volta** à loja "
        "física mais próxima."
    )

    # =======================================================================
    # Localização do utilizador
    # =======================================================================
    # Estratégia em camadas:
    # 1. Default: pedido de geolocation ao browser via streamlit-geolocation
    #    (usa navigator.geolocation.getCurrentPosition, com popup nativo de
    #    permissão). Funciona em localhost mesmo sem HTTPS.
    # 2. Fallback: expander "Definir manualmente" com lat/lon inputs e
    #    atalhos para cidades portuguesas, para casos em que o utilizador
    #    nega permissão, está num browser sem suporte, ou para demonstração.
    # 3. Persistência: a última localização confirmada é guardada num ficheiro
    #    JSON server-side (`data/user_data/localizacao.json`), sendo restaurada
    #    automaticamente quando o utilizador volta ao dashboard (sem precisar de
    #    re-autorizar GPS ou re-seleccionar cidade). Ficheiro em vez de
    #    ``localStorage`` porque ``streamlit-local-storage`` 0.0.25 tem race
    #    conditions na 1ª render após F5 — ver [[LOCALIZACAO_FILE_PATH]].
    # ---------------------------------------------------------------------

    # Carregar localização persistida (se existir, hidrata session_state na
    # primeira renderização da página).
    if "geo_loc_restaurada" not in st.session_state:
        st.session_state["geo_loc_restaurada"] = True
        persistida = _carregar_localizacao_persistida()
        if persistida is not None:
            lat_p, lon_p, fonte_p = persistida
            st.session_state["geo_manual_loc"] = (lat_p, lon_p)
            st.session_state["geo_fonte_persistida"] = fonte_p

    st.markdown("### 📍 A tua localização")
    col_btn, col_status = st.columns([1, 3])

    with col_btn:
        # O componente renderiza um botão; quando clicado pede ao browser a
        # localização. Devolve um dict com latitude/longitude (ou None enquanto
        # não houver permissão).
        loc_browser = streamlit_geolocation()

    # Determinar a localização activa, com prioridade:
    #   (1) GPS "fresco" do browser nesta render  >  (2) Override manual/preset
    #   guardado em session_state  >  (3) Nenhuma
    #
    # O GPS do browser tem prioridade porque clicar no botão "📍" é uma intenção
    # explícita do utilizador de usar a localização real — caso contrário, depois
    # de uma vez aplicar manual/preset, o botão GPS ficaria inerte (bug observado
    # em testes II3/II4 onde o clique no botão não tinha efeito após Vila Real
    # ter sido aplicada manualmente e persistida).
    manual_loc = st.session_state.get("geo_manual_loc")    # tuplo (lat, lon) ou None
    fonte_persistida = st.session_state.get("geo_fonte_persistida")

    gps_lat, gps_lon = None, None
    if isinstance(loc_browser, dict):
        gps_lat = loc_browser.get("latitude")
        gps_lon = loc_browser.get("longitude")

    if gps_lat is not None and gps_lon is not None:
        user_lat, user_lon = float(gps_lat), float(gps_lon)
        fonte = "browser"
        # GPS do browser é "frescamente obtido" — persistir para restaurar mais
        # tarde mesmo sem o utilizador clicar de novo. Limpa também o override
        # manual/preset para que o GPS não fique enterrado em renders seguintes.
        _persistir_localizacao(user_lat, user_lon, fonte="browser")
        st.session_state["geo_manual_loc"] = (user_lat, user_lon)
        st.session_state["geo_fonte_persistida"] = "browser"
    elif manual_loc is not None:
        user_lat, user_lon = manual_loc
        fonte = fonte_persistida if fonte_persistida else "manual"
    else:
        user_lat, user_lon = None, None
        fonte = None

    with col_status:
        if fonte == "browser":
            st.success(
                f"📡 **Localização obtida do browser**: "
                f"`{user_lat:.4f}, {user_lon:.4f}` "
                "_(guardada em ficheiro — restaura-se automaticamente após F5)_"
            )
        elif fonte in ("manual", "preset"):
            origem = "manual" if fonte == "manual" else "preset"
            st.info(
                f"🛠️ **Localização {origem} aplicada**: "
                f"`{user_lat:.4f}, {user_lon:.4f}` "
                "_(persiste entre sessões em ficheiro server-side)_"
            )
        else:
            st.warning(
                "Clica no botão à esquerda para o browser pedir a tua localização, "
                "ou usa o **expander abaixo** para definir manualmente."
            )

    # ---- Fallback manual ----
    with st.expander("🛠️ Definir localização manualmente (fallback)", expanded=False):
        st.caption(
            "Útil quando o navegador não tem permissão de geolocalização ou para "
            "demonstrar a feature noutra cidade. **Aplicar localização manual** "
            "sobrepõe-se à localização do browser e fica guardada entre sessões."
        )

        col_preset_loc, col_manual_lat, col_manual_lon = st.columns([2, 1, 1])

        with col_preset_loc:
            local_preset_sel = st.selectbox(
                "Cidade",
                list(LOCALIZACOES_PRESET.keys()),
                index=1,    # UMinho como default
                key="geo_local_preset_fallback",
            )
        preset_lat, preset_lon = LOCALIZACOES_PRESET[local_preset_sel]

        # Quando o preset muda, sincronizar os inputs manuais com as coordenadas
        # do preset (caso contrário ficam congelados nos valores anteriores e a UI
        # mente — mostra Braga enquanto o preset selecionado é Lisboa).
        prev_preset = st.session_state.get("_geo_prev_preset")
        if prev_preset != local_preset_sel and preset_lat is not None:
            st.session_state["geo_lat_manual"] = float(preset_lat)
            st.session_state["geo_lon_manual"] = float(preset_lon)
        st.session_state["_geo_prev_preset"] = local_preset_sel
        # Default inicial (1ª render): UMinho — só usado se ainda não existir no estado.
        st.session_state.setdefault("geo_lat_manual", float(preset_lat if preset_lat is not None else 41.5610))
        st.session_state.setdefault("geo_lon_manual", float(preset_lon if preset_lon is not None else -8.3970))

        with col_manual_lat:
            manual_lat_input = st.number_input(
                "Latitude (WGS84)",
                min_value=-90.0, max_value=90.0,
                step=0.0001, format="%.4f",
                disabled=(preset_lat is not None),
                key="geo_lat_manual",
            )
        with col_manual_lon:
            manual_lon_input = st.number_input(
                "Longitude (WGS84)",
                min_value=-180.0, max_value=180.0,
                step=0.0001, format="%.4f",
                disabled=(preset_lon is not None),
                key="geo_lon_manual",
            )

        # Se um preset estiver selecionado, sobrepor os inputs
        if preset_lat is not None:
            efetivo_lat, efetivo_lon = preset_lat, preset_lon
            fonte_a_guardar = "preset"
        else:
            efetivo_lat, efetivo_lon = manual_lat_input, manual_lon_input
            fonte_a_guardar = "manual"

        c_apply, c_clear = st.columns(2)
        with c_apply:
            if st.button("Aplicar e guardar localização", key="geo_apply_manual"):
                st.session_state["geo_manual_loc"] = (efetivo_lat, efetivo_lon)
                st.session_state["geo_fonte_persistida"] = fonte_a_guardar
                _persistir_localizacao(efetivo_lat, efetivo_lon, fonte_a_guardar)
                st.rerun()
        with c_clear:
            if manual_loc is not None and st.button(
                "🗑️ Limpar localização guardada", key="geo_clear_manual",
                help="Remove a localização guardada em ficheiro e volta ao GPS do browser.",
            ):
                _apagar_localizacao_persistida()
                st.session_state.pop("geo_fonte_persistida", None)
                st.rerun()

    # ---- Custo €/km (presets via API) ----
    st.divider()
    col_preset, col_raio = st.columns([2, 1])

    with col_preset:
        # Buscar presets à API uma única vez por sessão
        presets_data = api_get("/custo-deslocacao/presets", silent=True) or {}

        if presets_data:
            opcoes_preset = {
                f"{info['label']} ({info['valor']:.2f}€/km) — {info['descricao']}": key
                for key, info in presets_data.items()
            }
        else:
            # Fallback se a API estiver indisponível: presets hardcoded
            opcoes_preset = {
                "Equilibrado (0.20€/km) — combustível + manutenção": "equilibrado",
                "Só combustível (0.12€/km)": "so_combustivel",
                "Tarifa AT (0.36€/km) — todos os custos": "tarifa_at",
            }
        opcoes_preset["Personalizado..."] = "_custom"

        preset_label = st.selectbox(
            "Custo de deslocação (€/km)",
            list(opcoes_preset.keys()),
            index=0,
            key="geo_preset_label",
            help=(
                "Custo de circular um km. Os valores oficiais são:\n"
                "- **0.12 €/km** — só combustível (gasolina ~1.80€/L, 6 L/100km)\n"
                "- **0.20 €/km** — equilibrado (combustível + manutenção)\n"
                "- **0.36 €/km** — tarifa AT (Portaria 1553-D/2008)"
            ),
        )
        custo_km_key = opcoes_preset[preset_label]

        if custo_km_key == "_custom":
            custo_km_valor = st.number_input(
                "Custo €/km personalizado",
                # min=0.0 permite testar o cenário "deslocação grátis" (ex: bike,
                # carro de empresa). max=5.0 cobre extremos plausíveis (táxi com
                # bagagem ronda 1–2 €/km; 5 €/km cobre transporte de mudanças).
                value=0.20, min_value=0.0, max_value=5.0, step=0.01, format="%.2f",
                key="geo_custom_km",
                help="0.00 = sem custo de deslocação; 5.00 = limite superior plausível.",
            )
            custo_km_param = custo_km_valor
        else:
            custo_km_param = custo_km_key

    with col_raio:
        raio_km = st.number_input(
            "Raio máximo (km)",
            value=30, min_value=1, max_value=200, step=5,
            key="geo_raio",
            help="Cadeias sem loja física no raio são consideradas inalcançáveis.",
        )

    # ---- Lista de compras ----
    st.divider()
    lista_texto_geo = st.text_area(
        "Lista de compras",
        placeholder="arroz agulha\nazeite virgem extra\nleite meio-gordo\natum natural",
        height=180,
        key="geo_lista_texto",
    )

    multi_loja = st.checkbox(
        "🔀 Considerar dividir a lista entre 2 cadeias (rota multi-loja)",
        value=False,
        key="geo_multi_loja",
        help="Avalia se vale a pena ir a 2 supermercados na mesma viagem "
             "(rota triangular: casa → loja A → loja B → casa). Recomenda "
             "split apenas se for estritamente mais barato que comprar tudo numa loja.",
    )

    if st.button("Otimizar com Localização", type="primary", key="geo_btn_otimizar",
                  disabled=(user_lat is None or user_lon is None)):
        itens = [i.strip().rstrip(",").strip()
                 for i in lista_texto_geo.strip().splitlines()
                 if i.strip().rstrip(",").strip()]
        if not itens:
            st.warning("Introduz pelo menos um produto na lista.")
        elif user_lat is None or user_lon is None:
            # Defensivo — o botão está disabled mas se acontecer:
            st.warning("Define primeiro a tua localização (botão GPS ou expander manual).")
        else:
            spinner_label = (
                f"A otimizar {len(itens)} item(ns)"
                f"{' com avaliação multi-loja' if multi_loja else ''}…"
            )
            with st.spinner(spinner_label):
                resultado = api_post(
                    "/compras/otimizar-geo",
                    {"lista": itens, "lat": user_lat, "lon": user_lon,
                     "custo_km": custo_km_param, "raio_km": float(raio_km),
                     "multi_loja": multi_loja},
                )

            if resultado is not None:
                if "erro" in resultado:
                    st.error(resultado["erro"])
                else:
                    melhor = resultado.get("melhor_opcao")
                    custo_km_efetivo = resultado.get("custo_km", 0.20)

                    # Aviso de itens não encontrados na BD (não entram no cálculo).
                    nao_encontrados = resultado.get("nao_encontrados") or []
                    n_total = len(itens)
                    n_encontrados = n_total - len(nao_encontrados)
                    if nao_encontrados:
                        st.warning(
                            f"**{n_encontrados} de {n_total} itens encontrados.** "
                            f"Os seguintes não têm correspondência na base de dados "
                            f"(não foram considerados no cálculo):\n\n"
                            + "\n".join(f"- {it}" for it in nao_encontrados)
                        )

                    # ---- Resumo da recomendação ----
                    if melhor is None:
                        # Distinguir as 3 razões pelas quais não há recomendação:
                        #   (a) cadeias têm loja no raio mas a lista nunca está
                        #       100% coberta (itens não encontrados na BD).
                        #   (b) há cadeias com lista completa mas nenhuma tem
                        #       loja física no raio.
                        #   (c) ambos.
                        # Texto específico evita o aviso confuso visto no III1-bis,
                        # em que ✓ aparecia em todas as cadeias mas a mensagem
                        # sugeria "Tenta um raio maior" — quando o raio não era o
                        # problema.
                        detalhe_diag = resultado.get("detalhe_por_cadeia", {}) or {}
                        algum_alcancavel = any(
                            d.get("alcancavel") for d in detalhe_diag.values()
                        )
                        if nao_encontrados and algum_alcancavel:
                            st.warning(
                                "**Nenhuma cadeia tem a lista completa** — alguns itens "
                                "não foram encontrados na base de dados (apenas "
                                "alimentos/bebidas). A tabela em baixo mostra o custo "
                                "dos itens encontrados em cada cadeia."
                            )
                        elif not algum_alcancavel:
                            st.warning(
                                f"**Nenhuma cadeia tem loja física no raio de "
                                f"{raio_km} km** da tua localização. Tenta um raio maior."
                            )
                        else:
                            st.warning(
                                "**Nenhuma cadeia** tem a lista completa e loja física "
                                "no raio. Tenta um raio maior ou rever os itens em falta."
                            )
                    else:
                        st.success(
                            f"**Recomendação: {melhor['insignia']}** — "
                            f"custo total **{melhor['custo_total']:.2f}€** "
                            f"({melhor['custo_produtos']:.2f}€ produtos + "
                            f"{melhor['custo_deslocacao']:.2f}€ deslocação a "
                            f"{melhor['distancia_km']:.1f} km)"
                        )

                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("Custo Total", f"{melhor['custo_total']:.2f} €")
                        c2.metric("Produtos", f"{melhor['custo_produtos']:.2f} €")
                        c3.metric("Deslocação", f"{melhor['custo_deslocacao']:.2f} €",
                                  help=f"{melhor['distancia_km']:.1f} km × 2 (ida-volta) × {custo_km_efetivo}€/km")
                        c4.metric("Distância", f"{melhor['distancia_km']:.1f} km")

                        st.info(
                            f"**Loja recomendada:** {_formatar_nome_loja(melhor['loja_fisica']['nome_loja'])}  \n"
                            f"{_formatar_nome_loja(melhor['loja_fisica'].get('morada') or '')} — "
                            f"{_formatar_nome_loja(melhor['loja_fisica'].get('cidade') or '')}"
                        )

                    # ---- Análise multi-loja (se pedida e relevante) ----
                    melhor_par = resultado.get("melhor_par")
                    recomendacao = resultado.get("recomendacao", "single")
                    poupanca_par = resultado.get("poupanca_par", 0.0)

                    if multi_loja:
                        st.subheader("🔀 Análise multi-loja")

                        if melhor_par is None:
                            st.caption(
                                "Não há combinação viável de 2 cadeias para esta lista "
                                "(faltam itens em todos os pares possíveis)."
                            )
                        elif recomendacao == "single":
                            # Bug #16: empate (delta < 0.01€) não é "single ganha",
                            # é mesmo empate — mostrar com mensagem específica.
                            delta = melhor_par["custo_total"] - melhor["custo_total"]
                            if abs(delta) < 0.01:
                                st.info(
                                    f"**Empate** — dividir entre "
                                    f"{' + '.join(melhor_par['cadeias'])} dá o mesmo total "
                                    f"que comprar tudo no {melhor['insignia']} "
                                    f"({melhor['custo_total']:.2f}€). "
                                    "Recomendado o single-store por simplicidade."
                                )
                            else:
                                st.info(
                                    f"**Single-store continua melhor.** Dividir entre "
                                    f"{' + '.join(melhor_par['cadeias'])} custaria "
                                    f"**{melhor_par['custo_total']:.2f}€** (vs. "
                                    f"{melhor['custo_total']:.2f}€ no {melhor['insignia']})."
                                )
                            with st.expander("Ver detalhes do melhor par avaliado"):
                                n_a = len(melhor_par["itens_em_a"])
                                n_b = len(melhor_par["itens_em_b"])
                                st.write(
                                    f"- **{melhor_par['cadeias'][0]}** "
                                    f"({n_a} {'item' if n_a == 1 else 'itens'}, "
                                    f"{melhor_par['custo_produtos_a']:.2f}€)\n"
                                    f"- **{melhor_par['cadeias'][1]}** "
                                    f"({n_b} {'item' if n_b == 1 else 'itens'}, "
                                    f"{melhor_par['custo_produtos_b']:.2f}€)\n"
                                    f"- Rota: {melhor_par['distancia_total_km']:.1f} km "
                                    f"→ {melhor_par['custo_deslocacao']:.2f}€"
                                )
                        else:  # recomendacao == "par"
                            st.success(
                                f"**Recomendado dividir!** "
                                f"{' + '.join(melhor_par['cadeias'])} = "
                                f"**{melhor_par['custo_total']:.2f}€** "
                                f"(poupas {poupanca_par:.2f}€ vs. single-store)."
                            )
                            colp1, colp2, colp3 = st.columns(3)
                            colp1.metric(
                                f"{melhor_par['cadeias'][0]} ({len(melhor_par['itens_em_a'])} itens)",
                                f"{melhor_par['custo_produtos_a']:.2f} €",
                            )
                            colp2.metric(
                                f"{melhor_par['cadeias'][1]} ({len(melhor_par['itens_em_b'])} itens)",
                                f"{melhor_par['custo_produtos_b']:.2f} €",
                            )
                            colp3.metric(
                                "Deslocação (rota triangular)",
                                f"{melhor_par['custo_deslocacao']:.2f} €",
                                help=(
                                    f"Rota: tu → {melhor_par['cadeias'][0]} "
                                    f"({melhor_par['distancia_user_a_km']:.1f} km) → "
                                    f"{melhor_par['cadeias'][1]} "
                                    f"({melhor_par['distancia_a_b_km']:.1f} km) → tu "
                                    f"({melhor_par['distancia_b_user_km']:.1f} km). "
                                    f"Total: {melhor_par['distancia_total_km']:.1f} km."
                                ),
                            )

                            with st.expander(f"Itens em {melhor_par['cadeias'][0]} (mais barato aqui)"):
                                df_a = pd.DataFrame(melhor_par["itens_em_a"])[
                                    ["item_pesquisado", "nome_padronizado", "preco_atual"]
                                ].rename(columns={
                                    "item_pesquisado": "Pesquisado",
                                    "nome_padronizado": "Produto",
                                    "preco_atual": "Preço (€)",
                                })
                                st.dataframe(df_a, width="stretch", hide_index=True)

                            with st.expander(f"Itens em {melhor_par['cadeias'][1]} (mais barato aqui)"):
                                df_b = pd.DataFrame(melhor_par["itens_em_b"])[
                                    ["item_pesquisado", "nome_padronizado", "preco_atual"]
                                ].rename(columns={
                                    "item_pesquisado": "Pesquisado",
                                    "nome_padronizado": "Produto",
                                    "preco_atual": "Preço (€)",
                                })
                                st.dataframe(df_b, width="stretch", hide_index=True)

                        # Tabela de todos os pares avaliados
                        todos_pares = resultado.get("todos_os_pares", [])
                        if todos_pares:
                            with st.expander(f"Todos os {len(todos_pares)} pares avaliados"):
                                df_pares = pd.DataFrame([
                                    {
                                        "Pares":            " + ".join(p["cadeias"]),
                                        "Produtos (€)":     p["custo_produtos"],
                                        "Distância (km)":   p["distancia_total_km"],
                                        "Deslocação (€)":   p["custo_deslocacao"],
                                        "Total (€)":        p["custo_total"],
                                    }
                                    for p in todos_pares
                                ])
                                st.dataframe(df_pares, width="stretch", hide_index=True)

                    # ---- Comparação por cadeia ----
                    detalhe = resultado.get("detalhe_por_cadeia", {})
                    if detalhe:
                        rows = []
                        for ins, d in detalhe.items():
                            em_falta_n = len(d.get("em_falta", []) or [])
                            alcancavel = bool(d.get("alcancavel"))

                            # Caso ideal: lista completa + loja física alcançável.
                            if alcancavel and em_falta_n == 0:
                                rows.append({
                                    "Cadeia":           ins,
                                    "Loja no raio":     "✓",
                                    "Produtos (€)":     d["custo_produtos"],
                                    "Distância (km)":   d["distancia_km"],
                                    "Deslocação (€)":   d["custo_deslocacao"],
                                    "Total (€)":        d["custo_total"],
                                    "Nota":             "—",
                                })
                            else:
                                # Bug #14: separar visualmente "sem loja física no
                                # raio" de "lista incompleta" e combinar quando
                                # ambos os problemas se aplicam.
                                avisos: list[str] = []
                                if em_falta_n:
                                    avisos.append(
                                        f"lista incompleta ({em_falta_n} item(s) em falta)"
                                    )
                                if not alcancavel:
                                    avisos.append(f"sem loja física no raio {raio_km} km")
                                # Se a cadeia é alcançável, o backend já calculou
                                # ``custo_total`` (produtos parciais + deslocação) —
                                # mostrar mesmo com lista incompleta, em vez de
                                # ``None``, para a tabela ficar consistente com a
                                # recomendação principal. Só fica ``None`` quando
                                # não há loja no raio (sem deslocação para calcular).
                                rows.append({
                                    "Cadeia":           ins,
                                    "Loja no raio":     "✓" if alcancavel else "✗",
                                    "Produtos (€)":     d["custo_produtos"],
                                    "Distância (km)":   d.get("distancia_km"),
                                    "Deslocação (€)":   d.get("custo_deslocacao"),
                                    "Total (€)":        d.get("custo_total"),
                                    "Nota":             " · ".join(avisos),
                                })

                        df_cmp = pd.DataFrame(rows)
                        # Ordenar: primeiro os alcançáveis por custo total ascendente, depois os outros
                        df_cmp = df_cmp.sort_values(
                            "Total (€)", na_position="last",
                        ).reset_index(drop=True)
                        st.subheader("Comparação por cadeia")
                        # ``column_config`` força 2 decimais em todas as colunas
                        # numéricas. Sem isto, o Streamlit remove zeros à direita
                        # e fica "3" em vez de "3.00" — inconsistente com "0.07"
                        # noutras linhas (bug observado no teste IV4 com 5€/km).
                        st.dataframe(
                            df_cmp, width="stretch", hide_index=True,
                            column_config={
                                "Produtos (€)":   st.column_config.NumberColumn(format="%.2f"),
                                "Distância (km)": st.column_config.NumberColumn(format="%.2f"),
                                "Deslocação (€)": st.column_config.NumberColumn(format="%.2f"),
                                "Total (€)":      st.column_config.NumberColumn(format="%.2f"),
                            },
                        )

                    # ---- Mapa: utilizador + lojas físicas próximas ----
                    st.subheader("Mapa")
                    lojas_proximas_resp = api_get(
                        "/lojas-fisicas/proximas",
                        {"lat": user_lat, "lon": user_lon,
                         "raio_km": min(raio_km, 50), "limite": 30},
                        silent=True,
                    )
                    if lojas_proximas_resp:
                        df_map = pd.DataFrame(lojas_proximas_resp)
                        # Normalizar nomes em CAPS LOCK (~265 lojas PD) antes
                        # de renderizar nos tooltips do mapa.
                        if "nome_loja" in df_map.columns:
                            df_map["nome_loja"] = df_map["nome_loja"].map(_formatar_nome_loja)
                        # Adicionar marcador do utilizador
                        df_user = pd.DataFrame([{
                            "latitude": user_lat, "longitude": user_lon,
                            "nome_loja": "Tu", "insignia": "Utilizador",
                            "distancia_km": 0.0,
                        }])
                        df_full = pd.concat([df_user, df_map], ignore_index=True)

                        fig = px.scatter_map(
                            df_full,
                            lat="latitude", lon="longitude",
                            color="insignia",
                            hover_name="nome_loja",
                            hover_data={
                                "distancia_km": ":.2f",
                                "latitude": False, "longitude": False,
                                "insignia": False,
                            },
                            zoom=11,
                            height=500,
                            color_discrete_map={
                                "Utilizador":  "#E91E63",
                                "Continente":  "#E53935",
                                "Pingo Doce":  "#43A047",
                                "Auchan":      "#FB8C00",
                            },
                        )
                        fig.update_traces(marker={"size": 14})
                        fig.update_layout(
                            map_style="open-street-map",
                            margin=dict(l=0, r=0, t=0, b=0),
                            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
                        )
                        st.plotly_chart(fig, width="stretch")
                    else:
                        st.caption("(Mapa indisponível — verifica que o sistema multiagente está a correr com LocationAgent ativo.)")


# ===========================================================================
# Tab 3 — Previsão de preços (LSTM + Monte Carlo)
# ===========================================================================

with tab_previsao:
    st.subheader("Previsão de Preços (LSTM + Monte Carlo Dropout)")
    st.caption(
        "Introduz o nome de um produto — o sistema resolve o ID automaticamente "
        "e gera previsões LSTM para os próximos dias, com ou sem intervalos de "
        "confiança (Monte Carlo Dropout)."
    )

    # ---- Explicabilidade do modelo (feature ablation) ----
    _ablacao_path = Path("data/generated/feature_ablation.json")
    if _ablacao_path.exists():
        with st.expander("Explicabilidade do modelo · importância das features", expanded=False):
            with open(_ablacao_path, encoding="utf-8") as _f:
                _abl = json.load(_f)
            st.caption(
                "Impacto de cada feature de input na qualidade da previsão. Para cada uma das "
                f"{_abl['metadata']['n_features']} features, **zeramos** essa feature no conjunto de "
                "validação e medimos o aumento de RMSE. Quanto maior o Δ, mais o modelo depende "
                "dessa feature."
            )
            st.markdown(
                f"**Baseline (sem ablação)**: RMSE médio = "
                f"`{_abl['baseline']['rmse_euros']:.4f}€` em "
                f"{_abl['metadata']['n_produtos']} produtos · "
                f"gerado em {_abl['metadata']['gerado_em']}"
            )

            df_abl = pd.DataFrame(_abl["ablacao"])
            df_abl["cor"] = df_abl["delta_rmse"].apply(
                lambda x: "Importante" if x > 0 else "Sem impacto / ruído"
            )
            fig_abl = px.bar(
                df_abl, x="feature", y="delta_pct", color="cor",
                title="Aumento de RMSE (%) ao zerar cada feature",
                color_discrete_map={"Importante": "#1f77b4", "Sem impacto / ruído": "#aaaaaa"},
                labels={"delta_pct": "Δ RMSE (%)", "feature": "Feature"},
            )
            fig_abl.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_abl, width="stretch")

            st.dataframe(
                df_abl[["feature", "rmse_euros", "delta_rmse", "delta_pct"]].rename(columns={
                    "feature":   "Feature",
                    "rmse_euros": "RMSE com ablação (€)",
                    "delta_rmse": "Δ RMSE (€)",
                    "delta_pct":  "Δ RMSE (%)",
                }),
                width="stretch", hide_index=True,
            )
            st.caption(
                "💡 Features com Δ negativo são fontes de **ruído** — o modelo é mais "
                "preciso sem elas. Em retreino futuro, considera removê-las do `FEATURE_COLS`."
            )

    col_p, col_h, col_mc = st.columns([3, 1, 1])
    with col_p:
        prod_prev = st.text_input(
            "Nome do produto",
            placeholder="ex: arroz agulha, azeite virgem extra…",
            key="prev_produto",
        )
    with col_h:
        horizonte = st.number_input(
            "Horizonte (dias)", min_value=1, max_value=30, value=7, key="prev_horizonte"
        )
    with col_mc:
        amostras = st.number_input(
            "Amostras MC", min_value=10, max_value=200, value=50, key="prev_amostras"
        )

    # Dropdown de desambiguação: "arroz agulha" resolve para dezenas de variantes
    # em 3 cadeias. Em vez de escolher uma automaticamente, mostramos os candidatos
    # (nome + loja + preço) e deixamos o utilizador escolher explicitamente — incluindo
    # de que loja quer a previsão.
    candidatos_prev = (
        _ids_candidatos_para_nome(prod_prev, top_n=10, silent=True)
        if prod_prev.strip() else []
    )
    produto_sel = None
    if candidatos_prev:
        idx_sel = st.selectbox(
            "Produto a analisar",
            range(len(candidatos_prev)),
            format_func=lambda i: _label_candidato(candidatos_prev[i]),
            key="prev_produto_sel",
        )
        produto_sel = candidatos_prev[idx_sel]

    col_b1, col_b2, col_b3 = st.columns(3)
    btn_det = col_b1.button("Previsão LSTM", type="primary", key="btn_lstm")
    btn_mc = col_b2.button("Previsão com Incerteza (MC)", key="btn_mc")
    btn_momento = col_b3.button("Comprar agora ou esperar?", key="btn_momento")

    if btn_det or btn_mc or btn_momento:
        if not prod_prev.strip():
            st.warning("Introduz o nome do produto.")
        else:
            if produto_sel is None:
                st.error(
                    f"Produto '{prod_prev}' não encontrado. "
                    "Tenta um termo mais genérico ou verifica a ortografia."
                )
            else:
                produto_id = produto_sel["id"]
                produto_nome = (
                    f"{produto_sel['nome']} ({produto_sel['loja']})"
                    if produto_sel["loja"] else produto_sel["nome"]
                )
                # Confirma que o produto escolhido tem modelo LSTM treinado.
                probe = api_get(f"/previsao/{produto_id}", {"horizonte": 1}, silent=True)
                if probe is None:
                    st.warning(
                        f"O produto **{produto_nome}** não tem histórico suficiente "
                        "para previsão LSTM.  \n"
                        "Escolhe outra variante no dropdown acima, ou corre mais scraping "
                        "e retreina o modelo com `python models/price_predictor.py --treinar`."
                    )
                else:
                    st.info(f"Produto resolvido — `id_produto_loja = {produto_id}` — {produto_nome}")

                    # Carrega histórico para contexto no gráfico
                    hist_data = api_get(f"/historico/{produto_id}", {"dias": 60})
                    df_hist: pd.DataFrame | None = None
                    if hist_data and "historico" in hist_data and hist_data["historico"]:
                        df_hist = pd.DataFrame(hist_data["historico"])

                    # ---- Previsão determinista ----
                    if btn_det:
                        with st.spinner("A gerar previsão LSTM…"):
                            previsao = api_get(
                                f"/previsao/{produto_id}",
                                {"horizonte": horizonte},
                            )

                        if previsao:
                            fig = go.Figure()
                            if df_hist is not None:
                                fig.add_trace(go.Scatter(
                                    x=df_hist["data"], y=df_hist["preco"],
                                    mode="lines+markers", name="Histórico",
                                    line=dict(color="#2196F3"), marker=dict(size=4),
                                ))
                            df_prev = pd.DataFrame(previsao)
                            fig.add_trace(go.Scatter(
                                x=df_prev["data"], y=df_prev["preco_previsto"],
                                mode="lines+markers", name="Previsão LSTM",
                                line=dict(color="#FF5722", dash="dash"),
                                marker=dict(size=7, symbol="diamond"),
                            ))
                            fig.update_layout(
                                title=f"Previsão LSTM — {prod_prev} ({horizonte} dias)",
                                xaxis_title="Data", yaxis_title="Preço (€)",
                                hovermode="x unified",
                                legend=dict(orientation="h", y=1.1),
                            )
                            st.plotly_chart(fig, width="stretch")

                            st.dataframe(
                                df_prev.rename(columns={
                                    "data": "Data",
                                    "preco_previsto": "Preço Previsto (€)",
                                }),
                                width="stretch",
                                hide_index=True,
                                column_config={
                                    "Preço Previsto (€)": st.column_config.NumberColumn(format="%.2f"),
                                },
                            )

                    # ---- Previsão com incerteza (Monte Carlo Dropout) ----
                    elif btn_mc:
                        with st.spinner(f"A correr {amostras} simulações Monte Carlo…"):
                            resultado = api_get(
                                f"/previsao/{produto_id}/incerteza",
                                {"horizonte": horizonte, "amostras": amostras},
                            )

                        if resultado and "previsoes" in resultado:
                            df_mc = pd.DataFrame(resultado["previsoes"])
                            preco_atual = resultado.get("preco_atual")

                            fig = go.Figure()

                            if df_hist is not None:
                                fig.add_trace(go.Scatter(
                                    x=df_hist["data"], y=df_hist["preco"],
                                    mode="lines", name="Histórico",
                                    line=dict(color="#2196F3"),
                                ))

                            # Banda de incerteza IC 5%–95%
                            if "ic_95pct" in df_mc.columns and "ic_5pct" in df_mc.columns:
                                fig.add_trace(go.Scatter(
                                    x=list(df_mc["data"]) + list(df_mc["data"])[::-1],
                                    y=list(df_mc["ic_95pct"]) + list(df_mc["ic_5pct"])[::-1],
                                    fill="toself",
                                    fillcolor="rgba(255, 87, 34, 0.12)",
                                    line=dict(color="rgba(0,0,0,0)"),
                                    name="IC 90% (MC Dropout)",
                                ))

                            fig.add_trace(go.Scatter(
                                x=df_mc["data"], y=df_mc["preco_medio"],
                                mode="lines+markers", name="Média prevista",
                                line=dict(color="#FF5722", dash="dash"),
                                marker=dict(size=7, symbol="diamond"),
                            ))

                            if preco_atual:
                                fig.add_hline(
                                    y=preco_atual,
                                    line_dash="dot",
                                    line_color="gray",
                                    annotation_text=f"Atual: {preco_atual:.2f}€",
                                    annotation_position="bottom right",
                                )

                            fig.update_layout(
                                title=(
                                    f"Previsão com incerteza — {prod_prev} "
                                    f"({amostras} simulações MC)"
                                ),
                                xaxis_title="Data", yaxis_title="Preço (€)",
                                hovermode="x unified",
                                legend=dict(orientation="h", y=1.1),
                            )
                            st.plotly_chart(fig, width="stretch")

                            with st.expander("Ver dados de previsão"):
                                st.dataframe(
                                    df_mc.rename(columns={
                                        "data": "Data",
                                        "preco_medio": "Média (€)",
                                        "preco_std": "Desvio Padrão",
                                        "ic_5pct": "IC 5% (€)",
                                        "ic_95pct": "IC 95% (€)",
                                    }),
                                    width="stretch",
                                    hide_index=True,
                                    column_config={
                                        "Média (€)":      st.column_config.NumberColumn(format="%.2f"),
                                        "Desvio Padrão":  st.column_config.NumberColumn(format="%.4f"),
                                        "IC 5% (€)":      st.column_config.NumberColumn(format="%.2f"),
                                        "IC 95% (€)":     st.column_config.NumberColumn(format="%.2f"),
                                    },
                                )

                    # ---- Momento de compra ----
                    elif btn_momento:
                        with st.spinner("A calcular recomendação de compra…"):
                            resultado = api_get(
                                f"/compras/momento/{produto_id}",
                                {"horizonte": horizonte, "amostras": amostras},
                            )

                        if resultado:
                            if "erro" in resultado:
                                st.error(resultado["erro"])
                            else:
                                recomendacao = resultado.get("recomendacao", "")
                                preco_atual = resultado.get("preco_atual", 0)
                                preco_min = resultado.get("preco_minimo_previsto", 0)
                                descida = resultado.get("descida_pct", 0)
                                nome_prod = resultado.get("nome_produto", prod_prev)
                                data_min = resultado.get("data_minimo", "")

                                # Variação assinada entre o mínimo previsto e o preço atual:
                                # quando >0 estamos perante uma SUBIDA prevista (o min do horizonte
                                # já está acima do preço de hoje), pelo que a métrica "Descida esperada 0.0%"
                                # esconde o sinal mais importante. Calculamos a variação real (com sinal)
                                # para conduzir a mensagem e a 3ª métrica.
                                variacao_pct = (
                                    (preco_min - preco_atual) / preco_atual * 100
                                    if preco_atual > 0 else 0.0
                                )
                                ha_subida = variacao_pct > 0.5  # tolerância p/ ruído numérico

                                if recomendacao == "aguardar":
                                    # A previsão pode descer até um mínimo e voltar a subir
                                    # (ex: cereais/ketchup): sem a data do mínimo a mensagem
                                    # "Aguarda!" sugere erradamente esperar de forma indefinida.
                                    quando = f" — mínimo esperado em **{data_min}**" if data_min else ""
                                    st.success(
                                        f"**Aguarda!** O modelo prevê uma descida de "
                                        f"**{descida:.1f}%** nos próximos {horizonte} dias{quando}.  \n"
                                        f"Preço atual: **{preco_atual:.2f}€** → "
                                        f"Mínimo previsto: **{preco_min:.2f}€**  \n"
                                        f"Produto: *{nome_prod}*"
                                    )
                                elif ha_subida:
                                    st.warning(
                                        f"**Compra agora.** O modelo prevê uma **subida de "
                                        f"+{variacao_pct:.1f}%** nos próximos {horizonte} dias — "
                                        f"não compensa esperar.  \n"
                                        f"Preço atual: **{preco_atual:.2f}€** → "
                                        f"Mínimo previsto: **{preco_min:.2f}€**  \n"
                                        f"Produto: *{nome_prod}*"
                                    )
                                else:
                                    st.info(
                                        f"**Compra agora.** A descida prevista "
                                        f"({descida:.1f}%) não justifica esperar.  \n"
                                        f"Preço atual: **{preco_atual:.2f}€**  \n"
                                        f"Produto: *{nome_prod}*"
                                    )

                                c1, c2, c3 = st.columns(3)
                                c1.metric("Preço atual", f"{preco_atual:.2f} €")
                                delta_label = "Mínimo previsto" if preco_min <= preco_atual else "Previsão (subida)"
                                c2.metric(delta_label, f"{preco_min:.2f} €")
                                if ha_subida:
                                    c3.metric("Subida esperada", f"+{variacao_pct:.1f} %")
                                else:
                                    c3.metric("Descida esperada", f"{descida:.1f} %")

                                # Gráfico com histórico + previsão
                                previsoes_raw = resultado.get("previsoes")
                                if previsoes_raw:
                                    try:
                                        df_prev = pd.DataFrame(previsoes_raw)
                                        fig = go.Figure()

                                        if df_hist is not None:
                                            fig.add_trace(go.Scatter(
                                                x=df_hist["data"], y=df_hist["preco"],
                                                mode="lines", name="Histórico",
                                                line=dict(color="#2196F3"),
                                            ))

                                        # Banda IC se disponível
                                        if "ic_95pct" in df_prev.columns and "ic_5pct" in df_prev.columns:
                                            cor_banda = (
                                                "rgba(76,175,80,0.12)"
                                                if recomendacao == "aguardar"
                                                else "rgba(244,67,54,0.12)"
                                            )
                                            fig.add_trace(go.Scatter(
                                                x=list(df_prev["data"]) + list(df_prev["data"])[::-1],
                                                y=list(df_prev["ic_95pct"]) + list(df_prev["ic_5pct"])[::-1],
                                                fill="toself",
                                                fillcolor=cor_banda,
                                                line=dict(color="rgba(0,0,0,0)"),
                                                name="IC 90%",
                                            ))

                                        y_col = (
                                            "preco_medio" if "preco_medio" in df_prev.columns
                                            else "preco_previsto"
                                        )
                                        if y_col in df_prev.columns:
                                            cor_linha = (
                                                "#4CAF50" if recomendacao == "aguardar"
                                                else "#F44336"
                                            )
                                            fig.add_trace(go.Scatter(
                                                x=df_prev["data"], y=df_prev[y_col],
                                                mode="lines+markers", name="Previsão",
                                                line=dict(color=cor_linha, dash="dash"),
                                                marker=dict(size=7, symbol="diamond"),
                                            ))

                                        fig.add_hline(
                                            y=preco_atual,
                                            line_dash="dot",
                                            line_color="gray",
                                            annotation_text=f"Atual: {preco_atual:.2f}€",
                                            annotation_position="bottom right",
                                        )

                                        fig.update_layout(
                                            title=f"Evolução prevista — {nome_prod}",
                                            xaxis_title="Data", yaxis_title="Preço (€)",
                                            hovermode="x unified",
                                            legend=dict(orientation="h", y=1.1),
                                        )
                                        st.plotly_chart(fig, width="stretch")
                                    except Exception as exc:
                                        st.warning(
                                            f"Não foi possível renderizar o gráfico de previsão: {exc}"
                                        )


# ===========================================================================
# Tab 4 — Histórico de preços
# ===========================================================================

with tab_historico:
    st.subheader("Histórico de Preços")
    st.caption(
        "Introduz o nome de um produto para ver a evolução do preço ao longo do tempo. "
        "Os pontos a verde indicam dias com promoção."
    )

    col_p, col_d = st.columns([4, 1])
    with col_p:
        prod_hist = st.text_input(
            "Nome do produto",
            placeholder="ex: arroz agulha, leite meio-gordo…",
            key="hist_produto",
        )
    with col_d:
        dias = st.number_input("Dias", min_value=7, max_value=365, value=30, key="hist_dias")

    # Dropdown de desambiguação (mesmo padrão da Previsão): escolher explicitamente
    # qual variante/loja ver, em vez de resolver uma automaticamente.
    candidatos_hist = (
        _ids_candidatos_para_nome(prod_hist, top_n=10, silent=True)
        if prod_hist.strip() else []
    )
    produto_sel_hist = None
    if candidatos_hist:
        idx_hist = st.selectbox(
            "Produto a analisar",
            range(len(candidatos_hist)),
            format_func=lambda i: _label_candidato(candidatos_hist[i]),
            key="hist_produto_sel",
        )
        produto_sel_hist = candidatos_hist[idx_hist]

    if st.button("Ver Histórico", type="primary", key="btn_historico"):
        if not prod_hist.strip():
            st.warning("Introduz o nome do produto.")
        else:
            if produto_sel_hist is None:
                st.error(f"Produto '{prod_hist}' não encontrado.")
            else:
                produto_id = produto_sel_hist["id"]
                produto_nome = (
                    f"{produto_sel_hist['nome']} ({produto_sel_hist['loja']})"
                    if produto_sel_hist["loja"] else produto_sel_hist["nome"]
                )
                st.info(f"Produto resolvido — `id_produto_loja = {produto_id}` — {produto_nome}")
                with st.spinner("A carregar histórico…"):
                    hist_data = api_get(f"/historico/{produto_id}", {"dias": dias})

                if hist_data and "historico" in hist_data:
                    historico = hist_data["historico"]
                    if not historico:
                        st.info(f"Sem registos de preço nos últimos {dias} dias.")
                    else:
                        df_h = pd.DataFrame(historico)

                        # Métricas
                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("Observações", len(df_h))
                        c2.metric("Preço mínimo", f"{df_h['preco'].min():.2f} €")
                        c3.metric("Preço máximo", f"{df_h['preco'].max():.2f} €")
                        c4.metric("Preço médio", f"{df_h['preco'].mean():.2f} €")

                        # Gráfico temporal
                        em_promo = df_h.get("em_promocao", pd.Series([False] * len(df_h)))
                        cores = ["#4CAF50" if p else "#2196F3" for p in em_promo]
                        simbolos = ["star" if p else "circle" for p in em_promo]

                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=df_h["data"],
                            y=df_h["preco"],
                            mode="lines+markers",
                            name="Preço",
                            line=dict(color="#2196F3", width=2),
                            marker=dict(size=8, color=cores, symbol=simbolos),
                            hovertemplate=(
                                "Data: %{x}<br>Preço: %{y:.2f}€<extra></extra>"
                            ),
                        ))

                        fig.update_layout(
                            title=(
                                f"Histórico de preços — {produto_nome} "
                                f"(últimos {dias} dias)"
                            ),
                            xaxis_title="Data",
                            yaxis_title="Preço (€)",
                            hovermode="x unified",
                        )
                        # Com apenas 1 observação o Plotly auto-escala o eixo X a
                        # microssegundos (mostra "02:18:53.999" em vez de uma data).
                        # Forçamos um range de ±12h em torno do ponto único.
                        if len(df_h) <= 1:
                            ponto = pd.to_datetime(df_h["data"].iloc[0])
                            fig.update_xaxes(
                                range=[ponto - pd.Timedelta(hours=12),
                                       ponto + pd.Timedelta(hours=12)],
                                tickformat="%Y-%m-%d",
                            )
                            st.info(
                                "Apenas **1 observação** disponível — "
                                "histórico insuficiente para mostrar evolução. "
                                "É preciso mais ciclos de scraping para acumular dados."
                            )
                        st.plotly_chart(fig, width="stretch")
                        st.caption(
                            "Pontos a **verde** ★ = produto em promoção nesse dia."
                        )

                        with st.expander("Ver dados detalhados"):
                            st.dataframe(
                                df_h.rename(columns={
                                    "data": "Data",
                                    "preco": "Preço (€)",
                                    "em_promocao": "Promoção",
                                }),
                                width="stretch",
                                hide_index=True,
                                column_config={
                                    "Preço (€)": st.column_config.NumberColumn(format="%.2f"),
                                },
                            )


# ===========================================================================
# Tab 5 — Preços atuais por loja
# ===========================================================================

with tab_loja:
    st.subheader("Preços Atuais por Loja")
    st.caption(
        "Consulta todos os produtos disponíveis numa cadeia de supermercado, "
        "com estatísticas e distribuição de preços."
    )

    col_l, col_f = st.columns([2, 3])
    with col_l:
        loja_sel = st.selectbox("Cadeia de supermercado", LOJAS, key="loja_sel")
    with col_f:
        filtro = st.text_input(
            "Filtrar por nome (opcional)",
            placeholder="ex: arroz, leite…",
            key="loja_filtro",
        )

    col_o, col_lim = st.columns([3, 1])
    with col_o:
        ordenar = st.selectbox(
            "Ordenar por",
            [
                "Nome (A→Z)",
                "Preço (mais barato primeiro)",
                "Preço (mais caro primeiro)",
                "Promoções primeiro",
            ],
            key="loja_ord",
        )
    with col_lim:
        # 50000 cobre confortavelmente a cadeia com mais produtos na BD (Auchan).
        # 5000 (limite anterior) truncava o Auchan, distorcendo % em promoção e
        # outras estatísticas que dependem do total.
        limite_loja = st.number_input(
            "Máx. produtos", min_value=10, max_value=50000, value=200, key="loja_lim"
        )

    if st.button("Carregar Produtos", type="primary", key="btn_loja"):
        with st.spinner(f"A carregar produtos de {loja_sel}…"):
            resultado = api_get(f"/lojas/{loja_sel}/precos")
        if resultado and "produtos" in resultado:
            # Persistir o resultado bruto em session_state. Sem isto, todo o
            # render abaixo ficava dentro do `if st.button(...)`; ao mexer em
            # qualquer widget (ex: o toggle de escala log) o Streamlit faz
            # rerun, o botão devolve False e o output inteiro desaparecia.
            st.session_state["loja_produtos"] = resultado["produtos"]
            st.session_state["loja_carregada"] = loja_sel
        else:
            st.session_state.pop("loja_produtos", None)

    # Render independente do clique: lê os dados já carregados de session_state
    # e reaplica filtro/ordenação/escala log a cada rerun.
    produtos = st.session_state.get("loja_produtos")
    if produtos is not None:
        loja_carregada = st.session_state.get("loja_carregada", loja_sel)
        if not produtos:
            st.info(f"Nenhum produto registado para {loja_carregada}.")
        else:
            df_l = pd.DataFrame(produtos)

            # Filtrar por nome
            if filtro.strip():
                mask = df_l["nome"].str.contains(filtro.strip(), case=False, na=False)
                df_l = df_l[mask]

            if df_l.empty:
                # Filtro sem correspondências: sem esta guarda, .mean()/.min()
                # sobre o dataframe vazio produziam "nan €" nas métricas.
                st.info(
                    f"Nenhum produto de **{loja_carregada}** corresponde ao filtro "
                    f"«{filtro.strip()}». Tenta outro termo."
                )
            else:
                # Ordenar
                if ordenar == "Nome (A→Z)":
                    df_l = df_l.sort_values("nome")
                elif ordenar == "Preço (mais barato primeiro)":
                    df_l = df_l.sort_values("preco")
                elif ordenar == "Preço (mais caro primeiro)":
                    df_l = df_l.sort_values("preco", ascending=False)
                elif ordenar == "Promoções primeiro":
                    df_l = df_l.sort_values("em_promocao", ascending=False)

                # Métricas e histograma calculados sobre o set filtrado completo,
                # não sobre o head — caso contrário a % em promoção fica distorcida
                # quando o utilizador limita a amostra para fins de display.
                total_filtrado = len(df_l)
                df_l_display = df_l.head(limite_loja)

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Produtos", total_filtrado)
                em_promo_total = int(df_l["em_promocao"].sum())
                pct_promo = (em_promo_total / total_filtrado * 100) if total_filtrado else 0
                c2.metric(
                    "Em promoção",
                    f"{em_promo_total} ({pct_promo:.1f}%)",
                )
                c3.metric("Preço médio", f"{df_l['preco'].mean():.2f} €")
                c4.metric("Preço mais baixo", f"{df_l['preco'].min():.2f} €")

                if len(df_l_display) < total_filtrado:
                    st.caption(
                        f"A tabela mostra os primeiros {len(df_l_display)} de "
                        f"{total_filtrado} produtos (métricas e histograma "
                        f"calculados sobre o total)."
                    )

                # Distribuição de preços — pode ser muito assimétrica com outliers
                escala_log = st.toggle(
                    "Escala log no eixo Y (revela cauda longa)",
                    value=False,
                    key="loja_hist_log",
                    help="Útil quando há outliers de preço a dominar o histograma.",
                )
                fig = px.histogram(
                    df_l,
                    x="preco",
                    nbins=40,
                    title=f"Distribuição de preços — {loja_carregada}",
                    labels={"preco": "Preço (€)", "count": "Nº de produtos"},
                    color_discrete_sequence=["#2196F3"],
                    log_y=escala_log,
                )
                fig.update_layout(bargap=0.05)
                st.plotly_chart(fig, width="stretch")

                # Tabela de produtos (limitada para performance e legibilidade)
                df_show = df_l_display.rename(columns={
                    "nome": "Produto",
                    "preco": "Preço (€)",
                    "em_promocao": "Promoção",
                }).copy()
                df_show["Preço (€)"] = df_show["Preço (€)"].apply(_fmt_preco)
                st.dataframe(df_show, width="stretch", hide_index=True)


# ===========================================================================
# Tab 7 — Validação do recomendador (back-test vs baselines)
# ===========================================================================

with tab_validacao:
    st.subheader("Validação do Recomendador")
    st.caption(
        "Quanto poupa o utilizador ao seguir as recomendações do sistema vs "
        "comprar **aleatoriamente** ou **ser leal a uma única cadeia**. "
        "Resultados produzidos por `scripts/backtest_recomendador.py` sobre "
        "o histórico real de preços."
    )

    backtest_path = Path("data/generated/backtest_recomendador.json")
    if not backtest_path.exists():
        st.warning(
            "Ficheiro de back-test não encontrado. "
            "Corre `python scripts/backtest_recomendador.py` para o gerar."
        )
    else:
        with open(backtest_path, encoding="utf-8") as _f:
            bt = json.load(_f)

        meta = bt.get("metadata", {})
        glob = bt.get("global", {})

        st.markdown(
            f"**Configuração**: {meta.get('n_listas', '?')} listas × "
            f"{meta.get('n_dias', '?')} dias do histórico · "
            f"{meta.get('n_amostras_aleatorio', '?')} amostras Monte Carlo na estratégia aleatória "
            f"(seed={meta.get('seed', '?')})  ·  Gerado em {meta.get('gerado_em', '?')}"
        )

        c1, c2, c3 = st.columns(3)
        c1.metric(
            "Poupança vs aleatório",
            f"{glob.get('poupanca_vs_aleatorio_pct', 0):+.1f}%",
            help="O sistema vs um utilizador que não compara preços entre cadeias.",
        )
        c2.metric(
            "Poupança vs melhor cadeia única",
            f"{glob.get('poupanca_vs_melhor_lealdade_pct', 0):+.1f}%",
            help=(
                "O sistema vs um utilizador que compra sempre na mesma cadeia. "
                "Só conta cadeias que têm a lista na maioria dos dias e a "
                "comparação é feita nos MESMOS dias — por isso mede o ganho real "
                "de dividir a lista entre cadeias."
            ),
        )
        c3.metric(
            "Ganho médio por lista",
            f"{glob.get('poupanca_vs_aleatorio_eur', 0):+.2f} €",
            help="Em valor absoluto, contra a estratégia aleatória.",
        )

        st.divider()
        st.subheader("Detalhe por lista")
        agregado = bt.get("agregado_por_lista", {})
        if agregado:
            rows = []
            for nome, v in agregado.items():
                rows.append({
                    "Lista":                nome,
                    "Recomendador (€)":     v["custo_recomendador_medio"],
                    "Aleatório (€)":        v["custo_aleatorio_medio"],
                    "Poupança vs aleat. (€)": v["poupanca_vs_aleatorio_eur"],
                    "Poupança vs aleat. (%)": v["poupanca_vs_aleatorio_pct"],
                    "Cadeia leal":          v.get("melhor_cadeia_lealdade") or "—",
                    "vs melhor lealdade (%)": v["poupanca_vs_melhor_lealdade_pct"],
                    "n dias":               v["n_dias"],
                })
            df_bt = pd.DataFrame(rows)

            fig = px.bar(
                df_bt, x="Lista", y=["Recomendador (€)", "Aleatório (€)"],
                barmode="group",
                title="Custo médio por lista — recomendador vs aleatório",
                color_discrete_sequence=["#3aa757", "#d35400"],
            )
            fig.update_layout(yaxis_title="Custo médio (€)", xaxis_title=None)
            st.plotly_chart(fig, width="stretch")

            st.dataframe(
                df_bt,
                width="stretch",
                hide_index=True,
                column_config={
                    "Recomendador (€)": st.column_config.NumberColumn(format="%.2f"),
                    "Aleatório (€)": st.column_config.NumberColumn(format="%.2f"),
                    "Poupança vs aleat. (€)": st.column_config.NumberColumn(format="%.2f"),
                    "Poupança vs aleat. (%)": st.column_config.NumberColumn(format="%.1f"),
                    "vs melhor lealdade (%)": st.column_config.NumberColumn(format="%.1f"),
                },
            )

            # Itens das listas
            with st.expander("Itens de cada lista testada"):
                for nome, itens in bt.get("listas", {}).items():
                    st.markdown(f"**{nome}**: " + ", ".join(f"`{i}`" for i in itens))

        st.divider()
        st.caption(
            "💡 **Como interpretar**: o ganho vs aleatório quantifica o valor "
            "óbvio de comparar preços. O ganho vs melhor cadeia única é mais "
            "restritivo (e por isso menor) porque já compete contra a cadeia "
            "ótima conhecida em retrospectiva — capta apenas o ganho marginal "
            "de **dividir a lista entre cadeias**."
        )
