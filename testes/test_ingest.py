"""
Testes para scripts/ingest.py.

Cobre todas as funções puras (sem BD):
  - limpar_texto
  - normalizar_decimal
  - normalizar_texto_chave
  - decimal_para_str_sem_zeros
  - decimal_abs
  - normalizar_unidade
  - aproximar_quantidade_plausivel
  - converter_para_unidade_canonica
  - extrair_pack_quantidade
  - extrair_quantidade_simples
  - extrair_preco_unitario
  - inferir_quantidade_por_preco_unitario
  - extrair_info_quantidade
  - parse_data_recolha
  - limpar_nome_base_para_chave
  - construir_descriptor_quantidade
  - construir_chave_mestre
  - construir_nome_padronizado
  - deduplicar_registos
  - ingestao (dry-run)
"""

from __future__ import annotations

import json
from datetime import timezone
from decimal import Decimal
from zoneinfo import ZoneInfo

import pytest

from scripts.ingest import (
    aproximar_quantidade_plausivel,
    construir_chave_mestre,
    construir_nome_padronizado,
    converter_para_unidade_canonica,
    decimal_abs,
    decimal_para_str_sem_zeros,
    deduplicar_registos,
    extrair_info_quantidade,
    extrair_pack_quantidade,
    extrair_preco_unitario,
    extrair_quantidade_simples,
    inferir_quantidade_por_preco_unitario,
    ingestao,
    limpar_nome_base_para_chave,
    limpar_texto,
    normalizar_decimal,
    normalizar_texto_chave,
    normalizar_unidade,
    parse_data_recolha,
)

TZ_LOCAL = ZoneInfo("Europe/Lisbon")


# ---------------------------------------------------------------------------
# limpar_texto
# ---------------------------------------------------------------------------

class TestLimparTexto:
    def test_none_devolve_vazio(self):
        assert limpar_texto(None) == ""

    def test_vazio(self):
        assert limpar_texto("") == ""

    def test_espaços_colapsam(self):
        assert limpar_texto("  arroz  agulha  ") == "arroz agulha"

    def test_tabs_e_newlines(self):
        assert limpar_texto("arroz\n\tagulha") == "arroz agulha"


# ---------------------------------------------------------------------------
# normalizar_decimal
# ---------------------------------------------------------------------------

class TestNormalizarDecimal:
    def test_none_devolve_none(self):
        assert normalizar_decimal(None) is None

    def test_vazio_devolve_none(self):
        assert normalizar_decimal("") is None

    def test_ponto_decimal(self):
        assert normalizar_decimal("1.69") == Decimal("1.69")

    def test_virgula_decimal(self):
        assert normalizar_decimal("1,69") == Decimal("1.69")

    def test_separador_milhares(self):
        assert normalizar_decimal("1.234,56") == Decimal("1234.56")

    def test_com_euro(self):
        assert normalizar_decimal("3,15€") == Decimal("3.15")

    def test_inteiro(self):
        assert normalizar_decimal("5") == Decimal("5")

    def test_zero(self):
        assert normalizar_decimal("0") == Decimal("0")

    def test_nao_numerico_devolve_none(self):
        assert normalizar_decimal("abc") is None


# ---------------------------------------------------------------------------
# normalizar_texto_chave
# ---------------------------------------------------------------------------

class TestNormalizarTextoChave:
    def test_acentos_removidos(self):
        result = normalizar_texto_chave("Arroz Agulha")
        assert result == "arroz-agulha"

    def test_cedilha(self):
        result = normalizar_texto_chave("Açúcar")
        assert result == "acucar"

    def test_percentagem(self):
        result = normalizar_texto_chave("50% Desconto")
        assert "pct" in result

    def test_caracteres_especiais_viram_hifen(self):
        result = normalizar_texto_chave("Arroz & Massas")
        assert "--" not in result
        assert result == result.strip("-")

    def test_none_devolve_vazio(self):
        assert normalizar_texto_chave(None) == ""


# ---------------------------------------------------------------------------
# decimal_para_str_sem_zeros
# ---------------------------------------------------------------------------

class TestDecimalParaStrSemZeros:
    def test_none_devolve_vazio(self):
        assert decimal_para_str_sem_zeros(None) == ""

    def test_inteiro_sem_zeros(self):
        assert decimal_para_str_sem_zeros(Decimal("500.000")) == "500"

    def test_decimal_sem_zeros_finais(self):
        assert decimal_para_str_sem_zeros(Decimal("1.500")) == "1.5"

    def test_decimal_dois_digitos(self):
        assert decimal_para_str_sem_zeros(Decimal("1.250")) == "1.25"

    def test_valor_exato(self):
        assert decimal_para_str_sem_zeros(Decimal("1.69")) == "1.69"


# ---------------------------------------------------------------------------
# decimal_abs
# ---------------------------------------------------------------------------

class TestDecimalAbs:
    def test_none_devolve_none(self):
        assert decimal_abs(None) is None

    def test_positivo(self):
        assert decimal_abs(Decimal("1.5")) == Decimal("1.5")

    def test_negativo(self):
        assert decimal_abs(Decimal("-1.5")) == Decimal("1.5")

    def test_zero(self):
        assert decimal_abs(Decimal("0")) == Decimal("0")


# ---------------------------------------------------------------------------
# normalizar_unidade
# ---------------------------------------------------------------------------

class TestNormalizarUnidade:
    @pytest.mark.parametrize("entrada,esperado", [
        ("lt",       "l"),
        ("uni",      "un"),
        ("unid",     "un"),
        ("unids",    "un"),
        ("caps",     "un"),
        ("capsulas", "un"),
        ("saq",      "un"),
        ("saquetas", "un"),
        ("doses",    "un"),
        ("kg",       "kg"),
        ("g",        "g"),
        ("l",        "l"),
        ("ml",       "ml"),
    ])
    def test_mapeamento(self, entrada, esperado):
        assert normalizar_unidade(entrada) == esperado

    def test_desconhecida_lowercase(self):
        assert normalizar_unidade("XPTO") == "xpto"


# ---------------------------------------------------------------------------
# converter_para_unidade_canonica
# ---------------------------------------------------------------------------

class TestConverterParaUnidadeCanonica:
    def test_kg_para_g(self):
        q, u = converter_para_unidade_canonica(Decimal("1"), "kg")
        assert q == Decimal("1000")
        assert u == "g"

    def test_mg_para_g(self):
        q, u = converter_para_unidade_canonica(Decimal("500"), "mg")
        assert q == Decimal("0.5")
        assert u == "g"

    def test_l_para_ml(self):
        q, u = converter_para_unidade_canonica(Decimal("1"), "l")
        assert q == Decimal("1000")
        assert u == "ml"

    def test_cl_para_ml(self):
        q, u = converter_para_unidade_canonica(Decimal("33"), "cl")
        assert q == Decimal("330")
        assert u == "ml"

    def test_dl_para_ml(self):
        q, u = converter_para_unidade_canonica(Decimal("2"), "dl")
        assert q == Decimal("200")
        assert u == "ml"

    def test_g_inalterado(self):
        q, u = converter_para_unidade_canonica(Decimal("500"), "g")
        assert q == Decimal("500")
        assert u == "g"

    def test_ml_inalterado(self):
        q, u = converter_para_unidade_canonica(Decimal("330"), "ml")
        assert q == Decimal("330")
        assert u == "ml"

    def test_none_devolve_none(self):
        q, u = converter_para_unidade_canonica(None, "kg")
        assert q is None

    def test_lt_como_alias(self):
        q, u = converter_para_unidade_canonica(Decimal("1"), "lt")
        assert q == Decimal("1000")
        assert u == "ml"


# ---------------------------------------------------------------------------
# aproximar_quantidade_plausivel
# ---------------------------------------------------------------------------

class TestAproximarQuantidadePlausivel:
    def test_500g_exato(self):
        q, u = aproximar_quantidade_plausivel(Decimal("500"), "g")
        assert q == Decimal("500")
        assert u == "g"

    def test_499g_arredonda_para_500(self):
        # 499 está a 0.2% de 500 (< 8%) → arredonda para o tamanho comercial 500
        q, u = aproximar_quantidade_plausivel(Decimal("499"), "g")
        assert q == Decimal("500")

    def test_1000ml_exato(self):
        q, u = aproximar_quantidade_plausivel(Decimal("1000"), "ml")
        assert q == Decimal("1000")

    def test_6_un_exato(self):
        q, u = aproximar_quantidade_plausivel(Decimal("6"), "un")
        assert q == Decimal("6")

    def test_unidade_desconhecida_devolve_none(self):
        # Unidade fora de {g, ml, un} não tem candidatos nem fallback
        q, u = aproximar_quantidade_plausivel(Decimal("100"), "xyz")
        assert q is None
        assert u is None


# ---------------------------------------------------------------------------
# extrair_pack_quantidade
# ---------------------------------------------------------------------------

class TestExtrairPackQuantidade:
    def test_pack_simples(self):
        result = extrair_pack_quantidade("Iogurte Natural 3x250g")
        assert result is not None
        assert result["tipo"] == "pack"
        assert result["multiplicador"] == 3

    def test_pack_com_litros(self):
        result = extrair_pack_quantidade("Leite 6 x 1l")
        assert result is not None
        assert result["multiplicador"] == 6

    def test_pack_com_cl(self):
        result = extrair_pack_quantidade("Cerveja 6x33cl")
        assert result is not None
        assert result["multiplicador"] == 6
        assert result["unidade_total"] == "ml"

    def test_sem_pack(self):
        result = extrair_pack_quantidade("Arroz Agulha 1kg")
        assert result is None

    def test_quantidade_total_calculada(self):
        result = extrair_pack_quantidade("Iogurte 3x250g")
        assert result is not None
        assert result["quantidade_total"] == Decimal("750")

    def test_case_insensitive(self):
        result = extrair_pack_quantidade("Cerveja 4X500ML")
        assert result is not None


# ---------------------------------------------------------------------------
# extrair_quantidade_simples
# ---------------------------------------------------------------------------

class TestExtrairQuantidadeSimples:
    def test_gramas(self):
        result = extrair_quantidade_simples("Arroz Agulha 1000g")
        assert result is not None
        assert result["quantidade_total"] == Decimal("1000")
        assert result["unidade_total"] == "g"

    def test_kg_converte_para_g(self):
        result = extrair_quantidade_simples("Arroz Agulha 1kg")
        assert result is not None
        assert result["quantidade_total"] == Decimal("1000")
        assert result["unidade_total"] == "g"

    def test_litros_converte_para_ml(self):
        result = extrair_quantidade_simples("Leite Mimosa 1l")
        assert result is not None
        assert result["quantidade_total"] == Decimal("1000")
        assert result["unidade_total"] == "ml"

    def test_ml(self):
        result = extrair_quantidade_simples("Sumo 330ml")
        assert result is not None
        assert result["quantidade_total"] == Decimal("330")

    def test_ultima_ocorrencia(self):
        # "3 Queijos Barreado 200g" — o 3 é contagem, 200g é a quantidade
        result = extrair_quantidade_simples("3 Queijos Barreado 200g")
        assert result is not None
        assert result["quantidade_total"] == Decimal("200")

    def test_sem_quantidade(self):
        result = extrair_quantidade_simples("Azeite Virgem Extra")
        assert result is None


# ---------------------------------------------------------------------------
# extrair_preco_unitario
# ---------------------------------------------------------------------------

class TestExtrairPrecoUnitario:
    def test_formato_normal(self):
        v, u = extrair_preco_unitario("2.50€/kg")
        assert v == Decimal("2.50")
        assert u == "kg"

    def test_virgula_decimal(self):
        v, u = extrair_preco_unitario("1,69€/kg")
        assert v == Decimal("1.69")
        assert u == "kg"

    def test_unidade_alias(self):
        v, u = extrair_preco_unitario("1.69€/lt")
        assert u == "l"

    def test_sem_unidade(self):
        v, u = extrair_preco_unitario("1.69€")
        assert v is None
        assert u is None

    def test_none(self):
        v, u = extrair_preco_unitario(None)
        assert v is None

    def test_vazio(self):
        v, u = extrair_preco_unitario("")
        assert v is None


# ---------------------------------------------------------------------------
# inferir_quantidade_por_preco_unitario
# ---------------------------------------------------------------------------

class TestInferirQuantidadePorPrecoUnitario:
    def test_1kg_por_preco(self):
        item = {"preco": "1.69", "preco_unitario": "1.69€/kg"}
        result = inferir_quantidade_por_preco_unitario(item)
        assert result is not None
        # 1.69 / 1.69 = 1kg → 1000g
        assert result["quantidade_total"] == Decimal("1000")
        assert result["unidade_total"] == "g"

    def test_sem_preco_unitario(self):
        item = {"preco": "1.69", "preco_unitario": ""}
        result = inferir_quantidade_por_preco_unitario(item)
        assert result is None

    def test_preco_unitario_zero(self):
        item = {"preco": "1.69", "preco_unitario": "0€/kg"}
        result = inferir_quantidade_por_preco_unitario(item)
        assert result is None


# ---------------------------------------------------------------------------
# extrair_info_quantidade (integração)
# ---------------------------------------------------------------------------

class TestExtrairInfoQuantidade:
    def test_prefere_pack(self):
        item = {"nome": "Cerveja 6x33cl", "preco": "3.99", "preco_unitario": ""}
        result = extrair_info_quantidade(item)
        assert result is not None
        assert result["tipo"] == "pack"

    def test_fallback_simples(self):
        item = {"nome": "Arroz Agulha 1kg", "preco": "1.69", "preco_unitario": ""}
        result = extrair_info_quantidade(item)
        assert result is not None
        assert result["tipo"] == "simples"

    def test_fallback_preco_unitario(self):
        item = {"nome": "Azeite Virgem Extra", "preco": "1.69", "preco_unitario": "1.69€/l"}
        result = extrair_info_quantidade(item)
        assert result is not None
        assert result["fonte"] == "preco_unitario"

    def test_sem_quantidade(self):
        item = {"nome": "Produto Sem Quantidade", "preco": "1.99", "preco_unitario": ""}
        result = extrair_info_quantidade(item)
        assert result is None


# ---------------------------------------------------------------------------
# parse_data_recolha
# ---------------------------------------------------------------------------

class TestParseDataRecolha:
    def test_formato_datetime_completo(self):
        dt = parse_data_recolha("2026-04-18 10:00:00")
        assert dt.tzinfo == timezone.utc
        assert dt.year == 2026
        assert dt.month == 4
        assert dt.day == 18

    def test_formato_iso_t(self):
        dt = parse_data_recolha("2026-04-18T10:00:00")
        assert dt.tzinfo == timezone.utc

    def test_formato_data_apenas(self):
        dt = parse_data_recolha("2026-04-18")
        assert dt.tzinfo == timezone.utc
        assert dt.year == 2026

    def test_formato_utc_z(self):
        dt = parse_data_recolha("2026-04-18T10:00:00Z")
        assert dt.tzinfo == timezone.utc

    def test_vazio_lanca_valueerror(self):
        with pytest.raises(ValueError):
            parse_data_recolha("")

    def test_none_lanca_valueerror(self):
        with pytest.raises(ValueError):
            parse_data_recolha(None)

    def test_formato_invalido_lanca_valueerror(self):
        with pytest.raises(ValueError):
            parse_data_recolha("nao-e-data")


# ---------------------------------------------------------------------------
# limpar_nome_base_para_chave
# ---------------------------------------------------------------------------

class TestLimparNomeBaseParaChave:
    def test_remove_quantidade_simples(self):
        result = limpar_nome_base_para_chave("Arroz Agulha 1kg")
        assert "1kg" not in result
        assert "1" not in result.split()  # deve remover o número

    def test_remove_pack(self):
        result = limpar_nome_base_para_chave("Iogurte 3x250g")
        assert "3x250g" not in result

    def test_mantém_nome_base(self):
        result = limpar_nome_base_para_chave("Arroz Agulha 1kg")
        assert "arroz" in result

    def test_maiusculas_para_minusculas(self):
        result = limpar_nome_base_para_chave("ARROZ AGULHA 1KG")
        assert result == result.lower()


# ---------------------------------------------------------------------------
# construir_chave_mestre
# ---------------------------------------------------------------------------

class TestConstruirChaveMestre:
    def test_com_ean_usa_ean(self):
        item = {"ean": "5601234567890", "nome": "Arroz", "marca": "", "categoria": "", "preco": "1.69", "preco_unitario": ""}
        chave = construir_chave_mestre(item)
        assert chave == "ean:5601234567890"

    def test_sem_ean_usa_hash_semantico(self):
        item = {"ean": "", "nome": "Arroz Agulha Cigala 1kg", "marca": "Cigala", "categoria": "alimentacao", "preco": "1.69", "preco_unitario": "1.69€/kg"}
        chave = construir_chave_mestre(item)
        assert chave.startswith("n:")
        assert "|m:" in chave
        assert "|c:" in chave

    def test_deterministica(self):
        item = {"ean": "", "nome": "Leite Mimosa 1l", "marca": "Mimosa", "categoria": "frescos", "preco": "0.85", "preco_unitario": "0.85€/l"}
        chave1 = construir_chave_mestre(item)
        chave2 = construir_chave_mestre(item)
        assert chave1 == chave2

    def test_ean_diferente_gera_chave_diferente(self):
        item1 = {"ean": "111", "nome": "Arroz", "marca": "", "categoria": "", "preco": "1", "preco_unitario": ""}
        item2 = {"ean": "222", "nome": "Arroz", "marca": "", "categoria": "", "preco": "1", "preco_unitario": ""}
        assert construir_chave_mestre(item1) != construir_chave_mestre(item2)


# ---------------------------------------------------------------------------
# construir_nome_padronizado
# ---------------------------------------------------------------------------

class TestConstruirNomePadronizado:
    def test_capitaliza_palavras(self):
        item = {"nome": "arroz agulha cigala 1000g", "marca": "Cigala", "preco": "1.69", "preco_unitario": "1.69€/kg"}
        nome = construir_nome_padronizado(item)
        assert nome[0].isupper()

    def test_adiciona_marca_se_ausente(self):
        item = {"nome": "Arroz Agulha 1000g", "marca": "Cigala", "preco": "1.69", "preco_unitario": "1.69€/kg"}
        nome = construir_nome_padronizado(item)
        assert "Cigala" in nome

    def test_nao_duplica_marca(self):
        item = {"nome": "Arroz Agulha Cigala 1000g", "marca": "Cigala", "preco": "1.69", "preco_unitario": "1.69€/kg"}
        nome = construir_nome_padronizado(item)
        assert nome.lower().count("cigala") == 1

    def test_inclui_quantidade(self):
        item = {"nome": "Arroz Agulha Cigala 1kg", "marca": "", "preco": "1.69", "preco_unitario": "1.69€/kg"}
        nome = construir_nome_padronizado(item)
        assert "1000" in nome or "1kg" in nome.lower() or "kg" in nome.lower()


# ---------------------------------------------------------------------------
# deduplicar_registos
# ---------------------------------------------------------------------------

class TestDeduplicarRegistos:
    def test_sem_duplicados(self):
        registos = [
            {"loja": "Auchan", "id_externo": "1", "data_recolha": "2026-04-18", "nome": "A"},
            {"loja": "Auchan", "id_externo": "2", "data_recolha": "2026-04-18", "nome": "B"},
        ]
        result = deduplicar_registos(registos)
        assert len(result) == 2

    def test_duplicados_mesma_data(self):
        registos = [
            {"loja": "Auchan", "id_externo": "1", "data_recolha": "2026-04-18", "nome": "A1"},
            {"loja": "Auchan", "id_externo": "1", "data_recolha": "2026-04-18", "nome": "A2"},
        ]
        result = deduplicar_registos(registos)
        assert len(result) == 1

    def test_mantém_mais_recente(self):
        registos = [
            {"loja": "Auchan", "id_externo": "1", "data_recolha": "2026-04-17", "nome": "Antigo"},
            {"loja": "Auchan", "id_externo": "1", "data_recolha": "2026-04-18", "nome": "Recente"},
        ]
        result = deduplicar_registos(registos)
        assert len(result) == 1
        assert result[0]["nome"] == "Recente"

    def test_lojas_diferentes_nao_deduplicam(self):
        registos = [
            {"loja": "Auchan",     "id_externo": "1", "data_recolha": "2026-04-18", "nome": "A"},
            {"loja": "Continente", "id_externo": "1", "data_recolha": "2026-04-18", "nome": "A"},
        ]
        result = deduplicar_registos(registos)
        assert len(result) == 2

    def test_lista_vazia(self):
        assert deduplicar_registos([]) == []

    def test_ordem_primeira_aparicao_mantida(self):
        registos = [
            {"loja": "Auchan", "id_externo": "2", "data_recolha": "2026-04-18", "nome": "B"},
            {"loja": "Auchan", "id_externo": "1", "data_recolha": "2026-04-18", "nome": "A"},
        ]
        result = deduplicar_registos(registos)
        assert result[0]["id_externo"] == "2"

    def test_ids_vazios_com_urls_diferentes_nao_colapsam(self):
        registos = [
            {
                "loja": "Pingo Doce",
                "id_externo": "",
                "url": "https://www.pingodoce.pt/home/produtos/mercearia/arroz-a.html",
                "data_recolha": "2026-04-18 10:00:00",
                "nome": "Arroz A",
                "preco": "1.00",
            },
            {
                "loja": "Pingo Doce",
                "id_externo": "",
                "url": "https://www.pingodoce.pt/home/produtos/mercearia/arroz-b.html",
                "data_recolha": "2026-04-18 10:00:00",
                "nome": "Arroz B",
                "preco": "1.20",
            },
        ]
        result = deduplicar_registos(registos)
        assert len(result) == 2

    def test_ids_vazios_mesma_url_mantem_mais_recente(self):
        registos = [
            {
                "loja": "Pingo Doce",
                "id_externo": "",
                "url": "https://www.pingodoce.pt/home/produtos/mercearia/arroz.html",
                "data_recolha": "2026-04-18 09:00:00",
                "nome": "Antigo",
                "preco": "1.00",
            },
            {
                "loja": "Pingo Doce",
                "id_externo": "",
                "url": "https://www.pingodoce.pt/home/produtos/mercearia/arroz.html",
                "data_recolha": "2026-04-18 10:00:00",
                "nome": "Recente",
                "preco": "1.10",
            },
        ]
        result = deduplicar_registos(registos)
        assert len(result) == 1
        assert result[0]["nome"] == "Recente"

# ---------------------------------------------------------------------------
# ingestao dry-run
# ---------------------------------------------------------------------------

class TestIngestaoDryRun:
    def _produto_valido(self):
        return {
            "id_externo":     "12345",
            "nome":           "Arroz Agulha Cigala 1kg",
            "marca":          "Cigala",
            "categoria":      "alimentacao",
            "preco":          "1.69",
            "preco_original": "",
            "preco_unitario": "1.69€/kg",
            "url":            "https://www.auchan.pt/produto/12345",
            "imagem":         "",
            "loja":           "Auchan",
            "data_recolha":   "2026-04-18 10:00:00",
            "ean":            "",
        }

    def test_registo_valido_em_dry_run(self):
        total_ok, falhas = ingestao([self._produto_valido()], dry_run=True)
        assert total_ok == 1
        assert len(falhas) == 0

    def test_multiplos_validos_em_dry_run(self):
        registos = [
            {**self._produto_valido(), "id_externo": str(i), "loja": "Auchan"}
            for i in range(5)
        ]
        total_ok, falhas = ingestao(registos, dry_run=True)
        assert total_ok == 5
        assert len(falhas) == 0

    def test_registo_sem_preco_falha(self):
        produto = {**self._produto_valido(), "preco": "0"}
        total_ok, falhas = ingestao([produto], dry_run=True)
        assert total_ok == 0
        assert len(falhas) == 1

    def test_registo_sem_data_falha(self):
        produto = {**self._produto_valido(), "data_recolha": ""}
        total_ok, falhas = ingestao([produto], dry_run=True)
        assert total_ok == 0
        assert len(falhas) == 1

    def test_deduplicacao_em_dry_run(self):
        produto = self._produto_valido()
        # Dois registos com o mesmo (loja, id_externo) — devem ser deduplicados para 1
        registos = [produto, {**produto, "data_recolha": "2026-04-17 09:00:00"}]
        total_ok, falhas = ingestao(registos, dry_run=True)
        assert total_ok == 1

    def test_lista_vazia_dry_run(self):
        total_ok, falhas = ingestao([], dry_run=True)
        assert total_ok == 0
        assert falhas == []

    def test_ingestao_de_ficheiro_json(self, tmp_path):
        """Carrega ficheiro JSON real e valida em dry-run."""
        produtos = [
            {
                "id_externo": f"sku{i}",
                "nome": f"Produto {i}",
                "marca": "MarcaX",
                "categoria": "alimentacao",
                "preco": f"{1 + i * 0.1:.2f}",
                "preco_original": "",
                "preco_unitario": "",
                "url": f"https://example.com/{i}",
                "imagem": "",
                "loja": "Continente",
                "data_recolha": "2026-04-18 10:00:00",
                "ean": "",
            }
            for i in range(3)
        ]
        ficheiro = tmp_path / "produtos.json"
        ficheiro.write_text(json.dumps(produtos), encoding="utf-8")

        from scripts.ingest import carregar_ficheiros
        registos, caminhos = carregar_ficheiros([str(ficheiro)])
        assert len(registos) == 3
        total_ok, falhas = ingestao(registos, dry_run=True)
        assert total_ok == 3
