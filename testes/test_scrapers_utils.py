"""
Testes para scrapers/utils.py.

Cobre todas as funções públicas do módulo de utilitários partilhados:
  - clean_text
  - normalize_price_string
  - is_valid_current_price
  - normalize_unit
  - normalize_unit_price
  - slugify_category_part
  - normalize_category
  - build_product_record
  - save_to_csv / save_to_json  (I/O com ficheiros temporários)
"""

from __future__ import annotations

import json
import csv

import pytest

from scrapers.utils import (
    OUTPUT_FIELDNAMES,
    build_product_record,
    clean_text,
    is_valid_current_price,
    make_session,
    normalize_category,
    normalize_price_string,
    normalize_unit,
    normalize_unit_price,
    save_to_csv,
    save_to_json,
    slugify_category_part,
)


# ---------------------------------------------------------------------------
# clean_text
# ---------------------------------------------------------------------------

class TestCleanText:
    def test_none_devolve_vazio(self):
        assert clean_text(None) == ""

    def test_vazio_devolve_vazio(self):
        assert clean_text("") == ""

    def test_espacos_normais(self):
        assert clean_text("  arroz  ") == "arroz"

    def test_multiplos_espacos_colapsam(self):
        assert clean_text("arroz   agulha") == "arroz agulha"

    def test_tabs_e_newlines_colapsam(self):
        assert clean_text("arroz\tagulha\ncigala") == "arroz agulha cigala"

    def test_string_ja_limpa(self):
        assert clean_text("Arroz Agulha") == "Arroz Agulha"

    def test_so_espacos_devolve_vazio(self):
        assert clean_text("   ") == ""


# ---------------------------------------------------------------------------
# normalize_price_string
# ---------------------------------------------------------------------------

class TestNormalizePriceString:
    def test_none_devolve_vazio(self):
        assert normalize_price_string(None) == ""

    def test_vazio_devolve_vazio(self):
        assert normalize_price_string("") == ""

    def test_formato_pt_virgula(self):
        assert normalize_price_string("1,69") == "1.69"

    def test_formato_ponto(self):
        assert normalize_price_string("1.69") == "1.69"

    def test_com_simbolo_euro(self):
        assert normalize_price_string("1,69 €") == "1.69"

    def test_euro_sem_espaco(self):
        assert normalize_price_string("1.69€") == "1.69"

    def test_separador_milhares_europeu(self):
        assert normalize_price_string("1.234,56 €") == "1234.56"

    def test_pvpr_prefix(self):
        assert normalize_price_string("PVPR 3,15€") == "3.15"

    def test_inteiro(self):
        assert normalize_price_string("5") == "5"

    def test_espaco_nao_separavel(self):
        # \xa0 é o non-breaking space — após substituição por espaço normal
        # "1\xa069 €" → "1 69 €" (sem separador decimal claro) → extrai "1"
        # O caso útil é "\xa0" em torno do símbolo, não a separar número de vírgula
        assert normalize_price_string("1.69\xa0€") == "1.69"
        assert normalize_price_string("1,69\xa0€") == "1.69"

    def test_string_sem_numero_devolve_vazio(self):
        assert normalize_price_string("abc") == ""

    def test_zero(self):
        assert normalize_price_string("0") == "0"


# ---------------------------------------------------------------------------
# is_valid_current_price
# ---------------------------------------------------------------------------

class TestIsValidCurrentPrice:
    def test_none_e_invalido(self):
        assert not is_valid_current_price(None)

    def test_vazio_e_invalido(self):
        assert not is_valid_current_price("")

    def test_zero_e_invalido(self):
        assert not is_valid_current_price("0")

    def test_negativo_e_invalido(self):
        assert not is_valid_current_price("-1.5")

    def test_preco_normal_e_valido(self):
        assert is_valid_current_price("1.69")

    def test_centimos_e_valido(self):
        assert is_valid_current_price("0.01")

    def test_preco_alto_e_valido(self):
        assert is_valid_current_price("99.99")

    def test_nao_numerico_e_invalido(self):
        assert not is_valid_current_price("abc")

    def test_string_vazia_e_invalida(self):
        assert not is_valid_current_price("  ")


# ---------------------------------------------------------------------------
# normalize_unit
# ---------------------------------------------------------------------------

class TestNormalizeUnit:
    @pytest.mark.parametrize("entrada,esperado", [
        ("l",      "l"),
        ("lt",     "l"),
        ("ltr",    "l"),
        ("lts",    "l"),
        ("litro",  "l"),
        ("litros", "l"),
        ("cl",     "cl"),
        ("dl",     "dl"),
        ("ml",     "ml"),
        ("kg",     "kg"),
        ("g",      "g"),
        ("gr",     "g"),
        ("grs",    "g"),
        ("un",     "un"),
        ("uni",    "un"),
        ("unid",   "un"),
        ("unids",  "un"),
        ("caps",   "un"),
        ("emb",    "un"),
        ("dose",   "un"),
        ("doses",  "un"),
        ("pack",   "un"),
    ])
    def test_mapeamento(self, entrada, esperado):
        assert normalize_unit(entrada) == esperado

    def test_maiusculas_sao_normalizadas(self):
        assert normalize_unit("KG") == "kg"

    def test_desconhecida_devolve_lowercase(self):
        assert normalize_unit("XPTO") == "xpto"


# ---------------------------------------------------------------------------
# normalize_unit_price
# ---------------------------------------------------------------------------

class TestNormalizeUnitPrice:
    def test_none_devolve_vazio(self):
        assert normalize_unit_price(None) == ""

    def test_vazio_devolve_vazio(self):
        assert normalize_unit_price("") == ""

    def test_formato_normal(self):
        assert normalize_unit_price("1.69€/kg") == "1.69€/kg"

    def test_virgula_decimal(self):
        assert normalize_unit_price("22,05€/kg") == "22.05€/kg"

    def test_litro_alias(self):
        assert normalize_unit_price("2.14€/lt") == "2.14€/l"

    def test_espacos_em_volta(self):
        assert normalize_unit_price("1,69 € / l") == "1.69€/l"

    def test_sem_unidade_devolve_vazio(self):
        assert normalize_unit_price("1.69€") == ""

    def test_texto_arbitrario_devolve_vazio(self):
        assert normalize_unit_price("preço por kilo") == ""


# ---------------------------------------------------------------------------
# slugify_category_part
# ---------------------------------------------------------------------------

class TestSlugifyCategoryPart:
    def test_acentos_removidos(self):
        assert slugify_category_part("Café & Chá") == "cafe-e-cha"

    def test_cedilha(self):
        assert slugify_category_part("açúcar") == "acucar"

    def test_slash_vira_hifen(self):
        assert slugify_category_part("Arroz/Massas") == "arroz-massas"

    def test_ampersand_vira_e(self):
        assert slugify_category_part("Mercearia & Bebidas") == "mercearia-e-bebidas"

    def test_espacos_viram_hifen(self):
        assert slugify_category_part("Pingo Doce") == "pingo-doce"

    def test_maiusculas_normalizadas(self):
        assert slugify_category_part("ARROZ") == "arroz"

    def test_hifens_consecutivos_colapsam(self):
        result = slugify_category_part("Arroz & Massas & Cereais")
        assert "--" not in result

    def test_vazio_devolve_vazio(self):
        assert slugify_category_part("") == ""


# ---------------------------------------------------------------------------
# normalize_category
# ---------------------------------------------------------------------------

class TestNormalizeCategory:
    def test_none_devolve_vazio(self):
        assert normalize_category(None) == ""

    def test_vazio_devolve_vazio(self):
        assert normalize_category("") == ""

    def test_categoria_simples(self):
        assert normalize_category("alimentacao") == "alimentacao"

    def test_hierarquia(self):
        result = normalize_category("Mercearia/Arroz & Massas")
        assert "/" in result
        assert "mercearia" in result
        assert "arroz" in result

    def test_segmentos_vazios_ignorados(self):
        result = normalize_category("alimentacao//mercearia")
        assert result == "alimentacao/mercearia"

    def test_acentos_na_hierarquia(self):
        result = normalize_category("Alimentação/Bolachas & Cereais")
        assert "alimentacao" in result
        assert "/" in result


# ---------------------------------------------------------------------------
# build_product_record
# ---------------------------------------------------------------------------

class TestBuildProductRecord:
    def test_campos_presentes(self):
        record = build_product_record(
            id_externo="1",
            nome="Arroz",
            marca="Cigala",
            categoria="alimentacao",
            preco="1.69",
            preco_original="",
            preco_unitario="1.69€/kg",
            url="https://example.com",
            imagem="",
            loja="Auchan",
            data_recolha="2026-04-18",
        )
        for campo in OUTPUT_FIELDNAMES:
            assert campo in record

    def test_ean_default_vazio(self):
        record = build_product_record(
            id_externo="1", nome="Arroz", marca="", categoria="",
            preco="1.69", preco_original="", preco_unitario="",
            url="", imagem="", loja="Auchan", data_recolha="2026-04-18",
        )
        assert record["ean"] == ""

    def test_ean_explicito(self):
        record = build_product_record(
            id_externo="1", nome="Arroz", marca="", categoria="",
            preco="1.69", preco_original="", preco_unitario="",
            url="", imagem="", loja="Auchan", data_recolha="2026-04-18",
            ean="5601234567890",
        )
        assert record["ean"] == "5601234567890"

    def test_valores_preservados(self):
        record = build_product_record(
            id_externo="99",
            nome="Leite",
            marca="Mimosa",
            categoria="frescos",
            preco="0.85",
            preco_original="1.09",
            preco_unitario="0.85€/l",
            url="https://example.com/leite",
            imagem="https://example.com/img.jpg",
            loja="Pingo Doce",
            data_recolha="2026-04-18 10:00:00",
            ean="1234567890123",
        )
        assert record["id_externo"] == "99"
        assert record["nome"] == "Leite"
        assert record["marca"] == "Mimosa"
        assert record["preco"] == "0.85"
        assert record["preco_original"] == "1.09"
        assert record["loja"] == "Pingo Doce"
        assert record["ean"] == "1234567890123"


# ---------------------------------------------------------------------------
# save_to_csv / save_to_json
# ---------------------------------------------------------------------------

class TestSaveToCSV:
    def test_cria_ficheiro(self, tmp_path, produto_simples):
        result = save_to_csv([produto_simples], "test.csv", tmp_path)
        assert result is not None
        assert result.exists()

    def test_conteudo_csv(self, tmp_path, produto_simples):
        save_to_csv([produto_simples], "test.csv", tmp_path)
        with open(tmp_path / "test.csv", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 1
        assert rows[0]["nome"] == produto_simples["nome"]
        assert rows[0]["preco"] == produto_simples["preco"]

    def test_cabecalho_correto(self, tmp_path, produto_simples):
        save_to_csv([produto_simples], "test.csv", tmp_path)
        with open(tmp_path / "test.csv", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            assert set(reader.fieldnames) == set(OUTPUT_FIELDNAMES)

    def test_lista_vazia_devolve_none(self, tmp_path):
        result = save_to_csv([], "test.csv", tmp_path)
        assert result is None

    def test_multiplos_produtos(self, tmp_path, batch_produtos):
        save_to_csv(batch_produtos, "test.csv", tmp_path)
        with open(tmp_path / "test.csv", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 3


class TestSaveToJSON:
    def test_cria_ficheiro(self, tmp_path, produto_simples):
        result = save_to_json([produto_simples], "test.json", tmp_path)
        assert result is not None
        assert result.exists()

    def test_conteudo_json(self, tmp_path, produto_simples):
        save_to_json([produto_simples], "test.json", tmp_path)
        with open(tmp_path / "test.json", encoding="utf-8") as f:
            data = json.load(f)
        assert len(data) == 1
        assert data[0]["nome"] == produto_simples["nome"]

    def test_lista_vazia_devolve_none(self, tmp_path):
        result = save_to_json([], "test.json", tmp_path)
        assert result is None

    def test_caracteres_unicode(self, tmp_path, produto_simples):
        p = dict(produto_simples)
        p["nome"] = "Açúcar Refinado Branco"
        save_to_json([p], "test.json", tmp_path)
        with open(tmp_path / "test.json", encoding="utf-8") as f:
            data = json.load(f)
        assert data[0]["nome"] == "Açúcar Refinado Branco"


# ---------------------------------------------------------------------------
# make_session
# ---------------------------------------------------------------------------

class TestMakeSession:
    def test_session_tem_user_agent(self):
        session = make_session()
        ua = session.headers.get("User-Agent", "")
        assert "Mozilla" in ua

    def test_session_tem_accept_language(self):
        session = make_session()
        al = session.headers.get("Accept-Language", "")
        assert "pt" in al.lower()
