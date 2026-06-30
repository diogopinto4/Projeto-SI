"""
Testes para scrapers/auchan_scraper.py.

Cobre todas as funções de parsing sem necessidade de rede:
  - extract_pid_from_url
  - category_from_url
  - filter_urls
  - parse_xml_urls
  - extract_json_ld
  - parse_product_html
  - parse_product_ajax
  - fetch_product_ajax (com mock de sessão HTTP)
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

from bs4 import BeautifulSoup

from scrapers.auchan_scraper import (
    BASE_URL,
    category_from_url,
    extract_json_ld,
    extract_pid_from_url,
    filter_urls,
    parse_product_ajax,
    parse_product_html,
    parse_xml_urls,
    fetch_product_ajax,
)


# ---------------------------------------------------------------------------
# extract_pid_from_url
# ---------------------------------------------------------------------------

class TestExtractPidFromUrl:
    def test_pid_numerico(self):
        url = "https://www.auchan.pt/pt/alimentacao/mercearia/arroz/arroz-agulha/10.html"
        assert extract_pid_from_url(url) == "10"

    def test_pid_longo(self):
        url = "https://www.auchan.pt/pt/bebidas/sumos/sumo-laranja/3400089123.html"
        assert extract_pid_from_url(url) == "3400089123"

    def test_sem_html_devolve_vazio(self):
        url = "https://www.auchan.pt/pt/alimentacao/mercearia/"
        assert extract_pid_from_url(url) == ""

    def test_url_com_querystring(self):
        url = "https://www.auchan.pt/pt/alimentacao/arroz/12345.html?color=red"
        assert extract_pid_from_url(url) == "12345"

    def test_vazio_devolve_vazio(self):
        assert extract_pid_from_url("") == ""

    def test_pid_com_varios_segmentos(self):
        url = "https://www.auchan.pt/pt/alimentacao/mercearia/arroz-massas-cereais/arroz/arroz-agulha-cigala/67890.html"
        assert extract_pid_from_url(url) == "67890"


# ---------------------------------------------------------------------------
# category_from_url
# ---------------------------------------------------------------------------

class TestCategoryFromUrl:
    def test_hierarquia_completa(self):
        url = "https://www.auchan.pt/pt/alimentacao/mercearia/arroz-massas-cereais/arroz/arroz-agulha-cigala/12345.html"
        cat = category_from_url(url)
        assert "alimentacao" in cat
        assert "mercearia" in cat

    def test_apenas_categoria(self):
        url = "https://www.auchan.pt/pt/alimentacao/produto-simples/12345.html"
        cat = category_from_url(url)
        assert "alimentacao" in cat

    def test_url_sem_pt_devolve_vazio(self):
        url = "https://www.auchan.pt/en/food/rice/12345.html"
        cat = category_from_url(url)
        assert cat == ""

    def test_slug_produto_excluido(self):
        url = "https://www.auchan.pt/pt/bebidas/sumos/sumo-laranja/99999.html"
        cat = category_from_url(url)
        # slug do produto (sumo-laranja) não deve aparecer na categoria
        assert "sumo-laranja" not in cat

    def test_devolve_string(self):
        url = "https://www.auchan.pt/pt/alimentacao/arroz/12345.html"
        assert isinstance(category_from_url(url), str)


# ---------------------------------------------------------------------------
# filter_urls
# ---------------------------------------------------------------------------

class TestFilterUrls:
    URLS = [
        "https://www.auchan.pt/pt/alimentacao/mercearia/arroz/arroz-agulha/12345.html",
        "https://www.auchan.pt/pt/alimentacao/massas/espaguete/67890.html",
        "https://www.auchan.pt/pt/bebidas-e-garrafeira/cerveja/super-bock/11111.html",
        "https://www.auchan.pt/pt/produtos-frescos/carne/frango/22222.html",
    ]

    def test_filtro_categoria(self):
        result = filter_urls(self.URLS, categoria="alimentacao")
        assert len(result) == 2
        assert all("/pt/alimentacao/" in u for u in result)

    def test_filtro_subcategoria(self):
        result = filter_urls(self.URLS, subcategoria="arroz")
        assert len(result) == 1
        assert "arroz" in result[0]

    def test_filtro_categoria_e_subcategoria(self):
        result = filter_urls(self.URLS, categoria="alimentacao", subcategoria="arroz")
        assert len(result) == 1

    def test_sem_filtro_devolve_tudo(self):
        result = filter_urls(self.URLS)
        assert len(result) == len(self.URLS)

    def test_filtro_sem_correspondencia(self):
        result = filter_urls(self.URLS, categoria="inexistente")
        assert result == []

    def test_lista_vazia(self):
        result = filter_urls([], categoria="alimentacao")
        assert result == []

    def test_case_insensitive(self):
        result = filter_urls(self.URLS, categoria="ALIMENTACAO")
        assert len(result) == 2


# ---------------------------------------------------------------------------
# parse_xml_urls
# ---------------------------------------------------------------------------

class TestParseXmlUrls:
    def test_sitemap_index(self):
        xml = """<?xml version="1.0" encoding="UTF-8"?>
<sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <sitemap><loc>https://www.auchan.pt/sitemap_0-product.xml</loc></sitemap>
  <sitemap><loc>https://www.auchan.pt/sitemap_1-product.xml</loc></sitemap>
</sitemapindex>"""
        urls = parse_xml_urls(xml)
        assert len(urls) == 2
        assert "sitemap_0-product.xml" in urls[0]

    def test_urlset(self):
        xml = """<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url><loc>https://www.auchan.pt/pt/alimentacao/arroz/12345.html</loc></url>
  <url><loc>https://www.auchan.pt/pt/bebidas/cerveja/67890.html</loc></url>
</urlset>"""
        urls = parse_xml_urls(xml)
        assert len(urls) == 2
        assert "12345.html" in urls[0]

    def test_sitemap_vazio(self):
        xml = """<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
</urlset>"""
        urls = parse_xml_urls(xml)
        assert urls == []

    def test_espacos_removidos(self):
        xml = """<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url><loc>  https://www.auchan.pt/pt/produto/12345.html  </loc></url>
</urlset>"""
        urls = parse_xml_urls(xml)
        assert urls[0] == "https://www.auchan.pt/pt/produto/12345.html"


# ---------------------------------------------------------------------------
# extract_json_ld
# ---------------------------------------------------------------------------

class TestExtractJsonLd:
    def test_json_ld_produto_completo(self, html_com_json_ld):
        soup = BeautifulSoup(html_com_json_ld, "html.parser")
        data = extract_json_ld(soup)
        assert data is not None
        assert data["@type"] == "Product"
        assert data["name"] == "Arroz Agulha Cigala 1kg"
        assert data["sku"] == "12345"
        assert data["gtin"] == "5601234567890"
        assert data["brand"]["name"] == "Cigala"

    def test_sem_json_ld(self, html_sem_json_ld):
        soup = BeautifulSoup(html_sem_json_ld, "html.parser")
        data = extract_json_ld(soup)
        assert data is None

    def test_json_ld_nao_produto(self):
        html = """<html><head>
        <script type="application/ld+json">{"@type": "WebSite", "name": "Auchan"}</script>
        </head></html>"""
        soup = BeautifulSoup(html, "html.parser")
        data = extract_json_ld(soup)
        assert data is None

    def test_json_ld_invalido_ignorado(self):
        html = """<html><head>
        <script type="application/ld+json">{ invalido json }</script>
        </head></html>"""
        soup = BeautifulSoup(html, "html.parser")
        data = extract_json_ld(soup)
        assert data is None

    def test_tipo_como_lista(self):
        html = """<html><head>
        <script type="application/ld+json">
        {"@type": ["Product", "Thing"], "name": "Arroz", "offers": {"price": "1.69"}}
        </script></head></html>"""
        soup = BeautifulSoup(html, "html.parser")
        data = extract_json_ld(soup)
        assert data is not None
        assert data["name"] == "Arroz"


# ---------------------------------------------------------------------------
# parse_product_html
# ---------------------------------------------------------------------------

class TestParseProductHtml:
    TS = "2026-04-18 10:00:00"
    URL = "https://www.auchan.pt/pt/alimentacao/mercearia/arroz/arroz-agulha/12345.html"

    def test_produto_com_json_ld(self, html_com_json_ld):
        record = parse_product_html(html_com_json_ld, self.URL, self.TS)
        assert record is not None
        assert record["nome"] == "Arroz Agulha Cigala 1kg"
        assert record["preco"] == "1.69"
        assert record["id_externo"] == "12345"
        assert record["ean"] == "5601234567890"
        assert record["marca"] == "Cigala"
        assert record["loja"] == "Auchan"

    def test_produto_sem_preco_devolve_none(self, html_sem_preco):
        record = parse_product_html(html_sem_preco, self.URL, self.TS)
        assert record is None

    def test_produto_sem_nome_devolve_none(self, html_sem_nome):
        url = "https://www.auchan.pt/pt/alimentacao/produto/99999.html"
        record = parse_product_html(html_sem_nome, url, self.TS)
        assert record is None

    def test_pid_extraido_da_url_quando_sem_sku(self, html_sem_json_ld):
        url = "https://www.auchan.pt/pt/frescos/leite/leite-mimosa/54321.html"
        record = parse_product_html(html_sem_json_ld, url, self.TS)
        if record is not None:
            assert record["id_externo"] == "54321"

    def test_schema_completo(self, html_com_json_ld):
        from scrapers.utils import OUTPUT_FIELDNAMES
        record = parse_product_html(html_com_json_ld, self.URL, self.TS)
        assert record is not None
        for campo in OUTPUT_FIELDNAMES:
            assert campo in record

    def test_url_preservada(self, html_com_json_ld):
        record = parse_product_html(html_com_json_ld, self.URL, self.TS)
        assert record is not None
        assert record["url"] == self.URL

    def test_categoria_derivada_do_url(self, html_com_json_ld):
        record = parse_product_html(html_com_json_ld, self.URL, self.TS)
        assert record is not None
        assert "alimentacao" in record["categoria"]


# ---------------------------------------------------------------------------
# parse_product_ajax
# ---------------------------------------------------------------------------

class TestParseProductAjax:
    TS = "2026-04-18 10:00:00"
    URL = "https://www.auchan.pt/pt/alimentacao/mercearia/arroz/arroz-agulha/12345.html"

    def _payload_completo(self):
        return {
            "productName": "Arroz Agulha Cigala 1kg",
            "id": "12345",
            "brand": "Cigala",
            "price": {
                "sales": {"value": 1.69, "formatted": "1,69 €"},
                "list":  None,
            },
            "unitPerPrice": "1,69€/kg",
            "images": {
                "large": [{"url": "/img/12345.jpg"}],
            },
        }

    def test_produto_completo(self):
        record = parse_product_ajax(self._payload_completo(), self.URL, self.TS)
        assert record is not None
        assert record["nome"] == "Arroz Agulha Cigala 1kg"
        assert record["id_externo"] == "12345"
        assert record["marca"] == "Cigala"
        assert record["preco"] == "1.69"
        assert record["loja"] == "Auchan"

    def test_sem_nome_devolve_none(self):
        data = {"id": "1", "price": {"sales": {"value": 1.69}}}
        assert parse_product_ajax(data, self.URL, self.TS) is None

    def test_preco_invalido_devolve_none(self):
        data = {"productName": "Arroz", "id": "1", "price": {"sales": {"value": 0}}}
        assert parse_product_ajax(data, self.URL, self.TS) is None

    def test_fallback_pid_da_url(self):
        data = {
            "productName": "Arroz Agulha",
            "id": "",  # ID vazio — deve usar URL
            "price": {"sales": {"value": 1.69}},
        }
        record = parse_product_ajax(data, self.URL, self.TS)
        assert record is not None
        assert record["id_externo"] == "12345"

    def test_sem_pid_devolve_none(self):
        url_sem_pid = "https://www.auchan.pt/pt/alimentacao/mercearia/"
        data = {
            "productName": "Arroz",
            "id": "",
            "price": {"sales": {"value": 1.69}},
        }
        assert parse_product_ajax(data, url_sem_pid, self.TS) is None

    def test_em_promocao_detectado(self):
        data = {
            "productName": "Azeite Gallo",
            "id": "99999",
            "price": {
                # isPromotion truthy é obrigatório para o scraper marcar promoção
                # (evita falsos positivos de preço de referência/PVPR).
                "isPromotion": 1,
                "sales": {"value": 4.49, "formatted": "4,49 €"},
                "list":  {"value": 5.99, "formatted": "5,99 €"},
            },
        }
        record = parse_product_ajax(data, self.URL, self.TS)
        assert record is not None
        assert record["preco_original"] == "5.99"
        assert record["preco"] == "4.49"

    def test_preco_original_igual_preco_ignorado(self):
        data = {
            "productName": "Arroz",
            "id": "1",
            "price": {
                "sales": {"value": 1.69},
                "list":  {"value": 1.69},
            },
        }
        record = parse_product_ajax(data, self.URL, self.TS)
        assert record is not None
        assert record["preco_original"] == ""

    def test_imagem_com_url_absoluto(self):
        data = {
            "productName": "Arroz",
            "id": "1",
            "price": {"sales": {"value": 1.69}},
            "images": {"large": [{"url": "https://cdn.auchan.pt/img/1.jpg"}]},
        }
        record = parse_product_ajax(data, self.URL, self.TS)
        assert record is not None
        assert record["imagem"].startswith("https://")

    def test_imagem_relativa_recebe_base_url(self):
        data = {
            "productName": "Arroz",
            "id": "1",
            "price": {"sales": {"value": 1.69}},
            "images": {"large": [{"url": "/img/produto.jpg"}]},
        }
        record = parse_product_ajax(data, self.URL, self.TS)
        assert record is not None
        assert record["imagem"].startswith(BASE_URL)


# ---------------------------------------------------------------------------
# fetch_product_ajax (com mock HTTP)
# ---------------------------------------------------------------------------

class TestFetchProductAjax:
    def test_pid_vazio_devolve_none(self):
        session = MagicMock()
        result = fetch_product_ajax(session, "")
        assert result is None
        session.get.assert_not_called()

    def test_resposta_html_devolve_none(self):
        mock_resp = MagicMock()
        mock_resp.headers = {"Content-Type": "text/html"}
        mock_resp.text = "<html><body>...</body></html>"
        mock_resp.raise_for_status.return_value = None

        session = MagicMock()
        session.get.return_value = mock_resp

        result = fetch_product_ajax(session, "12345")
        assert result is None

    def test_resposta_json_valida_devolve_dict(self):
        payload = {"productName": "Arroz", "id": "12345", "price": {"sales": {"value": 1.69}}}
        mock_resp = MagicMock()
        mock_resp.headers = {"Content-Type": "application/json"}
        mock_resp.text = json.dumps(payload)
        mock_resp.json.return_value = payload
        mock_resp.raise_for_status.return_value = None

        session = MagicMock()
        session.get.return_value = mock_resp

        result = fetch_product_ajax(session, "12345")
        assert result is not None
        assert result["productName"] == "Arroz"

    def test_resposta_json_sem_productname_devolve_none(self):
        payload = {"id": "12345", "price": {}}
        mock_resp = MagicMock()
        mock_resp.headers = {"Content-Type": "application/json"}
        mock_resp.text = json.dumps(payload)
        mock_resp.json.return_value = payload
        mock_resp.raise_for_status.return_value = None

        session = MagicMock()
        session.get.return_value = mock_resp

        result = fetch_product_ajax(session, "12345")
        assert result is None

    def test_excecao_http_devolve_none(self):
        import requests
        session = MagicMock()
        session.get.side_effect = requests.RequestException("timeout")

        result = fetch_product_ajax(session, "12345")
        assert result is None
