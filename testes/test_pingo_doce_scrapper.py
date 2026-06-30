"""
Testes para scrapers/pingo_doce_scraper.py.

Cobre os fallbacks críticos de parsing sem fazer pedidos de rede.
"""

from __future__ import annotations

from scrapers.pingo_doce_scraper import extract_product_id, parse_product_html


class TestExtractProductId:
    def test_sku_do_json_ld_tem_prioridade(self):
        assert extract_product_id({"sku": "12345"}, "https://example.test/produto.html") == "12345"

    def test_sufixo_numerico_da_url(self):
        url = "https://www.pingodoce.pt/home/produtos/mercearia/sal-fino-1805.html"
        assert extract_product_id(None, url) == "1805"

    def test_url_sem_sufixo_numerico_recebe_id_estavel(self):
        url = "https://www.pingodoce.pt/home/produtos/mercearia/arroz-carolino.html"
        primeiro = extract_product_id(None, url)
        segundo = extract_product_id(None, url)
        assert primeiro.startswith("pingo-url-")
        assert primeiro == segundo


class TestParseProductHtml:
    def test_produto_sem_sku_usa_id_estavel_por_url(self):
        html = """
        <html>
          <head>
            <script type="application/ld+json">
              {
                "@context": "https://schema.org",
                "@type": "Product",
                "name": "Arroz Carolino Pingo Doce 1kg",
                "brand": {"@type": "Brand", "name": "Pingo Doce"},
                "offers": {"@type": "Offer", "price": "1.29"}
              }
            </script>
          </head>
          <body>
            <h1>Arroz Carolino Pingo Doce 1kg</h1>
          </body>
        </html>
        """
        url = "https://www.pingodoce.pt/home/produtos/mercearia/arroz-carolino.html"
        record = parse_product_html(html, url, "2026-04-18 10:00:00")

        assert record is not None
        assert record["id_externo"].startswith("pingo-url-")
        assert record["loja"] == "Pingo Doce"