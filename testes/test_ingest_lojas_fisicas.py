"""
Testes para scripts/ingest_lojas_fisicas.py.

Foca a validação de registos (que não depende de BD) e o carregamento de
ficheiros JSON. Os testes que requerem BD ficam fora — equivalente ao que
acontece com testes/test_ingest.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.ingest_lojas_fisicas import (
    CAMPOS_OBRIGATORIOS,
    carregar_ficheiros,
    validar_registo,
)


# ---------------------------------------------------------------------------
# validar_registo
# ---------------------------------------------------------------------------

class TestValidarRegisto:
    @pytest.fixture
    def loja_valida(self):
        return {
            "insignia": "Continente",
            "nome_loja": "Continente Braga",
            "latitude": 41.5409,
            "longitude": -8.4004,
            "external_id": "abc-123",
            "fonte": "scraper:continente",
            # opcionais
            "morada": "Rua X",
            "codigo_postal": "4700-001",
            "cidade": "Braga",
        }

    def test_loja_valida_passa(self, loja_valida):
        assert validar_registo(loja_valida) is None

    @pytest.mark.parametrize("campo", CAMPOS_OBRIGATORIOS)
    def test_campo_obrigatorio_em_falta_rejeitado(self, loja_valida, campo):
        del loja_valida[campo]
        erro = validar_registo(loja_valida)
        assert erro is not None
        assert campo in erro

    @pytest.mark.parametrize("campo", CAMPOS_OBRIGATORIOS)
    def test_campo_obrigatorio_vazio_rejeitado(self, loja_valida, campo):
        # 0 e 0.0 também são "vazios" pelo critério do not loja.get() —
        # validar_registo usa `not` para detectar isto. Strings vazias e None
        # são os casos mais realistas, mas o 0.0 também é rejeitado, o que
        # é aceitável: latitude/longitude exactamente 0 ficariam em pleno
        # Atlântico Sul, não fazem sentido para PT.
        loja_valida[campo] = "" if isinstance(loja_valida[campo], str) else 0
        erro = validar_registo(loja_valida)
        assert erro is not None

    def test_latitude_nao_numerica_rejeitada(self, loja_valida):
        loja_valida["latitude"] = "abc"
        assert "não numérica" in validar_registo(loja_valida)

    def test_latitude_fora_do_intervalo_global_rejeitada(self, loja_valida):
        loja_valida["latitude"] = 95.0
        erro = validar_registo(loja_valida)
        assert erro is not None
        assert "latitude" in erro

    def test_longitude_fora_do_intervalo_global_rejeitada(self, loja_valida):
        loja_valida["longitude"] = -200.0
        erro = validar_registo(loja_valida)
        assert erro is not None
        assert "longitude" in erro

    def test_latitude_string_numerica_aceite(self, loja_valida):
        """psycopg2 aceita string numérica para colunas NUMERIC, e validar_registo
        usa float() que também aceita. Sanidade: queremos que isto seja válido."""
        loja_valida["latitude"] = "41.5409"
        loja_valida["longitude"] = "-8.4004"
        assert validar_registo(loja_valida) is None


# ---------------------------------------------------------------------------
# carregar_ficheiros
# ---------------------------------------------------------------------------

class TestCarregarFicheiros:
    def test_carrega_um_ficheiro_json(self, tmp_path: Path):
        f = tmp_path / "lojas.json"
        f.write_text(json.dumps([
            {"insignia": "X", "nome_loja": "A", "latitude": 41.5, "longitude": -8.4},
            {"insignia": "X", "nome_loja": "B", "latitude": 38.7, "longitude": -9.1},
        ]))
        registos, caminhos = carregar_ficheiros([str(f)])
        assert len(registos) == 2
        assert caminhos == [str(f)]

    def test_carrega_glob_pattern(self, tmp_path: Path):
        (tmp_path / "lojas_a.json").write_text(json.dumps([{"insignia": "A"}]))
        (tmp_path / "lojas_b.json").write_text(json.dumps([{"insignia": "B"}]))
        registos, caminhos = carregar_ficheiros([str(tmp_path / "lojas_*.json")])
        assert len(registos) == 2
        assert len(caminhos) == 2

    def test_ficheiro_nao_lista_levanta_value_error(self, tmp_path: Path):
        f = tmp_path / "errado.json"
        f.write_text(json.dumps({"nao": "lista"}))  # objecto, não array
        with pytest.raises(ValueError, match="não contém uma lista"):
            carregar_ficheiros([str(f)])

    def test_nenhum_ficheiro_levanta_file_not_found(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            carregar_ficheiros([str(tmp_path / "inexistente_*.json")])

    def test_caminhos_sao_ordenados_e_unicos(self, tmp_path: Path):
        (tmp_path / "c.json").write_text(json.dumps([]))
        (tmp_path / "a.json").write_text(json.dumps([]))
        (tmp_path / "b.json").write_text(json.dumps([]))
        # Passar o mesmo glob duas vezes — devem ser deduplicados
        _, caminhos = carregar_ficheiros([
            str(tmp_path / "*.json"),
            str(tmp_path / "*.json"),
        ])
        assert caminhos == sorted(caminhos)
        assert len(caminhos) == 3
