"""
Testes para models/geolocation.py.

Cobre as partes puras do módulo (sem BD):
  - haversine_km — distância great-circle
  - custo_deslocacao_euros — conversão distância → custo
  - resolver_custo_km — resolução de preset/valor
  - _bbox_para_raio — bounding box para filtragem SQL
  - PRESETS_CUSTO_KM — sanidade do dicionário de presets

Os testes de ``lojas_proximas`` e ``distancia_minima_por_insignia`` requerem
BD e ficam fora desta suite (mesmo padrão de ``test_ingest.py``).
"""

from __future__ import annotations

import math

import pytest

from models.geolocation import (
    DEFAULT_CUSTO_KM,
    EARTH_RADIUS_KM,
    KM_POR_GRAU_LAT,
    PRESETS_CUSTO_KM,
    _bbox_para_raio,
    custo_deslocacao_euros,
    haversine_km,
    resolver_custo_km,
)


# ---------------------------------------------------------------------------
# Constantes — sanidade dos valores numéricos
# ---------------------------------------------------------------------------

class TestConstants:
    def test_earth_radius_consistente_com_iugg(self):
        # 6371.0088 km é o valor médio recomendado pela IUGG
        assert math.isclose(EARTH_RADIUS_KM, 6371.0088, abs_tol=0.01)

    def test_km_por_grau_lat(self):
        # 1° latitude ≈ 111 km (definição clássica)
        assert math.isclose(KM_POR_GRAU_LAT, 111.0, abs_tol=0.5)

    def test_default_custo_km_e_equilibrado(self):
        assert DEFAULT_CUSTO_KM == PRESETS_CUSTO_KM["equilibrado"]["valor"]


# ---------------------------------------------------------------------------
# haversine_km
# ---------------------------------------------------------------------------

class TestHaversineKm:
    def test_distancia_de_um_ponto_a_si_proprio_e_zero(self):
        assert haversine_km(41.5, -8.4, 41.5, -8.4) == 0.0

    def test_braga_porto_aproximadamente_49km(self):
        """Distância Braga (UMinho Gualtar) → Porto (Boavista) é ~49 km.
        Validação contra calculadoras online geodésicas."""
        d = haversine_km(41.561, -8.397, 41.158, -8.629)
        assert math.isclose(d, 49.0, abs_tol=1.0)

    def test_lisboa_porto_aproximadamente_275km(self):
        """Lisboa → Porto: ~275 km (referência clássica)."""
        d = haversine_km(38.7223, -9.1393, 41.1496, -8.6109)
        assert math.isclose(d, 275.0, abs_tol=3.0)

    def test_simetria_da_distancia(self):
        d1 = haversine_km(41.5, -8.4, 38.7, -9.1)
        d2 = haversine_km(38.7, -9.1, 41.5, -8.4)
        assert math.isclose(d1, d2)

    def test_distancia_e_sempre_positiva(self):
        # Pontos aleatórios distintos
        for a, b, c, d in [(0, 0, 1, 1), (-10, 10, 10, -10), (41, -8, 42, -7)]:
            assert haversine_km(a, b, c, d) > 0

    def test_distancia_polo_a_polo_meia_circunferencia(self):
        """Polo norte → polo sul deve ser ~π · R = ~20015 km."""
        d = haversine_km(90, 0, -90, 0)
        assert math.isclose(d, math.pi * EARTH_RADIUS_KM, abs_tol=1.0)


# ---------------------------------------------------------------------------
# custo_deslocacao_euros
# ---------------------------------------------------------------------------

class TestCustoDeslocacaoEuros:
    def test_default_e_ida_e_volta(self):
        # 10 km × 0.20 €/km × 2 (ida-volta) = 4.0
        assert custo_deslocacao_euros(10.0) == 4.0

    def test_custo_km_personalizado(self):
        # 5 km × 0.36 × 2 = 3.6
        assert custo_deslocacao_euros(5.0, custo_km=0.36) == 3.6

    def test_ida_simples(self):
        assert custo_deslocacao_euros(10.0, ida_e_volta=False) == 2.0

    def test_distancia_zero_devolve_zero(self):
        assert custo_deslocacao_euros(0.0) == 0.0

    def test_arredonda_a_2_casas(self):
        # 3.333 km × 0.20 × 2 = 1.3332 → 1.33
        result = custo_deslocacao_euros(3.333, custo_km=0.20)
        assert result == 1.33

    def test_presets_dao_resultados_esperados_para_10km(self):
        # Tabela explícita dos resultados esperados — útil para o relatório
        esperados = {
            "so_combustivel": 2.4,   # 10 × 0.12 × 2
            "equilibrado":    4.0,   # 10 × 0.20 × 2
            "tarifa_at":      7.2,   # 10 × 0.36 × 2
        }
        for preset, expected in esperados.items():
            valor = PRESETS_CUSTO_KM[preset]["valor"]
            assert custo_deslocacao_euros(10.0, custo_km=valor) == expected


# ---------------------------------------------------------------------------
# resolver_custo_km
# ---------------------------------------------------------------------------

class TestResolverCustoKm:
    def test_none_devolve_default(self):
        assert resolver_custo_km(None) == DEFAULT_CUSTO_KM

    @pytest.mark.parametrize("preset", ["so_combustivel", "equilibrado", "tarifa_at"])
    def test_preset_valido(self, preset):
        assert resolver_custo_km(preset) == PRESETS_CUSTO_KM[preset]["valor"]

    def test_numero_direto(self):
        assert resolver_custo_km(0.18) == 0.18

    def test_string_numerica(self):
        assert resolver_custo_km("0.25") == 0.25

    def test_int_aceite(self):
        assert resolver_custo_km(1) == 1.0

    def test_preset_inexistente_levanta_value_error(self):
        with pytest.raises(ValueError, match="inválido"):
            resolver_custo_km("preset_que_nao_existe")

    def test_string_nao_numerica_levanta_value_error(self):
        with pytest.raises(ValueError):
            resolver_custo_km("abc")

    @pytest.mark.parametrize("negativo", [-0.5, -1])
    def test_valor_negativo_rejeitado(self, negativo):
        with pytest.raises(ValueError, match="negativo"):
            resolver_custo_km(negativo)

    @pytest.mark.parametrize("zero", [0, 0.0, "0"])
    def test_custo_zero_aceite(self, zero):
        # 0 €/km é válido: modela deslocação sem custo marginal (bicicleta,
        # carro de empresa). Ver resolver_custo_km em models/geolocation.py.
        assert resolver_custo_km(zero) == 0.0


# ---------------------------------------------------------------------------
# _bbox_para_raio
# ---------------------------------------------------------------------------

class TestBboxParaRaio:
    def test_bbox_simetrica_em_latitude(self):
        # Para raio R, |lat - lat_centro| = R/111 em ambos os lados
        lat_min, lat_max, _, _ = _bbox_para_raio(41.0, -8.0, raio_km=11.1)
        assert math.isclose(lat_max - 41.0, 41.0 - lat_min, abs_tol=1e-6)
        assert math.isclose(lat_max - lat_min, 2 * 11.1 / KM_POR_GRAU_LAT, abs_tol=1e-4)

    def test_bbox_corrige_longitude_pelo_cosseno_da_latitude(self):
        # A 41°N, cos(41°) ≈ 0.755 → 1° lon ≈ 84 km
        # Logo um raio de 84 km dá delta_lon ≈ 1°
        _, _, lon_min, lon_max = _bbox_para_raio(41.0, -8.0, raio_km=84.0)
        assert math.isclose(lon_max - lon_min, 2.0, abs_tol=0.05)

    def test_bbox_no_equador_lat_lon_iguais(self):
        # No equador, cos(0)=1 → delta_lat = delta_lon
        lat_min, lat_max, lon_min, lon_max = _bbox_para_raio(0.0, 0.0, raio_km=50.0)
        assert math.isclose(lat_max - lat_min, lon_max - lon_min, abs_tol=1e-6)

    def test_bbox_contem_o_centro(self):
        lat_min, lat_max, lon_min, lon_max = _bbox_para_raio(41.5, -8.4, raio_km=20.0)
        assert lat_min < 41.5 < lat_max
        assert lon_min < -8.4 < lon_max


# ---------------------------------------------------------------------------
# PRESETS_CUSTO_KM — sanidade da configuração
# ---------------------------------------------------------------------------

class TestPresetsCustoKm:
    def test_tres_presets_definidos(self):
        assert set(PRESETS_CUSTO_KM.keys()) == {"so_combustivel", "equilibrado", "tarifa_at"}

    def test_todos_os_presets_tem_campos_obrigatorios(self):
        for key, info in PRESETS_CUSTO_KM.items():
            assert "valor" in info, f"preset {key} sem 'valor'"
            assert "label" in info, f"preset {key} sem 'label'"
            assert "descricao" in info, f"preset {key} sem 'descricao'"

    def test_valores_sao_positivos(self):
        for info in PRESETS_CUSTO_KM.values():
            assert info["valor"] > 0

    def test_valores_ordenados_por_conservadorismo(self):
        """Os presets devem estar em ordem crescente: combustível < equilibrado < AT."""
        v_comb = PRESETS_CUSTO_KM["so_combustivel"]["valor"]
        v_eq   = PRESETS_CUSTO_KM["equilibrado"]["valor"]
        v_at   = PRESETS_CUSTO_KM["tarifa_at"]["valor"]
        assert v_comb < v_eq < v_at

    def test_tarifa_at_e_036(self):
        """A tarifa AT é fixada pela Portaria 1553-D/2008 — não muda sem alteração legal."""
        assert PRESETS_CUSTO_KM["tarifa_at"]["valor"] == 0.36
