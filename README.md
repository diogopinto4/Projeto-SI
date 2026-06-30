# ProjetoSI

Projeto da UC **Projeto em Sistemas Inteligentes** — perfil **Sistemas Inteligentes**  
Mestrado em Engenharia Informática — Universidade do Minho — 2025/2026

## Tema

Plataforma de análise de preços e variedade de produtos de supermercado.  
Recolha automática de preços de cadeias nacionais (Auchan, Continente, Pingo Doce), previsão por deep learning e recomendação de compras, implementada como um sistema multiagente SPADE com API REST.

## Arquitetura do sistema

```
┌──────────────────────────┐    ┌─────────────────────────────────────┐
│  Dashboard (Streamlit)   │    │        API REST (FastAPI)           │
│  http://localhost:8501   │    │    http://localhost:8000/docs       │
└────────────┬─────────────┘    └──────────────┬──────────────────────┘
             │ HTTP                             │ HTTP
             └──────────────┬──────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     UserInterfaceAgent                              │
│              (ponte HTTP ↔ XMPP / FIPA-ACL)                        │
└──────┬──────────────────┬──────────────────┬────────────────────────┘
       │ XMPP             │ XMPP             │ XMPP
       ▼                  ▼                  ▼
┌─────────────┐  ┌──────────────────┐  ┌─────────────────┐  ┌────────────────┐
│Recommendation│  │  PredictionAgent │  │  DatabaseAgent  │  │ LocationAgent  │
│   Agent      │  │  (LSTM + MC)     │  │  (PostgreSQL)   │  │ (GPS, haversine)│
└─────────────┘  └──────────────────┘  └────────┬────────┘  └────────────────┘
                                                 │
┌────────────────────────────────────────────────┘
│
│  Ingestão / Coordenação
│
├─→ OrchestratorAgent  (coordena pipeline pós-ingestão)
├─→ MonitorAgent       (alertas de preço + deteção de anomalias)
├─→ AuchanScraper      (scraping periódico do Auchan via sitemap + AJAX SFCC)
├─→ ContinenteScraper  (scraping periódico do Continente Online)
└─→ PingoDoceScraper   (scraping periódico do Pingo Doce via sitemap)

Infraestrutura: PostgreSQL 16 + Openfire (XMPP) via Docker
```

## Agentes SPADE

| Agente | JID (default) | Responsabilidade |
|---|---|---|
| `OrchestratorAgent` | `orchestrator_agent@localhost` | Coordena o pipeline após cada ingestão; aciona retreino do LSTM |
| `DatabaseAgent` | `database_agent@localhost` | Ingestão de produtos na BD; responde a consultas de histórico e preços |
| `AuchanScraper` | `auchan_scraper@localhost` | Scraping periódico do Auchan Portugal (sitemap XML + endpoint AJAX SFCC) |
| `ContinenteScraper` | `continente_scraper@localhost` | Scraping periódico do Continente Online (categorias por paginação) |
| `PingoDoceScraper` | `pingo_doce_scraper@localhost` | Scraping periódico do Pingo Doce (sitemap XML, produto a produto) |
| `PredictionAgent` | `prediction_agent@localhost` | Previsão LSTM global + Monte Carlo Dropout para intervalos de confiança |
| `RecommendationAgent` | `recommendation_agent@localhost` | Pesquisa, melhor loja, otimização de lista, momento de compra |
| `LocationAgent` | `location_agent@localhost` | Geolocalização (haversine), lojas físicas próximas, custo de deslocação |
| `MonitorAgent` | `monitor_agent@localhost` | Monitor de mudanças de preço e deteção de anomalias (IQR) |
| `UserInterfaceAgent` | `ui_agent@localhost` | Ponte entre a API REST (FastAPI/uvicorn) e o sistema multiagente |

## Estrutura de ficheiros

```
ProjetoSI/
├── main.py                          # Ponto de entrada — arranca todos os agentes + API
├── agents/
│   ├── __init__.py
│   ├── AuchanScraper.py             # Agente scraper do Auchan (sitemap + AJAX SFCC)
│   ├── ContinenteScraper.py         # Agente scraper do Continente Online
│   ├── PingoDoceScraper.py          # Agente scraper do Pingo Doce (sitemap)
│   ├── DatabaseAgent.py             # Ingestão na BD + consultas
│   ├── PredictionAgent.py           # Previsão LSTM + retreino periódico
│   ├── RecommendationAgent.py       # Pesquisa, comparação e otimização de compras
│   ├── LocationAgent.py             # Geolocalização + custo de deslocação (haversine)
│   ├── MonitorAgent.py              # Monitor de preços + deteção de anomalias
│   ├── OrchestratorAgent.py         # Coordenação do pipeline pós-ingestão
│   └── UserInterfaceAgent.py        # Ponte HTTP ↔ XMPP
├── api/
│   ├── __init__.py
│   └── app.py                       # Aplicação FastAPI (9 endpoints REST)
├── models/
│   ├── price_predictor.py           # Modelo LSTM global + Monte Carlo Dropout
│   ├── gbm_predictor.py             # Modelo gradient boosting (alternativa ao LSTM)
│   ├── recommender.py               # Lógica de recomendação (pg_trgm + opcional GPS)
│   ├── geolocation.py               # Haversine + lojas próximas + custo de deslocação
│   └── baselines.py                 # Baselines de comparação (Naive, MA, ARIMA)
├── scrapers/
│   ├── auchan_scraper.py            # Scraper standalone do Auchan (produtos)
│   ├── continente_scraper.py        # Scraper standalone do Continente (produtos)
│   ├── pingo_doce_scraper.py        # Scraper standalone do Pingo Doce (produtos)
│   ├── lojas_fisicas_scraper.py     # Scraper de lojas físicas das 3 cadeias (coordenadas GPS)
│   └── utils.py                     # Funções partilhadas (normalização, HTTP, I/O)
├── scripts/
│   ├── ingest.py                    # Ingestão de JSON de produtos para PostgreSQL
│   ├── ingest_lojas_fisicas.py      # Ingestão de JSON de lojas físicas para PostgreSQL
│   ├── build_forecasting_dataset.py # Dataset temporal para o LSTM
│   ├── anomaly_detector.py          # Deteção de preços suspeitos (IQR)
│   ├── price_monitor.py             # Monitor e alertas de preço
│   ├── run_pipeline.py              # Pipeline completo (ingestão → dataset → monitor)
│   ├── scheduler.py                 # Agendador periódico (APScheduler)
│   ├── setup_openfire.py            # Criação automática das contas XMPP no Openfire
│   ├── db_config.py                 # Configuração centralizada da BD
│   ├── data_diagnostics.py          # Diagnóstico rápido de cobertura
│   ├── audit_data_quality.py        # Auditoria detalhada de qualidade
│   ├── relatorio_dados.py           # Relatório académico (8 secções + 5 gráficos)
│   └── comparar_modelos.py          # Comparação LSTM vs Naive/MA/ARIMA (win rate + boxplot)
├── testes/
│   ├── conftest.py                  # Fixtures partilhadas (pytest) + setup BD de teste
│   ├── test_scrapers_utils.py       # Testes de scrapers/utils.py
│   ├── test_auchan_scraper.py       # Testes do scraper do Auchan
│   ├── test_pingo_doce_scrapper.py  # Testes do scraper do Pingo Doce (sitemap + produto)
│   ├── test_lojas_fisicas_scraper.py# Testes do scraper de lojas físicas (sem rede)
│   ├── test_ingest.py               # Testes de scripts/ingest.py (puros + dry-run)
│   ├── test_ingest_lojas_fisicas.py # Testes de scripts/ingest_lojas_fisicas.py (puros)
│   ├── test_ingest_lojas_fisicas_bd.py # Testes BD: UPSERT, idempotência, trigger
│   ├── test_build_dataset.py        # Testes de scripts/build_forecasting_dataset.py
│   ├── test_price_predictor.py      # Testes de models/price_predictor.py
│   ├── test_baselines.py            # Testes de models/baselines.py (Naive, MA, ARIMA)
│   ├── test_geolocation.py          # Testes de models/geolocation.py (haversine, presets)
│   ├── test_geolocation_bd.py       # Testes BD: lojas_proximas, distancia_minima
│   ├── test_geolocation_cache.py    # Testes do cache TTL de lojas_proximas
│   ├── test_recommender_utils.py    # Testes de models/recommender.py (com mocks)
│   ├── test_recommender_bd.py       # Testes BD: pesquisar, otimizar, otimizar_geo
│   ├── test_recommender_geo.py      # Testes da função GPS-aware (com mocks)
│   ├── test_recommender_geo_multi_bd.py # Testes BD: otimização multi-loja (rota triangular)
│   ├── test_produto_perto_de_mim_bd.py  # Testes BD: produto perto de mim (preço + deslocação)
│   ├── test_price_monitor_bd.py     # Testes BD: detetar_mudancas_preco (window functions)
│   └── test_anomaly_detector_bd.py  # Testes BD: carregar_historico + detetar_anomalias
├── sql/
│   └── schema.sql                   # Schema PostgreSQL (tabelas, índices, triggers)
├── data/
│   ├── alertas/                     # CSVs gerados pelo MonitorAgent
│   └── generated/                   # Dataset de forecasting gerado pelo pipeline
├── docs/
│   └── decisions.md                 # Decisões arquiteturais e algorítmicas (18 secções, p/ relatório)
├── docker-compose.yml               # PostgreSQL 16 + Openfire (XMPP)
├── requirements.txt
└── .env.example
```

## Setup

```bash
# 1. Ambiente virtual
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. Dependências
pip install -r requirements.txt

# 3. Variáveis de ambiente
cp .env.example .env             # edita com as credenciais se necessário

# 4. Arrancar PostgreSQL + Openfire (XMPP)
docker compose up -d
```

### Configuração do Openfire (só na primeira vez)

O Openfire requer um setup inicial via browser:

1. Acede a `http://localhost:9090` e completa o assistente de configuração
   - Base de dados: seleciona **"Embedded Database"** (mais simples)
   - Cria a conta de administrador (guarda as credenciais — precisas no passo seguinte)
2. Instala o plugin REST API:
   **Plugins → Available Plugins → REST API → Install**

```bash
# 5. Criar as contas XMPP dos agentes automaticamente
python scripts/setup_openfire.py

# Se as credenciais de admin não forem as default (admin/admin), passa-as como argumento:
python scripts/setup_openfire.py --admin <user> --password <pass>

# 6. Seed inicial das lojas físicas (~1060 lojas reais a nível nacional)
#    Usado pela funcionalidade de custo de deslocação no agente de recomendação.
python scrapers/lojas_fisicas_scraper.py    # recolhe das 3 cadeias via APIs públicas
python scripts/ingest_lojas_fisicas.py      # ingere o JSON mais recente na BD

# 7. Arrancar o sistema
python main.py
```

> **Notas:**
> - Os passos 5 e 6 são idempotentes — podes correr várias vezes sem problemas.
> - O passo 5: se as contas XMPP já existirem, o script reporta `[=]` e continua.
> - O passo 6: re-executar actualiza os registos existentes via UPSERT
>   (`ON CONFLICT (insignia, external_id) DO UPDATE`), não cria duplicados.
>   Útil para refrescar coordenadas/horários periodicamente.

## Uso — Sistema multiagente

### Arranque básico

```bash
# Todas as funcionalidades — scraping ativo dos 3 supermercados por defeito:
#   Auchan: 500 produtos/ciclo | Pingo Doce: 500 produtos/ciclo | Continente: mercearia+bebidas
python main.py

# Retreino automático do LSTM após cada ingestão
python main.py --retreinar

# Teste rápido sem modelos de IA
python main.py --sem-predicao --sem-recomendacao --auchan-limite 50 --pingo-doce-limite 50

# Limites personalizados por supermercado
python main.py --auchan-limite 200 --pingo-doce-limite 200 --continente-categorias mercearia

# Sem servidor HTTP (só agentes SPADE)
python main.py --sem-api

# Porta HTTP personalizada
python main.py --api-porta 9000
```

### Flags disponíveis

| Flag | Descrição | Default |
|---|---|---|
| `--auchan-categoria` | Categoria Auchan a recolher (ex: `alimentacao`, `bebidas`) | alimentacao |
| `--auchan-subcategoria` | Subcategoria Auchan opcional (ex: `arroz`) | — |
| `--auchan-limite` | Limite de produtos Auchan (útil para testes) | — |
| `--auchan-periodo` | Período de scraping Auchan (segundos) | 21600 (6h) |
| `--continente-categorias` | Categorias a recolher (`mercearia`, `bebidas`) | nenhuma (inativo sem config) |
| `--continente-queries` | Termos de pesquisa separados por vírgula | — |
| `--continente-periodo` | Período de scraping (segundos) | 21600 (6h) |
| `--pingo-doce-categoria` | Categoria de produto | mercearia |
| `--pingo-doce-subcategoria` | Subcategoria (ex: arroz, conservas) | — |
| `--pingo-doce-limite` | Limite de produtos (útil para testes) | — |
| `--pingo-doce-periodo` | Período de scraping (segundos) | 21600 (6h) |
| `--monitor-periodo` | Período de monitorização (segundos) | 21600 (6h) |
| `--sem-scrapers` | Não iniciar nenhum scraper (consulta dados já existentes) | — |
| `--retreinar` | Retreinar o LSTM após cada ingestão | não |
| `--sem-predicao` | Desativar PredictionAgent | — |
| `--sem-recomendacao` | Desativar RecommendationAgent | — |
| `--sem-localizacao` | Desativar LocationAgent (sem feature de custo de deslocação) | — |
| `--sem-api` | Desativar API REST e UserInterfaceAgent | — |
| `--api-porta` | Porta do servidor HTTP | 8000 |

## Dashboard

Interface visual com 7 separadores: Pesquisa & Comparação, Lista de Compras, Lista com Localização, Previsão de Preços, Histórico de Preços, Preços por Loja e Validação (back-test do recomendador vs baselines).

**Pré-requisito:** o sistema multiagente tem de estar a correr (`python main.py`) para que o dashboard consiga comunicar com a API.

```bash
streamlit run dashboard.py
```

Abre automaticamente em **http://localhost:8501**.

## API REST

Documentação interativa disponível em **http://localhost:8000/docs** após arrancar o sistema.

### Endpoints

| Método | Endpoint | Descrição |
|---|---|---|
| `GET` | `/saude` | Estado da API e JIDs dos agentes ligados (versão leve) |
| `GET` | `/saude/completo` | Diagnóstico completo (BD, modelo LSTM, dataset, lojas físicas) |
| `GET` | `/produtos/pesquisar?termo=&limite=` | Pesquisa por palavra-chave (pg_trgm) |
| `GET` | `/produtos/melhor-loja?nome=&top_n=` | Melhores preços para um produto entre lojas |
| `POST` | `/compras/otimizar` | Otimizar lista de compras entre lojas |
| `GET` | `/compras/momento/{produto_id}` | Comprar agora ou esperar? (LSTM + Monte Carlo) |
| `GET` | `/previsao/{produto_id}` | Previsão determinista de preço (LSTM) |
| `GET` | `/previsao/{produto_id}/incerteza` | Previsão com intervalos de confiança (MC Dropout) |
| `GET` | `/historico/{produto_id}` | Histórico de preços nos últimos N dias |
| `GET` | `/lojas/{insignia}/precos` | Preços atuais de todos os produtos de uma loja |
| `GET` | `/lojas-fisicas/proximas?lat=&lon=&raio_km=` | Lojas físicas mais próximas de uma localização GPS |
| `GET` | `/produtos/perto-de-mim?nome=&lat=&lon=&` | Produto ordenado por custo total (preço + deslocação) em todas as cadeias |
| `POST` | `/compras/otimizar-geo` | Otimizar lista incluindo **custo de deslocação** (flag `multi_loja` opcional avalia divisão entre 2 cadeias) |
| `GET` | `/custo-deslocacao?distancia_km=&custo_km=` | Calcular custo monetário de uma deslocação |
| `GET` | `/custo-deslocacao/presets` | Listar presets de €/km (combustível / equilibrado / tarifa AT) |

### Exemplo — otimizar lista de compras

```bash
curl -X POST http://localhost:8000/compras/otimizar \
     -H "Content-Type: application/json" \
     -d '{"lista": ["arroz agulha", "azeite virgem extra", "atum natural"]}'
```

### Exemplo — previsão de preço

```bash
# Previsão dos próximos 7 dias para o produto com id_produto_loja=29
curl http://localhost:8000/previsao/29

# Com intervalos de confiança (100 simulações Monte Carlo)
curl "http://localhost:8000/previsao/29/incerteza?amostras=100"
```

## Uso — Scripts standalone

Os scripts em `scripts/` e `models/` podem ser usados independentemente do sistema multiagente.

### Scraping manual

```bash
python3 scrapers/continente_scraper.py --categorias mercearia --formato json
python3 scrapers/pingo_doce_scraper.py --categoria mercearia --formato json
python3 scrapers/pingo_doce_scraper.py --categoria mercearia --subcategoria arroz --limite 50
```

### Ingestão

```bash
python3 scripts/ingest.py --input "scrapers/output/*.json"
python3 scripts/ingest.py --input "scrapers/output/*.json" --dry-run
```

### Pipeline completo

```bash
python3 scripts/run_pipeline.py --input "scrapers/output/*.json"
python3 scripts/run_pipeline.py --input "scrapers/output/*.json" --retreinar
```

### Diagnóstico e auditoria

```bash
python3 scripts/data_diagnostics.py
python3 scripts/audit_data_quality.py
python3 scripts/anomaly_detector.py --desvios 4.0 --variacao 0.4 --output data/anomalias.csv

# Relatório completo (markdown + 5 gráficos) — pronto para copiar para o relatório académico
python3 scripts/relatorio_dados.py
# Output em data/relatorio/resumo.md + PNGs
```

### Previsão de preços (LSTM)

```bash
python3 scripts/build_forecasting_dataset.py
python3 models/price_predictor.py --treinar
python3 models/price_predictor.py --avaliar
python3 models/price_predictor.py --prever --produto-id 1 --horizonte 7
python3 models/price_predictor.py --prever --mc --produto-id 1 --horizonte 7

# Modelo alternativo — gradient boosting (HistGradientBoostingRegressor)
python3 models/gbm_predictor.py                    # avalia em todos os produtos
python3 models/gbm_predictor.py --importancia      # + importância das features

# Comparação de todos os modelos (Naive, MA 3d/7d, ARIMA, gradient boosting, LSTM)
python3 scripts/comparar_modelos.py
# Output: data/relatorio/comparacao_modelos.md + boxplot_rmse_modelos.png + win_rate_modelos.png
```

### Recomendações

```bash
python3 models/recommender.py --pesquisar "arroz"
python3 models/recommender.py --melhor-loja "arroz agulha"
python3 models/recommender.py --lista "arroz agulha,azeite virgem extra,atum natural"
python3 models/recommender.py --momento --produto-id 1

# Otimização com GPS (custo de deslocação)
python3 models/recommender.py --lista "arroz,azeite,atum" --lat 41.561 --lon -8.397
python3 models/recommender.py --lista "arroz,azeite,atum" --lat 41.561 --lon -8.397 --custo-km tarifa_at

# Multi-loja: avaliar split entre 2 cadeias (rota triangular)
python3 models/recommender.py --lista "arroz,azeite,atum" --lat 41.561 --lon -8.397 --multi-loja
```

## Testes

A suite tem **~570 testes** (unitários + integração com BD + comparação de modelos).

### Pré-requisito — BD de teste

Os testes que tocam na BD usam uma **base de dados separada** (`products_db_test`)
para garantir isolamento total face aos dados de produção. Setup (1 vez):

```bash
docker exec products-db psql -U postgres -c "CREATE DATABASE products_db_test;"
docker exec -i products-db psql -U postgres -d products_db_test < sql/schema.sql
```

A fixture do `conftest.py` define `DB_NAME=products_db_test` antes de qualquer
import, e o `_verificar_db_teste` (autouse) confirma o redireccionamento — testes
**nunca** escrevem na BD de produção.

### Correr testes

```bash
# Suite completa
pytest testes/ -q

# Apenas testes BD (ficheiros _bd.py)
pytest testes/test_*_bd.py -v

# Apenas testes puros (sem BD) — corre em ambientes sem Postgres
pytest testes/ --ignore-glob='testes/test_*_bd.py' -q

# Com cobertura (requer pytest-cov)
pytest testes/ --cov=models --cov=scripts --cov=scrapers --cov=agents
```

### Categorias de testes

| Categoria | Ficheiros | Conta |
|---|---|---|
| Scrapers (sem rede, com HTML/JSON mock) | `test_*_scraper.py`, `test_scrapers_utils.py` | ~150 |
| Ingestão (puros, com dry-run) | `test_ingest.py`, `test_ingest_lojas_fisicas.py` | ~135 |
| Dataset + modelo LSTM (com DataFrames sintéticos) | `test_build_dataset.py`, `test_price_predictor.py` | ~80 |
| Recommender + geolocation (puros + com mocks) | `test_recommender_*.py`, `test_geolocation.py`, `test_baselines.py` | ~100 |
| **Integração com BD** | `test_*_bd.py` (7 ficheiros) | **74** |
