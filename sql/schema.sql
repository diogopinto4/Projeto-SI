-- =============================================================================
-- schema.sql — Esquema da base de dados PostgreSQL do ProjetoSI
-- =============================================================================
--
-- Modelo de dados para análise de preços de supermercados.
-- Suporta múltiplas cadeias (Continente, Pingo Doce) e múltiplos canais.
--
-- Hierarquia de entidades:
--
--   lojas
--     └─ produtos_loja  (produto como aparece num site específico)
--           ├─ produtos_mestre  (identidade canónica, independente da loja)
--           ├─ precos_atuais    (último preço observado, 1 linha por produto-loja)
--           └─ historico_precos (todas as observações anteriores)
--
-- Como executar (reinício total da BD):
--
--   psql -U postgres -d products_db -f sql/schema.sql
--
-- AVISO: Este script faz DROP CASCADE em todas as tabelas antes de as recriar.
--        Todos os dados existentes serão eliminados. Usar apenas em ambiente
--        de desenvolvimento ou para reset controlado.
-- =============================================================================


-- =============================================================================
-- Limpeza (reset total)
-- =============================================================================
-- Ordem inversa às dependências de FK para evitar erros de referência.

DROP TABLE IF EXISTS historico_precos CASCADE;
DROP TABLE IF EXISTS precos_atuais    CASCADE;
DROP TABLE IF EXISTS produtos_loja    CASCADE;
DROP TABLE IF EXISTS produtos_mestre  CASCADE;
DROP TABLE IF EXISTS lojas            CASCADE;
DROP TABLE IF EXISTS lojas_fisicas    CASCADE;


-- =============================================================================
-- Tabela: lojas
-- =============================================================================
-- Representa uma cadeia de supermercado num canal específico.
-- Uma mesma insígnia pode ter múltiplos registos se operar em canais distintos
-- (ex: Continente online vs. Continente físico).

CREATE TABLE lojas (
    id_loja          SERIAL          PRIMARY KEY,
    insignia         VARCHAR(100)    NOT NULL,
    formato_loja     VARCHAR(50)     NOT NULL DEFAULT 'Online',
    localizacao      VARCHAR(255)    NOT NULL DEFAULT 'Nacional',
    canal            VARCHAR(50)     NOT NULL DEFAULT 'online',
    data_criacao     TIMESTAMPTZ     NOT NULL DEFAULT CURRENT_TIMESTAMP,
    data_atualizacao TIMESTAMPTZ     NOT NULL DEFAULT CURRENT_TIMESTAMP,

    -- Garante que não existem duplicados para a mesma combinação loja + canal
    UNIQUE (insignia, formato_loja, localizacao, canal)
);

COMMENT ON TABLE  lojas                  IS 'Cadeias de supermercado e respectivos canais de venda.';
COMMENT ON COLUMN lojas.insignia         IS 'Nome da cadeia (ex: "Continente", "Pingo Doce").';
COMMENT ON COLUMN lojas.formato_loja     IS 'Tipo de estabelecimento (ex: "Online", "Hipermercado").';
COMMENT ON COLUMN lojas.localizacao      IS 'Âmbito geográfico (ex: "Nacional", "Porto").';
COMMENT ON COLUMN lojas.canal            IS 'Canal de venda: "online" ou "fisico".';
COMMENT ON COLUMN lojas.data_atualizacao IS 'Actualizado automaticamente pelo trigger trg_lojas_data_atualizacao.';


-- =============================================================================
-- Tabela: produtos_mestre
-- =============================================================================
-- Identidade canónica de um produto, independente da loja onde é vendido.
-- O mesmo produto físico (ex: "Arroz Agulha Bom Sucesso 1kg") pode aparecer
-- com nomes e SKUs diferentes em lojas distintas — todos apontam para o mesmo
-- produto_mestre através de produtos_loja.
--
-- A chave_mestre é determinística: construída a partir do EAN (se disponível)
-- ou de um hash semântico baseado em nome + quantidade + marca (ingest.py).
-- Isto permite identificar o mesmo produto entre scrapers sem intervenção manual.

CREATE TABLE produtos_mestre (
    id_produto_mestre SERIAL       PRIMARY KEY,
    chave_mestre      TEXT         NOT NULL UNIQUE,
    ean               VARCHAR(50),
    nome_padronizado  TEXT         NOT NULL,
    marca             VARCHAR(255),
    categoria_geral   VARCHAR(255),
    quantidade_valor  NUMERIC(12,3),
    quantidade_unidade VARCHAR(20),
    data_criacao      TIMESTAMPTZ  NOT NULL DEFAULT CURRENT_TIMESTAMP
);

COMMENT ON TABLE  produtos_mestre                    IS 'Identidade canónica de cada produto, partilhada entre lojas.';
COMMENT ON COLUMN produtos_mestre.chave_mestre       IS 'Chave determinística: "ean:{EAN}" ou hash semântico nome+qtd+marca. Gerada em scripts/ingest.py.';
COMMENT ON COLUMN produtos_mestre.ean                IS 'Código de barras EAN/GTIN (NULL se não disponível no scraping).';
COMMENT ON COLUMN produtos_mestre.nome_padronizado   IS 'Nome normalizado e limpo de caracteres especiais. Indexado com GIN/trgm para pesquisa rápida.';
COMMENT ON COLUMN produtos_mestre.quantidade_valor   IS 'Quantidade numérica extraída do nome (ex: 1.0 para "1kg"). NULL se não detectável.';
COMMENT ON COLUMN produtos_mestre.quantidade_unidade IS 'Unidade de medida normalizada (ex: "kg", "l", "un").';


-- =============================================================================
-- Tabela: produtos_loja
-- =============================================================================
-- Registo de um produto específico numa loja específica.
-- Guarda o SKU da loja, o nome original (antes de padronização), URLs e
-- informação de pack/embalagem quando o produto é vendido em múltiplas unidades.

CREATE TABLE produtos_loja (
    id_produto_loja        SERIAL          PRIMARY KEY,
    id_produto_mestre      INTEGER         NOT NULL
                               REFERENCES produtos_mestre(id_produto_mestre) ON DELETE CASCADE,
    id_loja                INTEGER         NOT NULL
                               REFERENCES lojas(id_loja) ON DELETE CASCADE,
    sku_loja               VARCHAR(100)    NOT NULL,
    nome_na_loja           TEXT            NOT NULL,
    categoria_loja         TEXT,
    url_produto            TEXT,
    url_imagem             TEXT,
    quantidade_valor       NUMERIC(12,3),
    quantidade_unidade     VARCHAR(20),
    -- Campos para produtos vendidos em pack (ex: "6x200ml")
    multiplicador_pack     INTEGER,
    unidade_base_pack      NUMERIC(12,3),
    unidade_medida_pack    VARCHAR(20),
    data_criacao           TIMESTAMPTZ     NOT NULL DEFAULT CURRENT_TIMESTAMP,
    data_ultima_observacao TIMESTAMPTZ     NOT NULL DEFAULT CURRENT_TIMESTAMP,

    -- Um SKU é único dentro de cada loja
    UNIQUE (id_loja, sku_loja)
);

COMMENT ON TABLE  produtos_loja                       IS 'Produto tal como aparece num site/loja específico, com o seu SKU e nome original.';
COMMENT ON COLUMN produtos_loja.sku_loja              IS 'Identificador externo do produto na loja (id numérico, slug, etc.).';
COMMENT ON COLUMN produtos_loja.nome_na_loja          IS 'Nome original do produto no site da loja (antes de padronização).';
COMMENT ON COLUMN produtos_loja.multiplicador_pack    IS 'Número de unidades no pack (ex: 6 para "6x200ml"). NULL se produto individual.';
COMMENT ON COLUMN produtos_loja.unidade_base_pack     IS 'Quantidade por unidade dentro do pack (ex: 0.2 para "6x200ml").';
COMMENT ON COLUMN produtos_loja.unidade_medida_pack   IS 'Unidade da quantidade base do pack (ex: "l" para "6x200ml").';
COMMENT ON COLUMN produtos_loja.data_ultima_observacao IS 'Atualizado em cada ingestão — permite detetar produtos descontinuados.';


-- =============================================================================
-- Tabela: precos_atuais
-- =============================================================================
-- Armazena o preço mais recente de cada produto-loja.
-- Usa id_produto_loja como chave primária: garante exactamente uma linha
-- por produto-loja, actualizada por UPSERT em cada ciclo de ingestão.
-- Optimizada para queries de comparação de preços em tempo real.

CREATE TABLE precos_atuais (
    id_produto_loja          INTEGER         PRIMARY KEY
                                 REFERENCES produtos_loja(id_produto_loja) ON DELETE CASCADE,
    preco_atual              NUMERIC(10,2)   NOT NULL CHECK (preco_atual > 0),
    preco_original           NUMERIC(10,2),
    preco_unitario_valor     NUMERIC(10,4),
    preco_unitario_unidade   VARCHAR(20),
    em_promocao              BOOLEAN         NOT NULL DEFAULT FALSE,
    data_recolha             TIMESTAMPTZ     NOT NULL,
    agente_origem            VARCHAR(100),
    data_ultima_atualizacao  TIMESTAMPTZ     NOT NULL DEFAULT CURRENT_TIMESTAMP
);

COMMENT ON TABLE  precos_atuais                          IS 'Preço mais recente por produto-loja. Uma linha por produto-loja, actualizada por UPSERT.';
COMMENT ON COLUMN precos_atuais.preco_atual              IS 'Preço actual em EUR. CHECK garante valor positivo.';
COMMENT ON COLUMN precos_atuais.preco_original           IS 'Preço antes de desconto (PVPR). NULL se não estiver em promoção.';
COMMENT ON COLUMN precos_atuais.preco_unitario_valor     IS 'Preço por unidade de medida (ex: 1.69 para "1.69€/kg").';
COMMENT ON COLUMN precos_atuais.preco_unitario_unidade   IS 'Unidade do preço unitário (ex: "kg", "l", "un").';
COMMENT ON COLUMN precos_atuais.agente_origem            IS 'Nome do scraper que recolheu os dados (ex: "Continente", "Pingo Doce").';
COMMENT ON COLUMN precos_atuais.data_ultima_atualizacao  IS 'Actualizado automaticamente pelo trigger trg_precos_atuais_data_atualizacao.';


-- =============================================================================
-- Tabela: historico_precos
-- =============================================================================
-- Registo imutável de todas as observações de preço ao longo do tempo.
-- Cada scraping bem-sucedido insere uma nova linha — nunca actualiza.
-- É a fonte de dados para o dataset de forecasting e para o monitor de preços.
--
-- O índice único uq_historico_produto_data_origem evita duplicados quando
-- o mesmo scraper é executado mais do que uma vez no mesmo timestamp.

CREATE TABLE historico_precos (
    id_historico           SERIAL          PRIMARY KEY,
    id_produto_loja        INTEGER         NOT NULL
                               REFERENCES produtos_loja(id_produto_loja) ON DELETE CASCADE,
    preco_atual            NUMERIC(10,2)   NOT NULL CHECK (preco_atual > 0),
    preco_original         NUMERIC(10,2),
    preco_unitario_valor   NUMERIC(10,4),
    preco_unitario_unidade VARCHAR(20),
    em_promocao            BOOLEAN         NOT NULL,
    data_recolha           TIMESTAMPTZ     NOT NULL,
    agente_origem          VARCHAR(100),
    data_registo_historico TIMESTAMPTZ     NOT NULL DEFAULT CURRENT_TIMESTAMP
);

COMMENT ON TABLE  historico_precos                        IS 'Série temporal de preços — append-only, nunca actualizado. Base do LSTM e monitor de preços.';
COMMENT ON COLUMN historico_precos.data_recolha           IS 'Timestamp da recolha pelo scraper (UTC). Usado para agrupamento diário no dataset de forecasting.';
COMMENT ON COLUMN historico_precos.data_registo_historico IS 'Timestamp de inserção na BD. Distinto de data_recolha para auditoria.';
COMMENT ON COLUMN historico_precos.agente_origem          IS 'Identificador do scraper (ex: "Continente"). Incluído no índice único para evitar duplicados por scraper.';


-- =============================================================================
-- Índices de desempenho
-- =============================================================================

-- historico_precos: queries por produto e por data (build_forecasting_dataset, price_monitor)
CREATE INDEX idx_historico_produto_loja  ON historico_precos(id_produto_loja);
CREATE INDEX idx_historico_data_recolha  ON historico_precos(data_recolha);

-- produtos_loja: joins frequentes por produto-mestre e por loja
CREATE INDEX idx_produtos_loja_mestre    ON produtos_loja(id_produto_mestre);
CREATE INDEX idx_produtos_loja_loja      ON produtos_loja(id_loja);

-- produtos_mestre: pesquisa por nome e marca (recommender, price_monitor)
CREATE INDEX idx_produtos_mestre_nome    ON produtos_mestre(nome_padronizado);
CREATE INDEX idx_produtos_mestre_marca   ON produtos_mestre(marca);

-- precos_atuais: filtros por promoção e por data de recolha
CREATE INDEX idx_precos_atuais_em_promocao   ON precos_atuais(em_promocao);
CREATE INDEX idx_precos_atuais_data_recolha  ON precos_atuais(data_recolha);

-- Índice único: impede duplicados para o mesmo produto + timestamp + scraper.
-- COALESCE(agente_origem, '') trata NULL como string vazia na comparação de unicidade.
CREATE UNIQUE INDEX uq_historico_produto_data_origem
    ON historico_precos (id_produto_loja, data_recolha, COALESCE(agente_origem, ''));


-- =============================================================================
-- Funções de trigger — actualização automática de timestamps
-- =============================================================================

-- Trigger genérico para a coluna data_atualizacao da tabela lojas.
CREATE OR REPLACE FUNCTION atualizar_timestamp_data_atualizacao()
RETURNS TRIGGER LANGUAGE plpgsql AS $$
BEGIN
    NEW.data_atualizacao = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$;

-- Trigger genérico para a coluna data_ultima_atualizacao de precos_atuais.
CREATE OR REPLACE FUNCTION atualizar_timestamp_preco_atual()
RETURNS TRIGGER LANGUAGE plpgsql AS $$
BEGIN
    NEW.data_ultima_atualizacao = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$;


-- =============================================================================
-- Triggers
-- =============================================================================

DROP TRIGGER IF EXISTS trg_lojas_data_atualizacao ON lojas;
CREATE TRIGGER trg_lojas_data_atualizacao
    BEFORE UPDATE ON lojas
    FOR EACH ROW
    EXECUTE FUNCTION atualizar_timestamp_data_atualizacao();

DROP TRIGGER IF EXISTS trg_precos_atuais_data_atualizacao ON precos_atuais;
CREATE TRIGGER trg_precos_atuais_data_atualizacao
    BEFORE UPDATE ON precos_atuais
    FOR EACH ROW
    EXECUTE FUNCTION atualizar_timestamp_preco_atual();


-- =============================================================================
-- Extensão pg_trgm — pesquisa de texto por similaridade
-- =============================================================================
-- Usada pelo recommender (similarity()) e pelo ingest (pesquisa por nome).
-- O índice GIN acelera operadores ILIKE e % de forma significativa em tabelas
-- com muitos produtos.

CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS unaccent;

CREATE INDEX IF NOT EXISTS idx_produtos_mestre_nome_trgm
    ON produtos_mestre USING GIN (nome_padronizado gin_trgm_ops);


-- =============================================================================
-- Tabela: lojas_fisicas
-- =============================================================================
-- Representa estabelecimentos físicos onde o utilizador pode efectivamente ir
-- comprar — distinto da tabela `lojas`, que representa instâncias online da
-- cadeia onde o scraping recolhe preços.
--
-- A relação com `lojas` é semântica, via coluna `insignia` (mesma string usada
-- nas duas tabelas). Não usamos FK porque os ciclos de vida são independentes:
-- a loja online "Continente" existe sempre (uma linha em `lojas`), enquanto
-- existem centenas de lojas físicas Continente que podem abrir/fechar.
--
-- Suporta a funcionalidade de "custo de deslocação" do RecommendationAgent:
-- dada a localização GPS do utilizador, identifica as lojas físicas mais
-- próximas e calcula o custo total (preço dos produtos + deslocação).

CREATE TABLE lojas_fisicas (
    id_loja_fisica   SERIAL          PRIMARY KEY,
    insignia         VARCHAR(100)    NOT NULL,
    nome_loja        VARCHAR(255)    NOT NULL,
    morada           TEXT,
    codigo_postal    VARCHAR(20),
    cidade           VARCHAR(100),
    distrito         VARCHAR(100),
    latitude         NUMERIC(9,6)    NOT NULL CHECK (latitude  BETWEEN -90  AND 90),
    longitude        NUMERIC(9,6)    NOT NULL CHECK (longitude BETWEEN -180 AND 180),
    telefone         VARCHAR(50),
    horario          TEXT,
    fonte            VARCHAR(100)    NOT NULL DEFAULT 'manual',
    external_id      VARCHAR(100),
    ativa            BOOLEAN         NOT NULL DEFAULT TRUE,
    data_criacao     TIMESTAMPTZ     NOT NULL DEFAULT CURRENT_TIMESTAMP,
    data_atualizacao TIMESTAMPTZ     NOT NULL DEFAULT CURRENT_TIMESTAMP,

    -- Idempotência: o scraper pode re-executar sem duplicar entradas
    UNIQUE (insignia, external_id)
);

COMMENT ON TABLE  lojas_fisicas                  IS 'Estabelecimentos físicos das cadeias de supermercado, com coordenadas GPS.';
COMMENT ON COLUMN lojas_fisicas.insignia         IS 'Nome da cadeia. Deve coincidir com lojas.insignia para permitir join semântico.';
COMMENT ON COLUMN lojas_fisicas.nome_loja        IS 'Designação completa da loja (ex: "Continente Modelo Braga Nogueira").';
COMMENT ON COLUMN lojas_fisicas.latitude         IS 'Latitude em graus decimais, sistema WGS84. CHECK garante intervalo válido.';
COMMENT ON COLUMN lojas_fisicas.longitude        IS 'Longitude em graus decimais, sistema WGS84. CHECK garante intervalo válido.';
COMMENT ON COLUMN lojas_fisicas.fonte            IS 'Origem do registo (ex: "scraper:continente", "scraper:pingo_doce", "manual").';
COMMENT ON COLUMN lojas_fisicas.external_id      IS 'Identificador da loja no site da cadeia. Garante idempotência em re-scraping.';
COMMENT ON COLUMN lojas_fisicas.ativa            IS 'FALSE para lojas que fecharam mas mantidas para histórico de scraping.';
COMMENT ON COLUMN lojas_fisicas.data_atualizacao IS 'Actualizado automaticamente pelo trigger trg_lojas_fisicas_data_atualizacao.';


-- =============================================================================
-- Índices de desempenho para lojas_fisicas
-- =============================================================================

-- Índice composto (insignia + ativa): queries típicas filtram por cadeia + activa.
CREATE INDEX idx_lojas_fisicas_insignia ON lojas_fisicas(insignia) WHERE ativa = TRUE;

-- Índices para filtragem geográfica por bounding box (uso preferencial em
-- "lojas próximas"). Para ~800 lojas em PT, um índice B-tree composto é
-- suficiente; PostGIS GIST seria desproporcional para esta dimensão.
CREATE INDEX idx_lojas_fisicas_latitude  ON lojas_fisicas(latitude)  WHERE ativa = TRUE;
CREATE INDEX idx_lojas_fisicas_longitude ON lojas_fisicas(longitude) WHERE ativa = TRUE;


-- =============================================================================
-- Trigger para lojas_fisicas — actualização automática de data_atualizacao
-- =============================================================================

DROP TRIGGER IF EXISTS trg_lojas_fisicas_data_atualizacao ON lojas_fisicas;
CREATE TRIGGER trg_lojas_fisicas_data_atualizacao
    BEFORE UPDATE ON lojas_fisicas
    FOR EACH ROW
    EXECUTE FUNCTION atualizar_timestamp_data_atualizacao();
