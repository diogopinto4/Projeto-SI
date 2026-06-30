# Decisões de design — ProjetoSI

Documento de referência com as decisões arquiteturais e algorítmicas tomadas
ao longo do desenvolvimento. Cada secção tem **contexto · decisão · alternativas
consideradas · justificação** para que possa ser citado directamente no relatório
académico.

---

## 1. Sistema multiagente em SPADE + XMPP

**Contexto.** O enunciado da UC requer um sistema multiagente. SPADE é a
biblioteca Python mais usada para esse efeito em contexto académico em PT.

**Decisão.** Adotar **SPADE 3** sobre **XMPP (Openfire)** como middleware, com
mensagens **FIPA-ACL** entre agentes. Cada agente é uma classe Python
independente, com `__init__`, `setup`, e um ou mais *behaviours* (`CyclicBehaviour`,
`PeriodicBehaviour`, `OneShotBehaviour`).

**Alternativas.**
- *JADE (Java)*: mais maduro mas requer JVM e linguagem diferente do resto do projeto.
- *Mesa* (Python): orientado a simulação de ABM, não a sistemas em produção.
- *Mensagens diretas via gRPC*: técnico-correto mas perde a semântica FIPA-ACL.

**Justificação.** SPADE encaixa-se directamente no ecossistema Python (usado
no resto do projeto), mantém a separação entre lógica do agente (Python) e
infraestrutura de mensagens (XMPP). O Openfire é trivial de subir via Docker.

---

## 2. Openfire — escolha da imagem Docker

**Contexto.** O Openfire tem várias imagens Docker comunitárias.

**Decisão.** Usar **`nasqueron/openfire`** em vez de `sameersbn/openfire`.

**Alternativas.**
- `sameersbn/openfire`: imagem clássica, mas abandonada há vários anos.
- Build próprio: overkill para um projeto académico.

**Justificação.** Detectado durante o desenvolvimento que a imagem `sameersbn`
tinha problemas de setup intermitentes. A `nasqueron/openfire` é ativamente
mantida e funcionou consistentemente.

---

## 3. Auto-registo XMPP — opção `auto_register=False`

**Contexto.** O SPADE oferece `auto_register=True` para criar contas XMPP
ad-hoc quando o agente arranca. Requer o plugin "In-Band Registration" do
Openfire ativo.

**Decisão.** Arrancar todos os agentes com `auto_register=False` e criar
as contas previamente via `scripts/setup_openfire.py`, que usa a **REST API
do Openfire** com Basic Auth.

**Alternativas.**
- *Auto-registo*: simples mas depende do plugin "In-Band Registration"
  (não vem por defeito em todas as imagens Openfire).

**Justificação.** Desacopla o ciclo de vida dos agentes do estado do Openfire.
O `setup_openfire.py` é idempotente — corrê-lo várias vezes não cria duplicados.

---

## 4. Base de dados — PostgreSQL com schema normalizado

**Contexto.** Os dados são heterogéneos: produtos podem ser vendidos por
várias cadeias com nomes diferentes, em packs ou unidades, com ou sem promoção.

**Decisão.** Schema relacional em 5 tabelas:

| Tabela | Papel |
|---|---|
| `lojas` | Instância online da cadeia (1 linha por cadeia) |
| `produtos_mestre` | Identidade canónica do produto (independente da loja) |
| `produtos_loja` | Instanciação do produto numa loja (SKU + URL + categoria local) |
| `precos_atuais` | Snapshot do preço mais recente (UPSERT em cada ingestão) |
| `historico_precos` | Append-only com todas as observações (alimenta o LSTM) |

**Alternativas.**
- *MongoDB / NoSQL*: simplifica esquema mas complica queries analíticas
  (médias por categoria, joins entre produtos e lojas).
- *Schema desnormalizado* (1 tabela com tudo): inviabiliza pesquisa eficiente.

**Justificação.** Postgres dá:
- Constraints (`UNIQUE`, `CHECK`) que detectam dados inválidos cedo.
- Window functions (`ROW_NUMBER OVER PARTITION BY`) para o monitor de preços.
- Extensão `pg_trgm` para pesquisa de produtos por similaridade textual.

---

## 5. Chave mestre — identificação canónica de produtos

**Contexto.** O mesmo produto "Arroz Agulha Cigala 1kg" pode aparecer com
nomes ligeiramente diferentes em Continente, Pingo Doce e Auchan. Precisamos
de uma forma determinística de identificar a mesma "coisa" entre cadeias.

**Decisão.** Em `scripts/ingest.py`, construir uma `chave_mestre`:
- **Se EAN disponível**: `"ean:{ean}"` — chave inequívoca.
- **Caso contrário**: hash semântico composto:
  ```
  n:{nome_normalizado_slug} | m:{marca_slug} | c:{categoria_slug} | q:{descritor_quantidade}
  ```
  com `q:` a codificar a embalagem (pack `3x250g` vs. simples `500g`).

**Alternativas.**
- *Confiar só no nome*: produtos com variações tipográficas ficam como duplicados.
- *Confiar só no EAN*: nem todas as cadeias expõem EAN (Continente/Pingo Doce
  não expõem — só Auchan).
- *Aprendizagem automática (record linkage)*: overkill, não-determinístico,
  difícil de auditar.

**Justificação.** A chave determinística garante que re-ingerir o mesmo
produto não cria duplicados. Auditável (é só uma string), reproduzível
entre execuções, e não requer treino.

**Consequência cross-store (limitação conhecida).** Como o EAN só existe no
Auchan e o hash semântico inclui categoria e quantidade — cujos slugs diferem
entre cadeias (ex: Continente `mercearia/arroz` vs Pingo Doce `mercearia`) — a
`chave_mestre` raramente liga o mesmo produto entre cadeias: na prática quase
cada `produto_loja` tem o seu próprio `produto_mestre`. Por isso a **comparação
de preços entre lojas na dashboard não depende da `chave_mestre`**: é feita em
tempo de query por **similaridade de `nome_padronizado`** (`pg_trgm`,
`similarity > 0.20`) em `recomendar_melhor_loja`/`otimizar_lista_compras`. Isto
torna a comparação robusta a chaves mestre divergentes, à custa de poder, em
casos raros, agrupar variantes próximas. Produtos que só existem numa cadeia
aparecem corretamente com uma única loja (poupança 0%) — não é um erro, é a
cobertura real dos dados.

---

## 6. Scrapers — estratégia por cadeia

**Contexto.** Cada cadeia tem o seu site, formato e mecanismo anti-scraping.

**Decisão.**
- **Auchan**: Salesforce Commerce Cloud (SFCC). Descoberta via sitemap XML;
  dados via endpoint AJAX `Product-Variation` (JSON puro). Fallback HTML SSR
  se o AJAX falhar.
- **Pingo Doce**: sitemap XML. Dados via parsing JSON-LD da página do produto.
- **Continente**: paginação por categoria via parâmetros `start` + `sz`.
  Dados embebidos no atributo `data-product-tile-impression` de cada tile.

**Alternativas.**
- *Selenium* (browser headless): mais robusto a JS, mas 10× mais lento, traz
  Chrome de 200MB para dentro do container.
- *Scrapy* (framework): orientado a crawls grandes, exagerado para 3 cadeias.

**Justificação.** Combinação de `requests` + `BeautifulSoup` é suficiente.
Cada scraper foi adaptado à arquitetura específica do site, com fallback
para preservar a recolha quando a fonte primária muda.

**Deteção de promoção — semântica comum, sinais diferentes.** O schema é
partilhado (`preco_original` vazio = sem promoção), mas cada cadeia expõe a
promoção de forma diferente, pelo que o sinal é específico:
- **Auchan**: só marca promoção se `price.isPromotion` for verdadeiro *e*
  `price.list.value` existir. O gate em `isPromotion` evita falsos positivos
  de preço de referência/PVPR (em que `list.value ≠ sales.value` sem desconto
  ativo).
- **Continente**: lê o preço antigo riscado do tile (`extract_old_price_from_tile`).
- **Pingo Doce**: lê o preço antigo do JSON-LD (`priceBeforeDiscount`,
  `listPrice`, `oldPrice`, `regularPrice`), com fallback ao HTML.

Os três aplicam a mesma defesa final (`preco_original == preco → ""`). A
métrica "% em promoção" da dashboard é, por isso, **conservadora no Auchan** e
potencialmente mais inclusiva nas outras cadeias — a comparar entre cadeias,
interpretar com esta nuance.

---

## 7. Scraping em ciclo — `PeriodicBehaviour` no SPADE

**Contexto.** Queremos recolha automática regular (não one-shot).

**Decisão.** Cada scraper é um agente SPADE com `PeriodicBehaviour` (período
default: 6h). O scraping bloqueante (`requests.get` + `time.sleep`) corre
em **thread separada via `asyncio.to_thread`** para não bloquear o event loop.

**Justificação.** Mantém o paradigma de agentes (cada cadeia é um "sensor"
autónomo), com a infraestrutura assíncrona do SPADE intacta apesar do
código de scraping ser síncrono.

---

## 8. Modelo de previsão — LSTM global

**Contexto.** Queremos prever preços por produto. Há ~2000 produtos com
histórico suficiente — treinar um modelo por produto não escala.

**Decisão.** **Um único LSTM partilhado entre todos os produtos** ("modelo
global"), com:
- Janela de 7 dias.
- 2 camadas LSTM (hidden=64) + Dropout 0.3 + Linear(1).
- `MinMaxScaler` **por produto** (cada produto tem o seu scaler — preserva
  escalas absolutas distintas: arroz a 1.69€ vs. azeite a 7.99€).
- Split temporal **por produto** (últimos `janela` dias = validação).

**Alternativas.**
- *Um modelo por produto*: explode em complexidade (2000 modelos) e não
  partilha conhecimento entre produtos similares.
- *Modelos clássicos (ARIMA)*: necessitam estimação por produto, não captam
  features adicionais (cadeia, dia da semana).
- *Transformer*: overkill para séries diárias curtas (<60 dias).

**Justificação.** Um único LSTM com features identificadoras (`cadeia_id`,
`weekday`, `weekofyear`) consegue aprender padrões partilhados (ex: descontos
ao fim-de-semana) e específicos (ex: produtos com preço estável vs. volátil).
Treina-se uma vez, prevê-se qualquer produto.

---

## 8b. Validação experimental — comparação com baselines clássicos

**Contexto.** Como justificar quantitativamente a escolha de um modelo LSTM
em vez de algo mais simples? Sem comparação não é possível.

**Decisão.** Suite de **6 modelos avaliados na mesma janela de validação**
(últimos 7 dias por produto), com **mesmas métricas em euros reais**
(RMSE/MAE/MAPE médios + std + win rate):

| Modelo | Tipo | Implementação |
|---|---|---|
| Naive | Zero-knowledge | "amanhã = preço de hoje" |
| Média móvel (3d) | Linear simples | mean dos últimos 3 dias |
| Média móvel (7d) | Linear simples | mean dos últimos 7 dias |
| ARIMA(1,1,1) | Estatístico clássico | `statsmodels.tsa.arima.ARIMA` |
| Gradient Boosting | Árvores com boosting | `sklearn.HistGradientBoostingRegressor` |
| LSTM | Deep learning | modelo de aprendizagem profunda |

Implementado em `models/baselines.py` (baselines, interface uniforme
`prever_X(serie, horizonte) -> np.ndarray`), `models/gbm_predictor.py`
(gradient boosting) e `scripts/comparar_modelos.py` (orquestração + relatório).

**Alternativas consideradas.**
- *Prophet (Facebook)*: dependência pesada, comportamento similar ao ARIMA
  para o caso de uso.
- *auto-arima (pmdarima)*: ~10× mais lento que ARIMA(1,1,1) fixo, ganho
  marginal para séries curtas.
- *N-BEATS / N-HiTS*: deep learning igualmente complexo ao LSTM — não
  acrescenta variedade comparativa.

**Win rate como métrica secundária.** A média global pode esconder
heterogeneidade (modelo X melhor para 80% dos produtos mas com RMSE alto
em 20% restantes domina a média). Reportar **win rate** (% de produtos
onde cada modelo tem o menor RMSE) complementa as médias e revela esse
padrão.

**Resultados observados (~2745 produtos comuns, 81 dias de histórico, 7 dias
de validação cada).** O **gradient boosting** tem o **menor erro médio** por
larga margem (RMSE ≈ 0.097 € vs 0.140 € do Naive e 0.220 € do LSTM; MAE
≈ 0.042 €, cerca de metade do Naive) **e** o menor desvio-padrão (≈ 0.21 vs
≈ 0.33-0.35 dos restantes), ou seja, é também o **mais consistente**. O
**ARIMA(1,1,1)** tem o maior **win rate** (~74 %) mas um RMSE ~45 % pior que
o GBM, e o LSTM fica em **último** no erro médio (mesmo com 81 dias de
histórico, não supera o baseline naive). A divergência entre "menor erro
médio" (GBM) e "maior win rate" (ARIMA) é informativa: o ARIMA ganha por
margem pequena nas muitas séries quase constantes, mas o GBM é muito melhor
nos produtos difíceis (promoções, séries voláteis) — onde os outros falham
mais — puxando o erro médio global para baixo. **A métrica que decide a
qualidade global é o erro médio, não o win rate.**

**Porque é que o gradient boosting vence.** Com o volume actual (poucas
dezenas de dias por produto), uma rede recorrente fica **sub-treinada**. Um
modelo de árvores sobre as mesmas features tabulares (lags, médias móveis,
dia-da-semana, cadeia, promoção) generaliza melhor, treina em segundos e não
precisa de normalização. O segredo de estabilidade foi **parametrizar o alvo
como rácio `preço_{t+1}/preço_t`** (invariante à escala) em vez do preço
absoluto — sem isso, o modelo global colapsava em produtos caros.

**Justificação académica.** Sem esta análise comparativa, a escolha do modelo
seria "de moda" sem justificação. Com ela, o projeto demonstra rigor: mede-se
honestamente que **o gradient boosting é a melhor escolha actual** dado o
tamanho dos dados, mantém-se o LSTM como aposta de futuro (contingente em ter
mais observações por produto), e usa-se sempre a mesma metodologia de
avaliação para que a comparação seja justa.

---

## 9. Incerteza nas previsões — Monte Carlo Dropout

**Contexto.** Uma previsão pontual ("o preço será X €") é menos útil do que
uma previsão com intervalo de confiança ("o preço será X € ± Y €").

**Decisão.** **Monte Carlo Dropout** (Gal & Ghahramani, 2016): manter o
Dropout activo durante a inferência e correr N passes forward independentes.
A variância entre simulações capta a incerteza epistémica do modelo.

**Alternativas.**
- *Bayesian Neural Networks*: matematicamente mais sólido mas muito mais lento.
- *Bootstrap*: requer treinar N modelos.
- *Quantile regression*: requer reformular a loss e a arquitectura.

**Justificação.** MC Dropout é "quase grátis" — o modelo já tem Dropout para
regularização. Re-usar como estimador de incerteza apenas requer manter
`model.train()` durante inferência e correr 50-100 passes (poucos segundos).

---

## 10. Recomendação — agregação por loja com custo de deslocação

**Contexto.** Dado uma lista de compras, queremos minimizar o custo total
considerando que o utilizador tem de se deslocar para comprar.

**Decisão.** A função `otimizar_lista_compras_geo` calcula, para cada cadeia:
```
custo_total = custo_produtos + 2 × distância_até_loja_mais_próxima × €/km
```
e recomenda a cadeia com `custo_total` mínimo (entre as que têm a lista
completa disponível).

**Justificação.** Modelo simples mas realista: a deslocação ida-volta é o
caso típico ("vou ao supermercado e regresso a casa"). O **fator 2** é
explícito no código, não escondido num número mágico.

---

## 10b. Recomendação multi-loja — rota triangular

**Contexto.** O modelo single-store (10) pode subestimar a poupança real:
às vezes vale a pena ir a 2 supermercados se 1 tem certos produtos muito
mais baratos.

**Decisão.** Função `otimizar_lista_compras_geo_multi_loja` avalia também
divisões da lista entre **2 cadeias**, modelando uma **rota triangular**:
```
user → loja_A → loja_B → user
custo_total_par = custo_produtos_A + custo_produtos_B + dist_triangular × €/km
```
Para cada par (A, B), cada item da lista é atribuído à cadeia mais barata
(entre A e B). A recomendação só passa a "par" se `custo_total_par <
custo_single`.

**Alternativas.**
- *Múltiplas idas-voltas independentes* (`user → A → user` + `user → B → user`):
  penaliza demais; ninguém faz duas viagens separadas a supermercados na mesma
  tarde se pode fazer uma só passando por ambos.
- *Permitir 3+ lojas*: complexidade combinatorial cresce rapidamente
  (3 lojas → 3! = 6 ordens × 2³ = 8 alocações por item) e o realismo prático
  é baixo (poucos utilizadores fazem 3+ paragens).

**Justificação.** Triangulação modela uma "ida planeada" — a forma natural
como um utilizador organizaria a viagem se decidisse visitar 2 lojas. O
limite a 2 lojas é uma decisão de UX: a comparação fica entendível, o
algoritmo é O(N²) em número de cadeias (N=3 → 3 pares), e a recomendação
é interpretável ("vai a X para o arroz e a Y para o azeite").

**Observação prática.** Numa lista típica de 3-5 produtos a partir de Braga,
o split tende a poupar 0.30–1.00 € — valor pequeno em absoluto mas
significativo para o objectivo do projeto (auxiliar o orçamento doméstico).

---

## 10c. Cache TTL para queries de lojas próximas

**Contexto.** O dashboard pode fazer várias chamadas seguidas a
``lojas_proximas`` para a mesma lat/lon (cada checkbox toggle, cada
mudança de preset). O GPS do browser produz coords ligeiramente
diferentes entre pedidos consecutivos por causa de oscilação sub-métrica.

**Decisão.** Cache in-memory simples em ``models/geolocation.py``:

- **Chave**: ``(round(lat, 4), round(lon, 4), raio_km, insignia, limite)`` —
  arredondamento a 4 casas decimais (≈ 11 m) faz cache hit em pedidos
  quase-idênticos.
- **TTL**: 300 s (5 min) — equilibrio entre performance e correção
  (após re-correr o seed de lojas físicas, dados frescos visíveis em <5 min).
- **Escopo**: por processo Python. Não persistente entre re-arranques —
  adequado porque os agentes SPADE têm vida longa.
- **API**: ``limpar_cache_lojas_proximas()`` para invalidação manual.

**Justificação.** Speedup medido de ~4000× em hits (BD em <1 ms vs. ~25 ms
para query). Defensiva: a cache devolve **cópias** dos dicts para evitar
que callers mutem acidentalmente o estado interno.

---

## 11. Cálculo de distância — Haversine

**Contexto.** Precisamos de calcular distâncias entre pontos GPS (utilizador
→ loja física) com performance e precisão suficientes.

**Decisão.** **Fórmula de Haversine** (assume Terra como esfera de raio
6371.0088 km — média ponderada pela área, valor IUGG).

**Alternativas.**
- *Vincenty* (elipsoide WGS84): mais precisa (sub-metro) mas 5× mais lenta
  e exige iteração.
- *Distância euclidiana* (Pythagoras em lat/lon): ignora curvatura, erro
  significativo em distâncias >10 km.

**Justificação.** Para distâncias até 1000 km, Haversine tem erro <0.5% —
muito abaixo do que importa para "qual a loja mais próxima". Não requer
biblioteca externa (só `math`).

---

## 12. Custo de deslocação — presets €/km

**Contexto.** O custo monetário de um km de carro varia muito (gasolina,
manutenção, depreciação) e depende da perspectiva do utilizador.

**Decisão.** Três presets nomeados em `models/geolocation.py`:

| Preset | €/km | Origem |
|---|---|---|
| `so_combustivel` | 0.12 | Gasolina 95 a ~1.80 €/L × consumo médio 6 L/100km |
| `equilibrado` | 0.20 | Combustível + manutenção básica (pneus, óleo) — **default** |
| `tarifa_at` | 0.36 | Portaria nº 1553-D/2008 — tarifa oficial para reembolso de viatura própria |

**Alternativas.**
- *Pedir ao utilizador*: força o utilizador a pesquisar.
- *Valor único*: não respeita a heterogeneidade do uso real.

**Justificação.** Os 3 presets cobrem o espectro: minimalista (só gasolina),
realista (default), conservador (todos os custos via tarifa oficial AT). A
escolha é configurável no dashboard com explicação por tooltip.

---

## 13. Lojas físicas — fonte de dados

**Contexto.** Precisamos das coordenadas GPS de ~1000 lojas em Portugal para
calcular distâncias.

**Decisão.** Scraping das **APIs internas das próprias cadeias**:

| Cadeia | Endpoint | Resultado |
|---|---|---|
| Continente | `Stores-FindStores` (SFCC) com lat=40, long=-8, radius=500 | 228 lojas |
| Pingo Doce | mesma API SFCC + header `Accept: application/json` | 270 lojas |
| Auchan | `data-locations="[...]"` embebido na página `/pt/lojas` | 563 lojas |

Total: **1061 lojas físicas** com coordenadas WGS84 reais.

**Alternativas.**
- *Geocoding via Nominatim* (OSM): grátis mas lento (1 req/s) e nem todas as
  moradas das cadeias estão na OSM.
- *Google Places API*: precisa de chave + custo por uso.
- *Datasets abertos*: não encontrámos um actualizado para PT.

**Justificação.** As próprias cadeias expõem APIs públicas (descobertas via
inspecção das páginas oficiais). Idempotente (UPSERT por `(insignia, external_id)`),
re-executável periodicamente, sem custos.

---

## 14. UX da localização — GPS do browser + fallback manual

**Contexto.** No dashboard, o utilizador precisa de partilhar a sua localização
para a feature de custo de deslocação.

**Decisão.** Botão de geolocation que usa `navigator.geolocation.getCurrentPosition`
(via componente `streamlit-geolocation`), com **fallback** para input manual
de lat/lon ou seleção de cidade preset.

**Alternativas.**
- *Só input manual*: força o utilizador a saber coordenadas.
- *Só GPS*: falha se o utilizador negar permissão ou não estiver num browser
  com suporte.
- *IP geolocation*: privacidade questionável e precisão grosseira.

**Justificação.** GPS do browser é o padrão moderno (modelo de permissão
explícito, accuracy ~10m). O fallback manual permite demonstração com
qualquer cidade do país (útil para a apresentação académica).

---

## 14b. Persistência da localização entre sessões — localStorage

**Contexto.** O `streamlit-geolocation` pede permissão ao browser **cada vez**
que o utilizador clica no botão GPS. Mesmo dentro da mesma sessão, refrescar
a página perde a localização.

**Decisão.** Guardar a última localização confirmada no `localStorage` do
browser (chave `projetosi_user_location`) via componente
`streamlit-local-storage`. Quando a tab GPS é renderizada pela primeira vez
na sessão, lê do `localStorage` e hidrata `session_state`. Botão dedicado
para "limpar localização guardada".

**Alternativas.**
- *Cookies*: enviadas em cada request HTTP, desnecessário (a localização é
  só do lado do cliente).
- *Apenas `session_state`*: limpa-se a cada refresh da página — péssima UX.
- *URL params*: expõe localização na URL — questionável em termos de
  privacidade e UX (URLs com lat/lon não são partilháveis).

**Justificação.** O `localStorage` é o padrão moderno para dados do cliente
persistentes sem serem enviados ao servidor. Confidencial (não sai do browser),
fácil de limpar (botão dedicado + DevTools), respeita o modelo mental do
utilizador (autoriza GPS uma vez, fica guardado).

---

## 15. Custo de deslocação — porque não roteamento real?

**Contexto.** Haversine dá distância "great-circle". O caminho real por
estrada é sempre maior.

**Decisão.** Manter haversine simples (multiplica por ~1.3 para aproximar
distância de estrada, mas optámos por **não fazer essa correção** — o erro
absorve-se no preset €/km).

**Alternativas.**
- *OSRM* (Open Source Routing Machine): roteamento real, requer servidor OSRM
  (precomputado da OSM).
- *Google Directions API*: requer chave + custo.

**Justificação.** Para o caso de uso ("vale a pena ir mais longe para poupar
2 € em produtos?") a precisão sub-quilométrica de Haversine é suficiente.
O utilizador pode ajustar o €/km para compensar.

---

## 16. API REST — FastAPI sobre o sistema multiagente

**Contexto.** O dashboard (Streamlit) precisa de comunicar com os agentes,
mas falar XMPP a partir de um browser é impraticável.

**Decisão.** **FastAPI** + `UserInterfaceAgent` SPADE como ponte:
- O dashboard faz HTTP requests à API.
- A API delega para os agentes via XMPP, com `correlation_id` para correlacionar
  pedidos com respostas.
- `asyncio.Future` + `asyncio.shield` garantem que cada handler HTTP fica
  bloqueado até à resposta (ou timeout).

**Alternativas.**
- *Falar XMPP directamente do dashboard*: requer biblioteca XMPP no browser
  (BOSH/WebSocket) — complica o front.
- *gRPC entre dashboard e agentes*: perde o ponto de "agentes via XMPP".

**Justificação.** A API REST é uma "fachada" sobre o sistema multiagente —
clientes externos (incluindo IDEs como o Swagger UI em `/docs`) podem usar
o sistema sem conhecer XMPP. Mantém a integridade arquitectural.

---

## 17. Persistência e cache de modelos — disco + mtime cache

**Contexto.** O LSTM tem ~30k parâmetros + scalers para 2000 produtos. Não faz
sentido recarregar em cada chamada de previsão.

**Decisão.** Cache em memória dos artefactos (`lstm_global.pt`, `scalers.pkl`,
`model_meta.pkl`) com **invalidação por mtime combinado** — se qualquer um
dos 3 ficheiros for re-escrito (após retreino), a cache invalida-se e os
artefactos são recarregados.

**Justificação.** Equilibrio entre performance (não recarrega 200MB por query)
e correção (retreino automático periódico fica visível sem reiniciar o agente).

---

## 18. Estrutura de testes — unit + BD em separado

**Contexto.** O projeto tem código puro (parsing, fórmulas) e código que toca
em BD (queries SQL, agregações).

**Decisão.** Suite dividida em **dois grupos**:

| Grupo | Padrão | Independência |
|---|---|---|
| Unit (puros) | `test_*.py` (sem sufixo `_bd`) | Sem BD nem rede; usam fixtures Python ou HTML/JSON sintético |
| Integração com BD | `test_*_bd.py` | BD `products_db_test` separada, com fixture `db_clean` autouse |

A BD de teste é redirecionada via `os.environ["DB_NAME"] = "products_db_test"`
**antes** de qualquer import — necessário porque o projeto importa `db_config`
por 2 caminhos distintos (`scripts.db_config` e `db_config`), registando 2
módulos em `sys.modules` que não partilham referências mutáveis.

**Justificação.** Os unit tests correm em <1s (CI rápido). Os BD tests correm
em ~3s (dão confiança real) mas requerem Postgres. A defesa via assert no
`_verificar_db_teste` garante que **nunca** se escreve na BD de produção,
mesmo que algum import escape.

---

## Notas finais — formato deste documento

Este documento existe para servir o relatório académico. Cada decisão é
**rastreável até ao código** (módulo + ficheiro). Quando uma decisão for
revertida ou alterada substancialmente, **actualizar aqui** com nota de
mudança — o documento descreve o estado *actual* da arquitetura, não o
histórico evolutivo.
