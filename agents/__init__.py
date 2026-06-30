"""
Pacote de agentes do sistema multiagente de análise de preços de supermercado.

Expõe todos os agentes SPADE para importação directa a partir de ``agents.*``:

    from agents import DatabaseAgent, PredictionAgent, RecommendationAgent

Agentes disponíveis:
    AuchanScraper       — scraping periódico do Auchan (sitemap + AJAX SFCC)
    ContinenteScraper   — scraping periódico do Continente Online
    PingoDoceScraper    — scraping periódico do Pingo Doce (sitemap)
    DatabaseAgent       — ingestão na BD e resposta a consultas
    PredictionAgent     — previsão LSTM global + Monte Carlo Dropout
    RecommendationAgent — pesquisa, comparação e otimização de compras
    LocationAgent       — geolocalização + custo de deslocação (haversine, presets €/km)
    MonitorAgent        — monitor de preços + detetor de anomalias (IQR)
    OrchestratorAgent   — coordenação do pipeline pós-ingestão
    UserInterfaceAgent  — ponte HTTP ↔ XMPP (FastAPI ↔ SPADE)
"""
# Expõe todos os agentes do sistema multiagente para importação directa.
from agents.AuchanScraper       import AuchanScraper
from agents.ContinenteScraper   import ContinenteScraper
from agents.PingoDoceScraper    import PingoDoceScraper
from agents.DatabaseAgent       import DatabaseAgent
from agents.PredictionAgent     import PredictionAgent
from agents.RecommendationAgent import RecommendationAgent
from agents.LocationAgent       import LocationAgent
from agents.MonitorAgent        import MonitorAgent
from agents.OrchestratorAgent   import OrchestratorAgent
from agents.UserInterfaceAgent  import UserInterfaceAgent

__all__ = [
    "AuchanScraper",
    "ContinenteScraper",
    "PingoDoceScraper",
    "DatabaseAgent",
    "PredictionAgent",
    "RecommendationAgent",
    "LocationAgent",
    "MonitorAgent",
    "OrchestratorAgent",
    "UserInterfaceAgent",
]
