"""
Cria automaticamente as contas XMPP dos agentes no Openfire via REST API.

Pré-requisitos:
    1. Openfire a correr (docker compose up -d)
    2. Setup inicial do Openfire concluído (http://localhost:9090, só na primeira vez)
    3. Plugin "REST API" instalado no Openfire:
       Openfire admin → Plugins → Available Plugins → REST API → Install

Variáveis de ambiente (ver .env.example):
    OPENFIRE_ADMIN_USER     — utilizador administrador do Openfire (default: admin)
    OPENFIRE_ADMIN_PASSWORD — password do administrador (default: admin)
    OPENFIRE_URL            — URL base do painel admin (default: http://localhost:9090)
    XMPP_PASSWORD           — password a usar em todas as contas dos agentes
    XMPP_SERVER             — domínio XMPP (default: localhost)

Uso::

    python scripts/setup_openfire.py
    python scripts/setup_openfire.py --url http://localhost:9090 --admin admin --password admin
    python scripts/setup_openfire.py --dry-run   # mostra o que ia criar, sem criar
"""

from __future__ import annotations

import argparse
import base64
import os
import sys
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")


# ---------------------------------------------------------------------------
# Configuração — lida do .env ou sobreposta por argumentos CLI
# ---------------------------------------------------------------------------

OPENFIRE_URL = os.getenv("OPENFIRE_URL", "http://localhost:9090")
OPENFIRE_ADMIN_USER = os.getenv("OPENFIRE_ADMIN_USER", "admin")
OPENFIRE_ADMIN_PASSWORD = os.getenv("OPENFIRE_ADMIN_PASSWORD", "admin")
XMPP_PASSWORD = os.getenv("XMPP_PASSWORD", "password")
XMPP_SERVER = os.getenv("XMPP_SERVER", "localhost")


# ---------------------------------------------------------------------------
# JIDs dos agentes — lidos das mesmas variáveis de ambiente que main.py usa,
# garantindo consistência mesmo que os JIDs sejam personalizados no .env.
# ---------------------------------------------------------------------------

def _username(env_var: str, default_local: str) -> str:
    """Extrai a parte local (antes do @) de um JID lido do .env."""
    jid = os.getenv(env_var, f"{default_local}@{XMPP_SERVER}")
    return jid.split("@")[0]


AGENTES: list[tuple[str, str]] = [
    (_username("XMPP_ORCHESTRATOR_JID", "orchestrator_agent"), "Orchestrator Agent"),
    (_username("XMPP_DB_JID",           "database_agent"),     "Database Agent"),
    (_username("XMPP_AUCHAN_JID",       "auchan_scraper"),     "Auchan Scraper"),
    (_username("XMPP_CONTINENTE_JID",   "continente_scraper"), "Continente Scraper"),
    (_username("XMPP_PINGO_DOCE_JID",   "pingo_doce_scraper"), "Pingo Doce Scraper"),
    (_username("XMPP_PREDICTION_JID",   "prediction_agent"),   "Prediction Agent"),
    (_username("XMPP_RECOMMENDATION_JID","recommendation_agent"),"Recommendation Agent"),
    (_username("XMPP_LOCATION_JID",     "location_agent"),     "Location Agent"),
    (_username("XMPP_MONITOR_JID",      "monitor_agent"),      "Monitor Agent"),
    (_username("XMPP_UI_JID",           "ui_agent"),           "UI Agent"),
]


# ---------------------------------------------------------------------------
# Utilitários HTTP
# ---------------------------------------------------------------------------

def _auth_header(user: str, password: str) -> dict[str, str]:
    """Constrói o cabeçalho Basic Auth para a REST API do Openfire."""
    token = base64.b64encode(f"{user}:{password}".encode()).decode()
    return {
        "Authorization": f"Basic {token}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


# ---------------------------------------------------------------------------
# Verificação do Openfire
# ---------------------------------------------------------------------------

def verificar_openfire(url: str, headers: dict, timeout_s: int = 60) -> None:
    """Aguarda o Openfire e valida que o plugin REST API está instalado.

    Args:
        url: URL base do Openfire (ex: http://localhost:9090).
        headers: Cabeçalhos com autenticação.
        timeout_s: Segundos máximos de espera.

    Raises:
        SystemExit: Se o servidor não responder, as credenciais forem inválidas
                    ou o plugin REST API não estiver instalado.
    """
    endpoint = f"{url}/plugins/restapi/v1/users"
    print(f"[setup] A aguardar Openfire ({url})...", end="", flush=True)

    deadline = time.time() + timeout_s
    last_error = "timeout"

    while time.time() < deadline:
        try:
            r = requests.get(endpoint, headers=headers, timeout=5)

            if r.status_code == 200:
                print(" pronto.")
                return

            if r.status_code == 401:
                print()
                print("[setup] ERRO: credenciais de administrador inválidas.")
                print("  Verifica OPENFIRE_ADMIN_USER e OPENFIRE_ADMIN_PASSWORD no .env")
                sys.exit(1)

            if r.status_code == 404:
                print()
                print("[setup] ERRO: plugin REST API não encontrado no Openfire.")
                print("  Instala em: Openfire admin → Plugins → Available Plugins → REST API")
                sys.exit(1)

            last_error = f"HTTP {r.status_code}"

        except requests.exceptions.ConnectionError:
            last_error = "ligação recusada"
        except requests.exceptions.Timeout:
            last_error = "timeout"

        print(".", end="", flush=True)
        time.sleep(3)

    print()
    print(f"[setup] ERRO: Openfire não acessível após {timeout_s}s ({last_error}).")
    print("  Verifica se está a correr: docker compose ps")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Criação de contas
# ---------------------------------------------------------------------------

def criar_conta(
    url: str,
    username: str,
    name: str,
    password: str,
    headers: dict,
) -> str:
    """Cria uma conta XMPP no Openfire.

    Returns:
        ``"criada"`` se foi criada com sucesso.
        ``"existe"`` se já existia (idempotente).
        ``"erro <código>: <detalhe>"`` em caso de falha.
    """
    payload = {"username": username, "password": password, "name": name}
    try:
        r = requests.post(
            f"{url}/plugins/restapi/v1/users",
            json=payload,
            headers=headers,
            timeout=10,
        )
        if r.status_code == 201:
            return "criada"
        if r.status_code == 409:
            return "existe"
        return f"erro {r.status_code}: {r.text.strip()[:120]}"
    except requests.exceptions.RequestException as exc:
        return f"erro de rede: {exc}"


# ---------------------------------------------------------------------------
# Ponto de entrada
# ---------------------------------------------------------------------------

def main() -> None:
    """Ponto de entrada CLI: cria todas as contas XMPP dos agentes no Openfire."""
    parser = argparse.ArgumentParser(
        description="Cria as contas XMPP dos agentes no Openfire via REST API.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Pré-requisitos:
  1. docker compose up -d
  2. Setup inicial do Openfire concluído (http://localhost:9090)
  3. Plugin REST API instalado no Openfire
        """,
    )
    parser.add_argument(
        "--url",
        default=OPENFIRE_URL,
        metavar="URL",
        help=f"URL base do Openfire (default: {OPENFIRE_URL})",
    )
    parser.add_argument(
        "--admin",
        default=OPENFIRE_ADMIN_USER,
        metavar="USER",
        help=f"Utilizador administrador (default: {OPENFIRE_ADMIN_USER})",
    )
    parser.add_argument(
        "--password",
        default=OPENFIRE_ADMIN_PASSWORD,
        metavar="PASS",
        help="Password do administrador",
    )
    parser.add_argument(
        "--xmpp-password",
        default=XMPP_PASSWORD,
        metavar="PASS",
        help=f"Password para as contas dos agentes (default: {XMPP_PASSWORD})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Mostra o que seria criado sem fazer alterações.",
    )
    args = parser.parse_args()

    headers = _auth_header(args.admin, args.password)

    if args.dry_run:
        print("[setup] Modo dry-run — nenhuma conta será criada.")
        print(f"  Openfire: {args.url}")
        print(f"  Admin:    {args.admin}")
        print(f"  Agentes ({len(AGENTES)}):")
        for username, name in AGENTES:
            print(f"    {username}@{XMPP_SERVER}  ({name})")
        return

    # Verificar conectividade e plugin antes de tentar criar contas
    verificar_openfire(args.url, headers)

    print(f"[setup] A criar {len(AGENTES)} contas XMPP...")
    erros = 0

    for username, name in AGENTES:
        resultado = criar_conta(args.url, username, name, args.xmpp_password, headers)
        ok = resultado in ("criada", "existe")
        simbolo = "+" if resultado == "criada" else ("=" if resultado == "existe" else "!")
        print(f"  [{simbolo}] {username}@{XMPP_SERVER}  — {resultado}")
        if not ok:
            erros += 1

    print()
    if erros:
        print(f"[setup] {erros} erro(s) encontrado(s). Verifica o output acima.")
        sys.exit(1)
    else:
        print("[setup] Contas configuradas. Podes agora correr:")
        print("  python main.py")


if __name__ == "__main__":
    main()
