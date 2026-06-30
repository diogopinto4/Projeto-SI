"""
Ingestão de lojas físicas do scraper para a tabela ``lojas_fisicas``.

Lê o ficheiro JSON produzido por ``scrapers/lojas_fisicas_scraper.py`` e
insere/actualiza os registos via UPSERT idempotente (``ON CONFLICT DO UPDATE``).

A idempotência é garantida pelo constraint ``UNIQUE (insignia, external_id)``:
re-executar o script com novos dados actualiza os campos existentes em vez
de criar duplicados. Lojas que existiam no scraping anterior mas não no actual
mantêm-se como `ativa=TRUE` — não as marcamos como fechadas porque podem ter
sido apenas filtradas pela API por algum motivo transitório.

Uso::

    # Ingerir o ficheiro JSON mais recente
    python scripts/ingest_lojas_fisicas.py

    # Ingerir um ficheiro específico
    python scripts/ingest_lojas_fisicas.py --input scrapers/output/lojas_fisicas_20260518_032316.json

    # Múltiplos ficheiros ou padrão glob
    python scripts/ingest_lojas_fisicas.py --input "scrapers/output/lojas_fisicas_*.json"

    # Modo dry-run (validar sem escrever na BD)
    python scripts/ingest_lojas_fisicas.py --dry-run

Pré-requisitos:
    1. Tabela ``lojas_fisicas`` criada (ver ``sql/schema.sql``).
    2. Ficheiro JSON gerado por ``scrapers/lojas_fisicas_scraper.py``.
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import psycopg2

sys.path.insert(0, str(Path(__file__).parent))
from db_config import DB_CONFIG
# Reutiliza a função de carregamento da pipeline de ingestão de produtos —
# o formato de input (lista JSON) é idêntico ao do scraper de lojas físicas.
from ingest import carregar_ficheiros


# ---------------------------------------------------------------------------
# Campos obrigatórios / opcionais
# ---------------------------------------------------------------------------

#: Campos sem os quais um registo é rejeitado durante a validação.
#: O scraper já filtra coordenadas inválidas, mas validamos defensivamente.
CAMPOS_OBRIGATORIOS: tuple[str, ...] = (
    "insignia", "nome_loja", "latitude", "longitude", "external_id", "fonte",
)


# ---------------------------------------------------------------------------
# Carregamento de ficheiros
# ---------------------------------------------------------------------------

def encontrar_ficheiro_mais_recente() -> str | None:
    """Devolve o caminho do JSON de lojas físicas mais recente, ou ``None``.

    Procura em ``scrapers/output/lojas_fisicas_*.json`` (padrão usado pelo
    scraper). Usado como default quando o utilizador não passa ``--input``.
    """
    raiz = Path(__file__).parent.parent
    candidatos = sorted(glob.glob(str(raiz / "scrapers" / "output" / "lojas_fisicas_*.json")))
    return candidatos[-1] if candidatos else None


# ---------------------------------------------------------------------------
# Validação
# ---------------------------------------------------------------------------

def validar_registo(loja: dict) -> str | None:
    """Verifica se um registo tem os campos obrigatórios e tipos válidos.

    Args:
        loja: Registo de loja do scraper.

    Returns:
        ``None`` se válido, ou string descritiva do erro caso contrário.
    """
    for campo in CAMPOS_OBRIGATORIOS:
        if not loja.get(campo):
            return f"campo '{campo}' em falta ou vazio"

    try:
        lat = float(loja["latitude"])
        lon = float(loja["longitude"])
    except (TypeError, ValueError):
        return f"coordenadas não numéricas: lat={loja.get('latitude')!r} lon={loja.get('longitude')!r}"

    # Match com CHECK constraint da BD para falhar cedo com mensagem clara.
    if not (-90 <= lat <= 90):
        return f"latitude fora do intervalo [-90, 90]: {lat}"
    if not (-180 <= lon <= 180):
        return f"longitude fora do intervalo [-180, 180]: {lon}"

    return None


# ---------------------------------------------------------------------------
# UPSERT na BD
# ---------------------------------------------------------------------------

def upsert_loja_fisica(cur: psycopg2.extensions.cursor, loja: dict) -> None:
    """Insere ou actualiza uma loja física na tabela ``lojas_fisicas``.

    Usa ``ON CONFLICT (insignia, external_id) DO UPDATE`` para idempotência:
    re-correr o script com dados actualizados não cria duplicados.

    Apenas os campos que **podem mudar legitimamente** entre runs (lat/lon
    podem ser refinadas pela cadeia, horário pode mudar, morada pode mudar
    em mudanças de loja) são actualizados. A data_criacao preserva-se.

    Args:
        cur: Cursor psycopg2 aberto.
        loja: Registo de loja validado.
    """
    cur.execute(
        """
        INSERT INTO lojas_fisicas (
            insignia, nome_loja, morada, codigo_postal, cidade, distrito,
            latitude, longitude, telefone, horario, fonte, external_id, ativa
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, TRUE)
        ON CONFLICT (insignia, external_id) DO UPDATE SET
            nome_loja     = EXCLUDED.nome_loja,
            morada        = EXCLUDED.morada,
            codigo_postal = EXCLUDED.codigo_postal,
            cidade        = EXCLUDED.cidade,
            distrito      = COALESCE(EXCLUDED.distrito,  lojas_fisicas.distrito),
            latitude      = EXCLUDED.latitude,
            longitude     = EXCLUDED.longitude,
            telefone      = COALESCE(EXCLUDED.telefone,  lojas_fisicas.telefone),
            horario       = COALESCE(EXCLUDED.horario,   lojas_fisicas.horario),
            fonte         = EXCLUDED.fonte,
            ativa         = TRUE
        """,
        (
            loja["insignia"],
            loja["nome_loja"],
            loja.get("morada"),
            loja.get("codigo_postal"),
            loja.get("cidade"),
            loja.get("distrito"),
            float(loja["latitude"]),
            float(loja["longitude"]),
            loja.get("telefone"),
            loja.get("horario"),
            loja["fonte"],
            loja["external_id"],
        ),
    )


# ---------------------------------------------------------------------------
# Orquestração
# ---------------------------------------------------------------------------

def ingestao(registos: list[dict], dry_run: bool = False) -> tuple[int, list[dict]]:
    """Ingere uma lista de lojas físicas na BD.

    Cada registo é processado numa transação independente (commit por registo)
    para que falhas individuais não afectem os restantes. Esta estratégia
    espelha a usada em ``scripts/ingest.py`` para produtos.

    Args:
        registos: Lista de lojas (do scraper).
        dry_run: Se ``True``, valida os registos sem escrever na BD.

    Returns:
        Tupla (total_ok, lista_de_falhas).
        Cada falha é um dict com ``insignia``, ``nome_loja``, ``erro``.
    """
    total = len(registos)
    total_ok = 0
    falhas: list[dict] = []
    intervalo = max(1, total // 10)

    if dry_run:
        print(f"[DRY-RUN] {total} lojas a validar (sem escrita na BD)...")
        for loja in registos:
            erro = validar_registo(loja)
            if erro is None:
                total_ok += 1
            else:
                falhas.append({
                    "insignia": loja.get("insignia", "?"),
                    "nome_loja": loja.get("nome_loja", "?"),
                    "erro": erro,
                })
        return total_ok, falhas

    with psycopg2.connect(**DB_CONFIG) as conn:
        for i, loja in enumerate(registos, start=1):
            erro = validar_registo(loja)
            if erro is not None:
                falhas.append({
                    "insignia": loja.get("insignia", "?"),
                    "nome_loja": loja.get("nome_loja", "?"),
                    "erro": erro,
                })
                print(f"  [FALHA] {loja.get('insignia','?')} / {loja.get('nome_loja','?')[:50]} | {erro}")
                continue

            try:
                with conn.cursor() as cur:
                    upsert_loja_fisica(cur, loja)
                conn.commit()
                total_ok += 1
            except Exception as exc:
                conn.rollback()
                falhas.append({
                    "insignia": loja["insignia"],
                    "nome_loja": loja["nome_loja"],
                    "erro": str(exc),
                })
                print(f"  [FALHA] {loja['insignia']} / {loja['nome_loja'][:50]} | {exc}")

            if i % intervalo == 0 or i == total:
                pct = i / total * 100
                print(f"  [{i:>{len(str(total))}}/{total}] {pct:.0f}% — ok={total_ok} falhas={len(falhas)}")

    return total_ok, falhas


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    """Ponto de entrada CLI: carrega JSON, valida e ingere na BD."""
    parser = argparse.ArgumentParser(
        description="Ingestão de lojas físicas para a tabela lojas_fisicas.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  # Ingerir o JSON mais recente automaticamente
  python scripts/ingest_lojas_fisicas.py

  # Ingerir um ficheiro específico
  python scripts/ingest_lojas_fisicas.py --input scrapers/output/lojas_fisicas_*.json

  # Validar sem escrever
  python scripts/ingest_lojas_fisicas.py --dry-run
        """,
    )
    parser.add_argument(
        "--input", nargs="+", metavar="FICHEIRO",
        help="Ficheiros JSON ou padrões glob. Default: ficheiro mais recente em scrapers/output/.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Validar registos sem escrever na BD.",
    )
    args = parser.parse_args()

    if args.input:
        padroes = args.input
    else:
        recente = encontrar_ficheiro_mais_recente()
        if recente is None:
            print("[ERRO] Nenhum ficheiro lojas_fisicas_*.json em scrapers/output/.")
            print("       Corre primeiro: python scrapers/lojas_fisicas_scraper.py")
            sys.exit(1)
        padroes = [recente]
        print(f"[INFO] A usar ficheiro mais recente: {recente}")

    print(f"[INGEST] A carregar ficheiros: {padroes}")
    registos, caminhos = carregar_ficheiros(padroes)
    print(f"[INGEST] {len(registos)} lojas em {len(caminhos)} ficheiro(s).")

    if args.dry_run:
        print("[INGEST] Modo dry-run — nenhum dado será escrito na BD.")

    total_ok, falhas = ingestao(registos, dry_run=args.dry_run)

    separador = "=" * 50
    print(f"\n{separador}")
    print(f"  Ficheiros processados : {len(caminhos)}")
    print(f"  Registos processados  : {len(registos)}")
    print(f"  Ingeridos com sucesso : {total_ok}")
    print(f"  Com falha             : {len(falhas)}")
    if args.dry_run:
        print("  [DRY-RUN: nada foi escrito na BD]")
    print(separador)


if __name__ == "__main__":
    main()
