"""
`dl-set-ingest-token` – enable or rotate ingest guardrails on an existing deployment.

Writes ``.ingest_token_hash`` (SHA-256 only, mode 0600) and sets
``guardrails.require_ingest_token`` in ``lake_config.toml``.  Does not touch
data tiles.
"""

from __future__ import annotations

from pathlib import Path

import click

from ..cli_utils import INGEST_TOKEN_ENV, config_option, write_ingest_token_hash
from ..config import CONFIG_FILENAME, LakeConfig, LakeConfigNotFound

_INGEST_HASH_GITIGNORE_LINE = ".ingest_token_hash"


def _ensure_gitignore_lists_ingest_hash(deployment_dir: Path) -> bool:
    """Append ``.ingest_token_hash`` to ``.gitignore`` when missing."""
    gitignore = deployment_dir / ".gitignore"
    if not gitignore.is_file():
        return False
    text = gitignore.read_text(encoding="utf-8")
    if _INGEST_HASH_GITIGNORE_LINE in text:
        return False
    suffix = "" if text.endswith("\n") or not text else "\n"
    gitignore.write_text(text + suffix + _INGEST_HASH_GITIGNORE_LINE + "\n", encoding="utf-8")
    return True


def set_ingest_token(
    ingest_token: str,
    *,
    config_path: Path | None = None,
) -> Path:
    """Enable ingest guardrails for an existing deployment.

    Parameters
    ----------
    ingest_token:
        Plaintext token (hashed before writing to disk).
    config_path:
        Path to ``lake_config.toml``, or ``None`` to use discovery
        (``$DATA_LAKE_CONFIG`` / walk up from CWD).

    Returns
    -------
    Path
        Deployment directory (parent of ``lake_config.toml``).
    """
    token = ingest_token.strip()
    if not token:
        raise ValueError("ingest token must be non-empty")

    cfg = LakeConfig.discover(config_path)
    if cfg.source_path is None:
        raise RuntimeError("loaded config has no source_path")

    deployment = cfg.source_path.parent
    cfg.guardrails.require_ingest_token = True
    cfg.guardrails.ingest_token_hash = ""
    cfg.write(cfg.source_path)
    write_ingest_token_hash(deployment, token)
    _ensure_gitignore_lists_ingest_hash(deployment)
    return deployment


@click.command("dl-set-ingest-token")
@config_option
@click.option(
    "--ingest-token",
    required=True,
    envvar=INGEST_TOKEN_ENV,
    help="Ingest guard token (hashed to .ingest_token_hash; also reads "
         f"${INGEST_TOKEN_ENV}).",
)
def dl_set_ingest_token(config_path: Path | None, ingest_token: str) -> None:
    """Enable or rotate ingest guardrails on an existing deployment."""
    try:
        deployment = set_ingest_token(ingest_token, config_path=config_path)
    except LakeConfigNotFound as exc:
        raise click.ClickException(str(exc)) from exc
    except ValueError as exc:
        raise click.BadParameter(str(exc)) from exc

    cfg_path = deployment / CONFIG_FILENAME
    click.echo(f"Deployment:         {deployment}")
    click.echo(f"Config updated:     {cfg_path}")
    click.echo(f"Ingest guardrails:  on ({_INGEST_HASH_GITIGNORE_LINE} written, chmod 600)")
    click.echo("")
    click.echo(f"  export {INGEST_TOKEN_ENV}='<your ingest token>'  # before dl-ingest-*")
