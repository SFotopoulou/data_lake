"""Send a catalog table to a running TOPCAT via SAMP."""

from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger(__name__)

_TOPCAT_NAMES = frozenset({"topcat", "topcat (beta)"})


class SampSendError(RuntimeError):
    """Hub / TOPCAT / SAMP protocol failure with an operator-facing message."""


def find_topcat_client_id(client) -> str | None:
    """Return the SAMP client id whose ``samp.name`` looks like TOPCAT."""
    for cid in client.get_registered_clients():
        try:
            meta = client.get_metadata(cid)
        except Exception:
            continue
        name = str(meta.get("samp.name", "")).strip().lower()
        if name in _TOPCAT_NAMES or name.startswith("topcat"):
            return str(cid)
    return None


def send_votable_to_topcat(
    path: Path | str,
    *,
    name: str | None = None,
    timeout_sec: float = 30.0,
) -> str:
    """Load *path* (VOTable on disk) into a connected TOPCAT via SAMP.

    Returns the TOPCAT SAMP client id. Raises :class:`SampSendError` when no
    hub is running or TOPCAT is not registered.
    """
    from astropy.samp import SAMPIntegratedClient

    vot = Path(path)
    if not vot.is_file():
        raise FileNotFoundError(f"VOTable not found: {vot}")

    table_name = name or vot.stem
    url = vot.resolve().as_uri()
    message = {
        "samp.mtype": "table.load.votable",
        "samp.params": {"url": url, "name": table_name},
    }

    client = SAMPIntegratedClient()
    try:
        try:
            client.connect()
        except Exception as exc:
            raise SampSendError(
                "No SAMP hub is running. Start TOPCAT first (it starts a hub), "
                "then retry."
            ) from exc

        topcat_id = find_topcat_client_id(client)
        if topcat_id is None:
            raise SampSendError(
                "TOPCAT is not connected to the SAMP hub. "
                "Open TOPCAT and confirm SAMP is enabled, then retry."
            )

        timeout = str(max(1, int(timeout_sec)))
        try:
            client.call_and_wait(topcat_id, message, timeout)
        except Exception as exc:
            raise SampSendError(
                f"SAMP table.load.votable to TOPCAT failed: {exc}"
            ) from exc

        log.info(
            "Sent VOTable %s as %r to TOPCAT client %s",
            vot, table_name, topcat_id,
        )
        return topcat_id
    finally:
        try:
            client.disconnect()
        except Exception:
            pass
