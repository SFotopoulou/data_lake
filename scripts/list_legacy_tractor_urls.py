#!/usr/bin/env python3
"""List Legacy Survey tractor FITS URLs under a NERSC portal directory tree.

Default root is DR11 north::

    https://portal.nersc.gov/cfs/cosmo/data/legacysurvey/dr11/north/tractor/

Brick catalogs live in three-digit subfolders (``000``, ``001``, …). Not every
index between 000 and 359 exists — by default this script discovers folders
from the parent directory listing.

Examples::

    # All tractor FITS under DR11 north (discovered folders)
    python scripts/list_legacy_tractor_urls.py -o tractor_north_urls.txt

    # Only folders 000–010
    python scripts/list_legacy_tractor_urls.py --folder-min 0 --folder-max 10

    # Explicit folders + south footprint
    python scripts/list_legacy_tractor_urls.py \\
      --base-url https://portal.nersc.gov/cfs/cosmo/data/legacysurvey/dr11/south/tractor/ \\
      --folders 100,101,102
"""

from __future__ import annotations

import fnmatch
import re
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin, urlparse

import click

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional for bare envs
    def tqdm(iterable, **_kwargs):  # type: ignore[misc]
        return iterable

DEFAULT_BASE_URL = (
    "https://portal.nersc.gov/cfs/cosmo/data/legacysurvey/dr11/north/tractor/"
)
DEFAULT_PATTERN = "tractor-*.fits"
_HREF_RE = re.compile(r'href=["\']([^"\']+)["\']', re.IGNORECASE)


class _HrefParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.hrefs: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "a":
            return
        for key, value in attrs:
            if key.lower() == "href" and value:
                self.hrefs.append(value)


def _fetch_text(url: str, *, timeout: float) -> str:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "data_lake-list_legacy_tractor_urls/1.0"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        charset = resp.headers.get_content_charset() or "utf-8"
        return resp.read().decode(charset, errors="replace")


def _parse_hrefs(html: str) -> list[str]:
    parser = _HrefParser()
    try:
        parser.feed(html)
        if parser.hrefs:
            return parser.hrefs
    except Exception:
        pass
    return _HREF_RE.findall(html)


def _normalize_base(url: str) -> str:
    return url if url.endswith("/") else url + "/"


def list_subfolders(base_url: str, *, timeout: float) -> list[str]:
    """Return sorted three-digit folder names present under *base_url*."""
    html = _fetch_text(base_url, timeout=timeout)
    folders: set[str] = set()
    for href in _parse_hrefs(html):
        name = href.rstrip("/").rsplit("/", 1)[-1]
        if re.fullmatch(r"\d{3}", name):
            folders.add(name)
    return sorted(folders)


def list_matching_urls(
    folder_url: str,
    *,
    pattern: str,
    timeout: float,
) -> list[str]:
    """Return absolute URLs of files in *folder_url* matching *pattern*."""
    html = _fetch_text(folder_url, timeout=timeout)
    folder_url = _normalize_base(folder_url)
    urls: list[str] = []
    seen: set[str] = set()
    for href in _parse_hrefs(html):
        if href in ("../", "./") or href.startswith("?"):
            continue
        name = urlparse(href).path.rsplit("/", 1)[-1]
        if not name or name.endswith("/"):
            continue
        if not fnmatch.fnmatch(name, pattern):
            continue
        absolute = urljoin(folder_url, href)
        if absolute not in seen:
            seen.add(absolute)
            urls.append(absolute)
    return sorted(urls)


def _parse_folder_list(ctx: click.Context, param: click.Parameter, value: str | None) -> list[str] | None:
    if value is None:
        return None
    folders: list[str] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if not part.isdigit():
            raise click.BadParameter(f"folder id must be numeric, got {part!r}", ctx=ctx, param=param)
        folders.append(f"{int(part):03d}")
    return folders


@click.command(context_settings={"help_option_names": ["-h", "--help"]}, help=__doc__)
@click.option(
    "--base-url",
    default=DEFAULT_BASE_URL,
    show_default=True,
    help="Tractor root URL.",
)
@click.option(
    "--folders",
    callback=_parse_folder_list,
    default=None,
    help="Comma-separated folder ids (e.g. 0,1,10 → 000,001,010).",
)
@click.option(
    "--folder-min",
    type=int,
    default=None,
    help="Inclusive minimum folder id when scanning a numeric range.",
)
@click.option(
    "--folder-max",
    type=int,
    default=None,
    help="Inclusive maximum folder id when scanning a numeric range.",
)
@click.option(
    "--pattern",
    default=DEFAULT_PATTERN,
    show_default=True,
    help="fnmatch pattern for files inside each folder.",
)
@click.option(
    "-o",
    "--output",
    type=click.Path(path_type=Path),
    default=None,
    help="Write URLs one per line (default: stdout).",
)
@click.option("--list-folders", is_flag=True, help="Only print discovered/selected folder names, then exit.")
@click.option("--workers", type=int, default=8, show_default=True, help="Parallel HTTP workers for folder listings.")
@click.option("--timeout", type=float, default=60.0, show_default=True, help="HTTP timeout in seconds.")
@click.option("--no-progress", is_flag=True, help="Disable tqdm progress bar.")
def main(
    base_url: str,
    folders: list[str] | None,
    folder_min: int | None,
    folder_max: int | None,
    pattern: str,
    output: Path | None,
    list_folders: bool,
    workers: int,
    timeout: float,
    no_progress: bool,
) -> int:
    base_url = _normalize_base(base_url)
    if (folder_min is None) ^ (folder_max is None):
        print("ERROR: --folder-min and --folder-max must be used together.", file=sys.stderr)
        return 1
    if folders is not None and folder_min is not None:
        print("ERROR: pass either --folders or --folder-min/--folder-max, not both.", file=sys.stderr)
        return 1
    if workers < 1:
        print("ERROR: --workers must be >= 1.", file=sys.stderr)
        return 1

    try:
        if folders is not None:
            folder_list = folders
        elif folder_min is not None:
            if folder_min < 0 or folder_max < folder_min:
                print("ERROR: invalid folder range.", file=sys.stderr)
                return 1
            folder_list = [f"{i:03d}" for i in range(folder_min, folder_max + 1)]
        else:
            print(f"Discovering folders under {base_url} …", file=sys.stderr)
            folder_list = list_subfolders(base_url, timeout=timeout)
    except urllib.error.URLError as exc:
        print(f"ERROR: failed to fetch {base_url}: {exc}", file=sys.stderr)
        return 1

    if not folder_list:
        print("ERROR: no folders selected or discovered.", file=sys.stderr)
        return 1

    print(f"folders={len(folder_list)} pattern={pattern!r}", file=sys.stderr)

    if list_folders:
        out = sys.stdout if output is None else output.open("w", encoding="utf-8")
        try:
            for name in folder_list:
                print(name, file=out)
        finally:
            if output is not None:
                out.close()
        return 0

    urls: list[str] = []
    failed: list[tuple[str, str]] = []

    def _one(folder: str) -> tuple[str, list[str] | None, str | None]:
        folder_url = urljoin(base_url, folder + "/")
        try:
            return folder, list_matching_urls(
                folder_url, pattern=pattern, timeout=timeout
            ), None
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return folder, [], None
            return folder, None, f"HTTP {exc.code}"
        except urllib.error.URLError as exc:
            return folder, None, str(exc)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_one, folder) for folder in folder_list]
        iterator = as_completed(futures)
        if not no_progress:
            iterator = tqdm(iterator, total=len(futures), desc="folders", unit="dir")
        for fut in iterator:
            folder, found, err = fut.result()
            if err is not None:
                failed.append((folder, err))
                continue
            assert found is not None
            urls.extend(found)

    urls = sorted(set(urls))
    if output is None:
        for url in urls:
            print(url)
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("\n".join(urls) + ("\n" if urls else ""), encoding="utf-8")
        print(f"Wrote {len(urls)} URL(s) to {output}", file=sys.stderr)

    print(f"total_urls={len(urls)} failed_folders={len(failed)}", file=sys.stderr)
    for folder, err in failed:
        print(f"  FAILED {folder}: {err}", file=sys.stderr)

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
