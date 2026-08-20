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

import argparse
import fnmatch
import re
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin, urlparse

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


def _parse_folder_list(spec: str) -> list[str]:
    folders: list[str] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if not part.isdigit():
            raise argparse.ArgumentTypeError(
                f"folder id must be numeric, got {part!r}"
            )
        folders.append(f"{int(part):03d}")
    return folders


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help=f"Tractor root URL (default: {DEFAULT_BASE_URL})",
    )
    parser.add_argument(
        "--folders",
        type=_parse_folder_list,
        default=None,
        help="Comma-separated folder ids (e.g. 0,1,10 → 000,001,010).",
    )
    parser.add_argument(
        "--folder-min",
        type=int,
        default=None,
        help="Inclusive minimum folder id when scanning a numeric range.",
    )
    parser.add_argument(
        "--folder-max",
        type=int,
        default=None,
        help="Inclusive maximum folder id when scanning a numeric range.",
    )
    parser.add_argument(
        "--pattern",
        default=DEFAULT_PATTERN,
        help=f"fnmatch pattern for files inside each folder (default: {DEFAULT_PATTERN}).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Write URLs one per line (default: stdout).",
    )
    parser.add_argument(
        "--list-folders",
        action="store_true",
        help="Only print discovered/selected folder names, then exit.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Parallel HTTP workers for folder listings (default: 8).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="HTTP timeout in seconds (default: 60).",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bar.",
    )
    args = parser.parse_args()

    base_url = _normalize_base(args.base_url)
    if (args.folder_min is None) ^ (args.folder_max is None):
        print("ERROR: --folder-min and --folder-max must be used together.", file=sys.stderr)
        return 1
    if args.folders is not None and args.folder_min is not None:
        print("ERROR: pass either --folders or --folder-min/--folder-max, not both.", file=sys.stderr)
        return 1
    if args.workers < 1:
        print("ERROR: --workers must be >= 1.", file=sys.stderr)
        return 1

    try:
        if args.folders is not None:
            folders = args.folders
        elif args.folder_min is not None:
            if args.folder_min < 0 or args.folder_max < args.folder_min:
                print("ERROR: invalid folder range.", file=sys.stderr)
                return 1
            folders = [f"{i:03d}" for i in range(args.folder_min, args.folder_max + 1)]
        else:
            print(f"Discovering folders under {base_url} …", file=sys.stderr)
            folders = list_subfolders(base_url, timeout=args.timeout)
    except urllib.error.URLError as exc:
        print(f"ERROR: failed to fetch {base_url}: {exc}", file=sys.stderr)
        return 1

    if not folders:
        print("ERROR: no folders selected or discovered.", file=sys.stderr)
        return 1

    print(f"folders={len(folders)} pattern={args.pattern!r}", file=sys.stderr)

    if args.list_folders:
        out = sys.stdout if args.output is None else args.output.open("w", encoding="utf-8")
        try:
            for name in folders:
                print(name, file=out)
        finally:
            if args.output is not None:
                out.close()
        return 0

    urls: list[str] = []
    failed: list[tuple[str, str]] = []

    def _one(folder: str) -> tuple[str, list[str] | None, str | None]:
        folder_url = urljoin(base_url, folder + "/")
        try:
            return folder, list_matching_urls(
                folder_url, pattern=args.pattern, timeout=args.timeout
            ), None
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return folder, [], None
            return folder, None, f"HTTP {exc.code}"
        except urllib.error.URLError as exc:
            return folder, None, str(exc)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_one, folder) for folder in folders]
        iterator = as_completed(futures)
        if not args.no_progress:
            iterator = tqdm(iterator, total=len(futures), desc="folders", unit="dir")
        for fut in iterator:
            folder, found, err = fut.result()
            if err is not None:
                failed.append((folder, err))
                continue
            assert found is not None
            urls.extend(found)

    urls = sorted(set(urls))
    if args.output is None:
        for url in urls:
            print(url)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text("\n".join(urls) + ("\n" if urls else ""), encoding="utf-8")
        print(f"Wrote {len(urls)} URL(s) to {args.output}", file=sys.stderr)

    print(f"total_urls={len(urls)} failed_folders={len(failed)}", file=sys.stderr)
    for folder, err in failed:
        print(f"  FAILED {folder}: {err}", file=sys.stderr)

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
