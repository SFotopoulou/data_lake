"""
query_from_master – build DuckDB SQL for want → master → catalog joins.

Uses ``<master>.meta.json`` (or heuristics from ``dl-describe-master``) to map
master ID columns to per-survey catalog join keys.  Column picks come from
``dl-describe-survey`` / schema manifests.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from data_lake.lake_registry import load_master_meta, read_parquet_column_names
from data_lake.schema_registry import load_catalog_schema_manifest

log = logging.getLogger(__name__)

_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def survey_sql_alias(survey: str) -> str:
    """DuckDB-safe table alias derived from survey name."""
    slug = re.sub(r"[^A-Za-z0-9]+", "_", survey).strip("_").lower()
    if not slug or slug[0].isdigit():
        slug = f"s_{slug or 'survey'}"
    return slug[:24]


def parse_column_picks(specs: Sequence[str]) -> dict[str, list[str]]:
    """
    Parse ``SURVEY:col1,col2`` or ``SURVEY.col1,col2`` tokens into a survey → columns map.
    """
    out: dict[str, list[str]] = {}
    for raw in specs:
        token = raw.strip()
        if not token:
            continue
        if ":" in token:
            survey, cols_part = token.split(":", 1)
        elif "." in token:
            survey, cols_part = token.split(".", 1)
        else:
            raise ValueError(
                f"Invalid column pick {raw!r}; use SURVEY:col1,col2 or SURVEY.col1,col2"
            )
        survey = survey.strip()
        cols = [c.strip() for c in cols_part.split(",") if c.strip()]
        if not survey or not cols:
            raise ValueError(f"Invalid column pick {raw!r}")
        out.setdefault(survey, []).extend(cols)
    return out


def _quote_ident(name: str) -> str:
    if _IDENT_RE.match(name):
        return name
    return '"' + name.replace('"', '""') + '"'


def _partner_by_survey(meta: Mapping[str, Any]) -> dict[str, dict[str, str]]:
    partners: dict[str, dict[str, str]] = {}
    for p in meta.get("partners", []):
        survey = p.get("survey")
        if survey:
            partners[survey] = p
    return partners


def _default_master_columns(
    meta: Mapping[str, Any],
    master_parquet: Path,
    primary_survey: str,
) -> list[str]:
    available = set(read_parquet_column_names(master_parquet))
    cols: list[str] = []
    for p in meta.get("partners", []):
        if p.get("survey") == primary_survey:
            continue
        mc = p.get("master_column")
        if mc and mc in available and mc not in cols:
            cols.append(mc)
    for qa in ("sep_arcsec", "match_rank", "healpix_npix", "norder"):
        if qa in available and qa not in cols:
            cols.append(qa)
    return cols


def build_catalog_view_ddl(
    lake_root: Path | str,
    survey: str,
    *,
    norder: int | None = None,
    hive_partitioning: bool = False,
    view_name: str | None = None,
) -> str:
    """``CREATE OR REPLACE VIEW`` for one HATS-partitioned catalog."""
    lake_root = Path(lake_root)
    catalog_root = lake_root / "catalogs" / survey
    if not catalog_root.is_dir():
        raise FileNotFoundError(f"Catalog not found: {catalog_root}")

    if norder is None:
        info_path = catalog_root / "catalog_info.json"
        if info_path.is_file():
            with open(info_path) as fh:
                norder = int(json.load(fh).get("hats_order", 5))
        else:
            norder = 5

    view = view_name or survey
    glob = catalog_root / f"Norder={norder}" / "**" / "*.parquet"
    hp = "true" if hive_partitioning else "false"
    return (
        f"CREATE OR REPLACE VIEW {_quote_ident(view)} AS\n"
        f"SELECT * FROM parquet_scan('{glob.as_posix()}', hive_partitioning={hp});"
    )


def build_master_view_ddl(
    master_parquet: Path | str,
    *,
    view_name: str = "master",
    lake_root: Path | str | None = None,
) -> str:
    """``CREATE OR REPLACE VIEW`` for the association master Parquet."""
    master_parquet = Path(master_parquet)
    if not master_parquet.is_file():
        raise FileNotFoundError(master_parquet)
    path = master_parquet.resolve()
    return (
        f"CREATE OR REPLACE VIEW {_quote_ident(view_name)} AS\n"
        f"SELECT * FROM read_parquet('{path.as_posix()}');"
    )


@dataclass
class MasterQueryPlan:
    """Generated SQL bundle for DuckDB execution."""

    sql: str
    view_ddls: list[str] = field(default_factory=list)
    aliases: dict[str, str] = field(default_factory=dict)
    primary_survey: str = ""
    want_table: str = "want"
    want_id_column: str = "id"

    def all_sql(self) -> str:
        """DDL statements followed by the SELECT."""
        parts = list(self.view_ddls)
        parts.append(self.sql)
        return "\n\n".join(parts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sql": self.sql,
            "view_ddls": self.view_ddls,
            "aliases": self.aliases,
            "primary_survey": self.primary_survey,
            "want_table": self.want_table,
            "want_id_column": self.want_id_column,
            "all_sql": self.all_sql(),
        }


def build_select_from_master(
    lake_root: Path | str,
    master_parquet: Path | str,
    primary_survey: str,
    columns: Mapping[str, Sequence[str]],
    *,
    want_table: str = "want",
    want_id_column: str = "id",
    master_table: str = "master",
    master_meta: Mapping[str, Any] | None = None,
    master_columns: Sequence[str] | None = None,
    include_views: bool = True,
    hive_partitioning: bool = False,
    validate_columns: bool = False,
) -> MasterQueryPlan:
    """
    Build ``want → master → catalog(s)`` SELECT SQL from master metadata and column picks.

    Parameters
    ----------
    primary_survey
        Home survey: ``want.<id>`` joins ``master.<partner_id>`` and
        ``<primary>.<catalog_id_column>``.
    columns
        Survey name → catalog column names to project (from ``dl-describe-survey``).
    master_columns
        Extra columns from the master Parquet (partner IDs, ``sep_arcsec``, …).
        Default: partner ID columns + common QA columns present in the master file.
    include_views
        When True, prepend ``CREATE OR REPLACE VIEW`` DDL for master and catalogs.
    validate_columns
        When True, warn if a picked column is missing from ``schema_manifest.json``.
    """
    lake_root = Path(lake_root)
    master_parquet = Path(master_parquet)
    if not columns:
        raise ValueError("columns must list at least one survey and column")

    meta = dict(master_meta) if master_meta is not None else load_master_meta(
        master_parquet, lake_root
    )
    partners = _partner_by_survey(meta)
    if primary_survey not in partners:
        raise ValueError(
            f"primary_survey {primary_survey!r} not in master partners: "
            f"{sorted(partners)}"
        )
    primary_partner = partners[primary_survey]

    surveys = list(columns.keys())
    for survey in surveys:
        if survey not in partners:
            raise ValueError(
                f"Survey {survey!r} in columns but not in master partners"
            )

    if master_columns is None:
        master_columns = _default_master_columns(meta, master_parquet, primary_survey)

    aliases = {s: survey_sql_alias(s) for s in surveys}
    w = _quote_ident(want_table)
    m = _quote_ident(master_table)
    wid = _quote_ident(want_id_column)

    select_parts: list[str] = [f"{w}.{wid} AS want_{want_id_column}"]
    for col in master_columns:
        select_parts.append(f"{m}.{_quote_ident(col)} AS master_{col}")

    if validate_columns:
        for survey, cols in columns.items():
            manifest_path = lake_root / "catalogs" / survey / "schema_manifest.json"
            if not manifest_path.is_file():
                continue
            manifest = load_catalog_schema_manifest(lake_root / "catalogs" / survey)
            known = {c["name"] for c in manifest.get("columns", [])}
            for col in cols:
                if col not in known:
                    log.warning(
                        "Column %s not in %s schema_manifest (%d known)",
                        col,
                        survey,
                        len(known),
                    )

    for survey, cols in columns.items():
        alias = _quote_ident(aliases[survey])
        partner = partners[survey]
        cat_id = _quote_ident(partner["catalog_id_column"])
        for col in cols:
            c = _quote_ident(col)
            select_parts.append(f"{alias}.{c} AS {aliases[survey]}_{col}")

    join_parts: list[str] = [
        f"FROM {w}",
        f"INNER JOIN {m} ON {m}.{_quote_ident(primary_partner['master_column'])} = {w}.{wid}",
    ]

    for survey in surveys:
        alias = _quote_ident(aliases[survey])
        partner = partners[survey]
        cat_id = _quote_ident(partner["catalog_id_column"])
        view = _quote_ident(survey)
        if survey == primary_survey:
            join_parts.append(
                f"INNER JOIN {view} AS {alias} ON {alias}.{cat_id} = {w}.{wid}"
            )
        else:
            join_parts.append(
                f"INNER JOIN {view} AS {alias} ON {alias}.{cat_id} = "
                f"{m}.{_quote_ident(partner['master_column'])}"
            )

    sql = "SELECT\n    " + ",\n    ".join(select_parts) + "\n" + "\n".join(join_parts)

    view_ddls: list[str] = []
    if include_views:
        view_ddls.append(
            build_master_view_ddl(master_parquet, view_name=master_table, lake_root=lake_root)
        )
        for survey in surveys:
            view_ddls.append(
                build_catalog_view_ddl(
                    lake_root,
                    survey,
                    hive_partitioning=hive_partitioning,
                    view_name=survey,
                )
            )

    return MasterQueryPlan(
        sql=sql,
        view_ddls=view_ddls,
        aliases=aliases,
        primary_survey=primary_survey,
        want_table=want_table,
        want_id_column=want_id_column,
    )


try:
    import click

    from data_lake.cli_utils import config_option, load_optional_config

    @click.command("dl-build-query-from-master")
    @click.argument("master_parquet", type=click.Path(exists=True, path_type=Path))
    @click.option(
        "--primary-survey",
        required=True,
        help="Home survey (want IDs join this catalog on source_id).",
    )
    @click.option(
        "--column",
        "column_picks",
        multiple=True,
        required=True,
        help="Column pick: SURVEY:col1,col2 (repeat per survey).",
    )
    @click.argument("output_root", type=click.Path(path_type=Path), required=False)
    @config_option
    @click.option("--want-table", default="want", show_default=True)
    @click.option("--want-id-column", default="id", show_default=True)
    @click.option(
        "--master-column",
        "master_columns",
        multiple=True,
        help="Extra master Parquet columns (default: partner IDs + sep_arcsec).",
    )
    @click.option("--no-views", is_flag=True, help="Emit SELECT only (no CREATE VIEW DDL).")
    @click.option("--json", "as_json", is_flag=True, help="Emit plan as JSON.")
    def cli(
        master_parquet: Path,
        primary_survey: str,
        column_picks: tuple[str, ...],
        output_root: Path | None,
        config_path: Path | None,
        want_table: str,
        want_id_column: str,
        master_columns: tuple[str, ...],
        no_views: bool,
        as_json: bool,
    ) -> None:
        """Print DuckDB SQL: want → master → catalogs from master.meta.json."""
        cfg = load_optional_config(config_path)
        if cfg is None and output_root is None:
            raise click.UsageError(
                "Provide OUTPUT_ROOT or set DATA_LAKE_CONFIG / lake_config.toml."
            )
        lake_root = Path(output_root) if output_root is not None else cfg.lake.root
        columns = parse_column_picks(column_picks)
        plan = build_select_from_master(
            lake_root,
            master_parquet,
            primary_survey,
            columns,
            want_table=want_table,
            want_id_column=want_id_column,
            master_columns=master_columns or None,
            include_views=not no_views,
        )
        if as_json:
            click.echo(json.dumps(plan.to_dict(), indent=2))
        else:
            click.echo(plan.all_sql())

except ImportError:
    cli = None  # type: ignore[misc, assignment]
