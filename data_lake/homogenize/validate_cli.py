"""``dl-validate-homogenization`` – lint transforms and golden spot checks."""

from __future__ import annotations

from pathlib import Path


def print_validation_messages(rep, *, strict: bool) -> None:
    import click

    for msg in rep.warnings:
        click.echo(f"WARN: {msg}", err=True)
    for msg in rep.errors:
        click.echo(f"ERROR: {msg}", err=True)
    if rep.ok(strict=strict):
        click.echo("Homogenization validation passed.")
    else:
        raise SystemExit(1)


try:
    import click

    from data_lake.cli_utils import (
        config_option,
        configure_cli_logging,
        load_optional_config,
        logging_options,
        require_output_root,
        resolve_log_level,
        validate_quiet_verbose,
    )
    from data_lake.homogenize.validate_homogenization import run_validation

    @click.command("dl-validate-homogenization")
    @click.argument("output_root", type=click.Path(path_type=Path), required=False)
    @config_option
    @click.option(
        "--transform", "transform_ids", multiple=True,
        help="Transform pack id to validate (default: all bundled).",
    )
    @click.option(
        "--golden", is_flag=True,
        help="Run golden input/output spot checks for --transform packs.",
    )
    @click.option(
        "--product", "products", multiple=True,
        help="Homogenized catalog product to validate under catalogs/.",
    )
    @click.option("--strict", is_flag=True, help="Treat warnings as errors.")
    @logging_options
    def cli(
        output_root: Path | None,
        config_path: Path | None,
        transform_ids: tuple[str, ...],
        golden: bool,
        products: tuple[str, ...],
        strict: bool,
        quiet: bool,
        verbose: bool,
    ) -> None:
        """Validate homogenization transform registry and optional products."""
        validate_quiet_verbose(quiet, verbose)
        cfg = load_optional_config(config_path)
        configure_cli_logging(
            level=resolve_log_level(
                quiet=quiet,
                verbose=verbose,
                config_level=cfg.ingest.log_level if cfg else None,
            ),
            quiet=quiet,
        )
        lake = require_output_root(output_root, cfg) if (output_root or cfg) else None
        if products and lake is None:
            raise click.ClickException(
                "OUTPUT_ROOT is required when validating --product"
            )

        rep = run_validation(
            lake,
            transform_ids=transform_ids or None,
            golden=golden,
            products=products,
            strict=strict,
        )
        print_validation_messages(rep, strict=strict)

except ImportError:
    pass
