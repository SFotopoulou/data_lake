"""
Ingest command and Slurm script advisor (read-only).

Maps survey inputs (modality, format, file count, lifecycle) to a recommended
``dl-*`` invocation, rationale, optional preflight checks, and Slurm wrapper
text. Never runs ingest or submits jobs.
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from pathlib import Path
from typing import Any, Literal


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent

Modality = Literal["catalog", "spectra", "cutout"]
Lifecycle = Literal["static", "live"]

_KNOWN_SLURM_SCRIPTS: dict[str, str] = {
    "2df": "scripts/slurm_ingest_2df_spectra.sh",
    "6df": "scripts/slurm_ingest_6df_spectra.sh",
}

_DOC_REFS: dict[str, list[str]] = {
    "catalog": [
        "docs/ingest/catalog.md",
        "docs/ingest/batch-and-checkpoints.md",
        "docs/performance.md",
    ],
    "spectra": [
        "docs/ingest/spectra.md",
        "docs/ingest/batch-and-checkpoints.md",
    ],
    "cutout": [
        "docs/ingest/cutouts.md",
        "docs/ingest/batch-and-checkpoints.md",
    ],
}


@dataclass
class IngestRecommendation:
    command: str
    rationale: list[str]
    slurm_script_path: str | None = None
    slurm_script: str | None = None
    preflight: list[str] = field(default_factory=list)
    doc_refs: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "command": self.command,
            "rationale": self.rationale,
            "slurm_script_path": self.slurm_script_path,
            "slurm_script": self.slurm_script,
            "preflight": self.preflight,
            "doc_refs": self.doc_refs,
            "warnings": self.warnings,
        }


def _fmt_lower(fmt: str | None) -> str | None:
    return fmt.strip().lower() if fmt else None


def _inspect_preflight(
    sample_file: Path | str | None,
    file_list: Path | str | None,
) -> tuple[list[str], list[str]]:
    """Optional FITS layout probe; returns (preflight cmds, warnings)."""
    preflight: list[str] = []
    warnings: list[str] = []
    if sample_file is None and file_list is None:
        return preflight, warnings

    from data_lake.ingest.check_fits_table_format import inspect_fits_table_format

    paths: list[Path] = []
    if sample_file is not None:
        paths.append(Path(sample_file))
    elif file_list is not None:
        preflight.append(
            f"dl-check-fits-table-format --file-list {file_list} --json"
        )
        list_path = Path(file_list)
        if list_path.is_file():
            for line in list_path.read_text(encoding="utf-8").splitlines()[:3]:
                p = line.strip()
                if p and not p.startswith("#"):
                    paths.append(Path(p))
                    break
        return preflight, warnings

    if paths:
        preflight.append(f"dl-check-fits-table-format {paths[0]}")
        report = inspect_fits_table_format(paths[0])
        if report.ok and report.format == "packed-vector":
            warnings.append(
                "FITS is packed-vector (colfits): parallel whole-file ingest is "
                "slow and RAM-heavy; convert to row-normal FITS or Parquet first."
            )
        elif not report.ok and report.error:
            warnings.append(f"FITS preflight: {report.error}")
    return preflight, warnings


def _catalog_command(
    survey: str,
    *,
    n_files: int,
    file_list: str | None,
    lifecycle: Lifecycle,
    packed_vector: bool,
    streaming: bool,
) -> tuple[str, list[str]]:
    rationale: list[str] = []
    list_arg = file_list or "files.txt"

    if lifecycle == "live":
        rationale.append(
            "Live/incremental survey: defer expensive metadata rebuild per tile."
        )
        defer = " --defer-finalize --lifecycle live"
    else:
        defer = ""

    if n_files <= 1 and not packed_vector:
        if streaming:
            rationale.append("Single large FITS: use --streaming for bounded RAM.")
            cmd = (
                f"dl-ingest-catalog survey.fits --survey {survey} "
                f"--ra-col ra --dec-col dec --streaming{defer}"
            )
        else:
            rationale.append("Single catalog file: direct ingest.")
            cmd = (
                f"dl-ingest-catalog survey.fits --survey {survey} "
                f"--ra-col ra --dec-col dec{defer}"
            )
        return cmd, rationale

    if packed_vector:
        rationale.append(
            "Many packed-vector FITS files: convert first; batch parallel ingest "
            "loads full tables per worker."
        )
        cmd = (
            f"# Convert to Parquet or row-normal FITS first, then:\n"
            f"dl-ingest-catalog-batch {list_arg} --survey {survey} "
            f"--ra-col ra --dec-col dec --norder 5 --tile-mode append "
            f"--on-duplicate-id skip --n-workers 4{defer}"
        )
        return cmd, rationale

    rationale.append(
        "Many catalog files: parallel decode with single-thread Parquet writer "
        "(dl-ingest-catalog-batch)."
    )
    rationale.append(
        "Run dl-check-fits-table-format on the file list before starting."
    )
    cmd = (
        f"dl-ingest-catalog-batch {list_arg} --survey {survey} "
        f"--ra-col ra --dec-col dec --norder 5 --tile-mode append "
        f"--on-duplicate-id skip --n-workers 8 --files-per-worker 4"
        f"{defer}"
    )
    if lifecycle != "live":
        rationale.append(
            "If a Slurm job is killed mid-run, run dl-finalize-catalog before "
            "dl-refresh-lake-registry."
        )
    return cmd, rationale


def _spectra_command(
    survey: str,
    *,
    fmt: str | None,
    n_files: int,
    file_list: str | None,
) -> tuple[str, list[str], str | None]:
    rationale: list[str] = []
    list_arg = file_list or "spec_files.txt"
    fmt_key = _fmt_lower(fmt)

    if fmt_key == "2df":
        rationale.append("2dF 1-D spectra: dl-ingest-spectra-from-list --fmt 2df.")
        cmd = (
            f"dl-ingest-spectra-from-list {list_arg} --survey {survey} "
            f"--fmt 2df --norder 5 --wavelength-mode shared --on-duplicate skip "
            f"--on-length-mismatch pad --checkpoint checkpoint.json "
            f"--failures-log failures.jsonl --update-catalog"
        )
        return cmd, rationale, _KNOWN_SLURM_SCRIPTS["2df"]

    if fmt_key == "6df":
        rationale.append("6dF VR spectra: dl-ingest-spectra-from-list --fmt 6df.")
        cmd = (
            f"dl-ingest-spectra-from-list {list_arg} --survey {survey} "
            f"--fmt 6df --norder 5 --wavelength-mode shared --on-duplicate skip "
            f"--checkpoint checkpoint.json --failures-log failures.jsonl --update-catalog"
        )
        return cmd, rationale, _KNOWN_SLURM_SCRIPTS["6df"]

    if fmt_key in ("desi_coadd", "desi-coadd", "desi"):
        rationale.append("DESI coadds: dedicated parallel batch ingest.")
        cmd = (
            f"dl-ingest-spectra-batch-desi-coadds --survey {survey} "
            f"--file-list {list_arg} --n-workers 16 --on-duplicate skip"
        )
        return cmd, rationale, None

    if fmt_key in ("sdss_spplate", "spplate"):
        rationale.append("SDSS spPlate: use specobj lookup for fiber→ID mapping.")
        cmd = (
            f"dl-ingest-spectra-from-list {list_arg} --survey {survey} "
            f"--fmt sdss_spplate --specobj-lookup lookup.parquet --on-duplicate skip"
        )
        return cmd, rationale, None

    if n_files > 1:
        rationale.append(
            "Generic spectrum file list: parallel decode with checkpoint/resume."
        )
        cmd = (
            f"dl-ingest-spectra-from-list {list_arg} --survey {survey} "
            f"--norder 5 --on-duplicate skip --on-length-mismatch pad "
            f"--n-workers 4 --checkpoint checkpoint.json --failures-log failures.jsonl "
            f"--update-catalog"
        )
    else:
        rationale.append("Single spectrum FITS: dl-ingest-spectra.")
        cmd = (
            f"dl-ingest-spectra spectrum.fits --survey {survey} --on-duplicate skip"
        )
    return cmd, rationale, None


def _cutout_command(survey: str, *, file_list: str | None) -> tuple[str, list[str]]:
    list_arg = file_list or "cutout_files.txt"
    rationale = [
        "Cutout file list: sequential ingest with --on-duplicate skip (idempotent)."
    ]
    cmd = (
        f"dl-ingest-cutouts-from-list {list_arg} --survey {survey} "
        f"--ra-col ra --dec-col dec --link-id-col source_id --on-duplicate skip"
    )
    return cmd, rationale


def _generate_slurm_script(
    *,
    job_name: str,
    ingest_command: str,
    survey: str,
    state_subdir: str,
    cpus: int = 4,
    mem_gb: int = 32,
    time_limit: str = "7-00:00:00",
    include_finalize: bool = True,
) -> str:
    finalize_block = ""
    if include_finalize:
        finalize_block = textwrap.dedent("""
            echo "Running dl-finalize-catalog …"
            dl-finalize-catalog \\
                --config "${DATA_LAKE_CONFIG}" \\
                --survey "${SURVEY}" \\
                --ra-col ra \\
                --dec-col dec
        """).strip()

    return textwrap.dedent(f"""\
        #!/bin/bash
        # Generated Slurm wrapper for {survey} ingest (data-lake ingest advisor).
        #
        # Required env:
        #   DATA_LAKE_CONFIG  — path to lake_config.toml
        #   LAKE_INGEST_TOKEN — ingest secret
        #   FILE_LIST         — one input path per line (when using *-from-list)
        #
        # Submit:
        #   mkdir -p logs
        #   export DATA_LAKE_CONFIG=/path/to/lake_config.toml
        #   export LAKE_INGEST_TOKEN='secret'
        #   export FILE_LIST=/path/to/files.txt
        #   sbatch this_script.sh

        #SBATCH --job-name={job_name}
        #SBATCH --partition=slow
        #SBATCH --nodes=1
        #SBATCH --ntasks=1
        #SBATCH --cpus-per-task={cpus}
        #SBATCH --mem={mem_gb}G
        #SBATCH --time={time_limit}
        #SBATCH --output=logs/{job_name}-%j.out
        #SBATCH --error=logs/{job_name}-%j.err

        set -euo pipefail

        : "${{DATA_LAKE_CONFIG:?Set DATA_LAKE_CONFIG}}"
        : "${{LAKE_INGEST_TOKEN:?Set LAKE_INGEST_TOKEN}}"

        SCRIPT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
        REPO="${{DATA_LAKE_REPO:-$(cd "${{SCRIPT_DIR}}/.." && pwd)}}"
        SURVEY="${{SURVEY:-{survey}}}"
        CONFIG_DIR="$(dirname "${{DATA_LAKE_CONFIG}}")"
        STATE_DIR="${{CONFIG_DIR}}/ingest_state/{state_subdir}"
        mkdir -p "${{STATE_DIR}}" "${{REPO}}/logs"

        CHECKPOINT="${{STATE_DIR}}/checkpoint.json"
        FAILURES_LOG="${{STATE_DIR}}/failures.jsonl"

        source "${{REPO}}/.venv/bin/activate"

        echo "Ingest start: $(date --iso-8601=seconds) survey=${{SURVEY}}"

        {ingest_command}

        EXIT_CODE=$?
        echo "Ingest end: $(date --iso-8601=seconds) exit=${{EXIT_CODE}}"

        {finalize_block}

        exit "${{EXIT_CODE}}"
    """)


def _load_existing_slurm(relative_path: str) -> str | None:
    path = _repo_root() / relative_path
    if path.is_file():
        return path.read_text(encoding="utf-8")
    return None


def recommend_ingest(
    modality: Modality,
    survey: str,
    *,
    file_list: Path | str | None = None,
    sample_file: Path | str | None = None,
    fmt: str | None = None,
    lifecycle: Lifecycle = "static",
    n_files: int = 1,
    total_size_gb: float | None = None,
    streaming: bool = False,
) -> IngestRecommendation:
    """
    Recommend a ``dl-*`` ingest command and Slurm wrapper for the given inputs.

    Parameters
    ----------
    n_files
        Number of input files (use >1 to trigger batch/list paths).
    streaming
        For single-file catalog FITS, suggest ``--streaming`` when True or when
        ``total_size_gb`` exceeds ~8 GiB.
    """
    file_list_str = str(file_list) if file_list is not None else None
    warnings: list[str] = []
    preflight, inspect_warnings = _inspect_preflight(sample_file, file_list)
    warnings.extend(inspect_warnings)

    packed_vector = any("packed-vector" in w for w in inspect_warnings)

    if modality == "catalog":
        if total_size_gb is not None and total_size_gb > 8:
            streaming = True
        preflight.append(f"dl-recommend-catalog-norder --file-list {file_list_str or 'files.txt'}")
        command, rationale = _catalog_command(
            survey,
            n_files=n_files,
            file_list=file_list_str,
            lifecycle=lifecycle,
            packed_vector=packed_vector,
            streaming=streaming,
        )
        slurm_path = None
        ingest_for_slurm = command.split("\n")[-1] if "\n" in command else command
        slurm_body = _generate_slurm_script(
            job_name=f"dl-{survey.lower()[:20]}-catalog",
            ingest_command=ingest_for_slurm,
            survey=survey,
            state_subdir=f"{survey.lower()}/catalog",
        )
    elif modality == "spectra":
        command, rationale, slurm_path = _spectra_command(
            survey, fmt=fmt, n_files=n_files, file_list=file_list_str,
        )
        if slurm_path:
            slurm_body = _load_existing_slurm(slurm_path)
        else:
            slurm_body = _generate_slurm_script(
                job_name=f"dl-{survey.lower()[:20]}-spectra",
                ingest_command=command,
                survey=survey,
                state_subdir=f"{survey.lower()}/spectra",
            )
    else:
        command, rationale = _cutout_command(survey, file_list=file_list_str)
        slurm_path = None
        slurm_body = _generate_slurm_script(
            job_name=f"dl-{survey.lower()[:20]}-cutout",
            ingest_command=command,
            survey=survey,
            state_subdir=f"{survey.lower()}/cutout",
            include_finalize=False,
        )

    if n_files > 1000:
        rationale.append(
            f"Large file list ({n_files:,} files): use checkpoint + failures log; "
            "re-submit the same Slurm job to resume."
        )

    return IngestRecommendation(
        command=command,
        rationale=rationale,
        slurm_script_path=slurm_path,
        slurm_script=slurm_body,
        preflight=preflight,
        doc_refs=_DOC_REFS.get(modality, []),
        warnings=warnings,
    )
