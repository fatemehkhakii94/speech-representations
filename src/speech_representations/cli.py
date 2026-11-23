"""Single command-line interface for every project workflow."""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import sys
from collections.abc import Iterable, Sequence

import torch

from speech_representations import __version__
from speech_representations.analysis.cka import run_cka
from speech_representations.analysis.downstream import run_downstream_emotion
from speech_representations.analysis.plots import generate_figures
from speech_representations.config import ProjectConfig, load_config
from speech_representations.data.datasets import prepare_data
from speech_representations.data.records import read_audio_records
from speech_representations.data.validation import validate_prepared_data
from speech_representations.features.store import extract_features
from speech_representations.probing.experiment import run_probing


def _selection(value: str, available: Iterable[str]) -> list[str]:
    choices = list(available)
    if value == "all":
        return choices
    requested = [item.strip() for item in value.split(",") if item.strip()]
    unknown = set(requested) - set(choices)
    if unknown:
        raise ValueError(f"Unknown values {sorted(unknown)}; choose from {choices} or 'all'")
    return requested


def _doctor(config: ProjectConfig) -> int:
    checks = {
        "python_supported": (3, 10) <= sys.version_info[:2] < (3, 13),
        "configuration_loaded": config.schema_version == 1,
        "git_available": shutil.which("git") is not None,
        "cuda_available": torch.cuda.is_available(),
    }
    report = {
        "checks": checks,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "config": str(config.root / "configs" / "project.yaml"),
    }
    print(json.dumps(report, indent=2))
    required = ["python_supported", "configuration_loaded", "git_available"]
    return 0 if all(checks[key] for key in required) else 1


def _extract(
    config: ProjectConfig, models: Sequence[str], datasets: Sequence[str], overwrite: bool
) -> None:
    for dataset in datasets:
        manifest = config.paths.data / "manifests" / "datasets" / f"{dataset}.jsonl"
        if not manifest.is_file():
            raise FileNotFoundError(
                f"Dataset manifest missing: {manifest}; run `speech-repr prepare` first"
            )
        records = read_audio_records(manifest)
        for model in models:
            print(extract_features(config, model, dataset, records, overwrite=overwrite))


def _probe(config: ProjectConfig, models: list[str], tasks: list[str], overwrite: bool) -> None:
    for task in tasks:
        for model in models:
            print(run_probing(config, model, task, overwrite=overwrite))


def _cka_all(config: ProjectConfig, models: list[str]) -> None:
    for index, first in enumerate(models):
        for second in models[index:]:
            print(run_cka(config, first, second))


def _reproduce(config: ProjectConfig, overwrite: bool) -> None:
    task_names = list(config.tasks)
    model_names = list(config.models)
    counts = prepare_data(config, task_names)
    print(f"Prepared tasks: {counts}")
    datasets = sorted({config.task(task).dataset for task in task_names})
    _extract(config, model_names, datasets, overwrite)
    _probe(config, model_names, task_names, overwrite)
    _cka_all(config, model_names)
    print(run_downstream_emotion(config, overwrite=overwrite))
    for path in generate_figures(config):
        print(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="speech-repr", description=__doc__)
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--config", default="configs/project.yaml", help="Canonical project YAML")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("doctor", help="Validate the local runtime")

    prepare = subparsers.add_parser("prepare", help="Acquire datasets and write split manifests")
    prepare.add_argument("--tasks", default="all", help="Comma-separated task names or all")
    subparsers.add_parser("validate", help="Audit prepared manifests and leakage constraints")

    extract = subparsers.add_parser("extract", help="Build resumable frozen feature caches")
    extract.add_argument("--models", default="all", help="Comma-separated model names or all")
    extract.add_argument("--datasets", default="all", help="Comma-separated dataset names or all")
    extract.add_argument("--overwrite", action="store_true")

    probe = subparsers.add_parser("probe", help="Run layer-wise probing")
    probe.add_argument("--models", default="all")
    probe.add_argument("--tasks", default="all")
    probe.add_argument("--overwrite", action="store_true")

    cka = subparsers.add_parser("cka", help="Compute ID-aligned linear CKA")
    cka.add_argument("--models", default="all")

    downstream = subparsers.add_parser(
        "downstream", help="Run unbiased emotion layer-selection validation"
    )
    downstream.add_argument("--model", default=None)
    downstream.add_argument("--overwrite", action="store_true")

    subparsers.add_parser("figures", help="Regenerate all figures from results")
    reproduce = subparsers.add_parser("reproduce", help="Run the complete corrected study")
    reproduce.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    if args.command == "doctor":
        return _doctor(config)
    if args.command == "prepare":
        tasks = _selection(args.tasks, config.tasks)
        print(json.dumps(prepare_data(config, tasks), indent=2))
    elif args.command == "validate":
        print(json.dumps(validate_prepared_data(config), indent=2))
    elif args.command == "extract":
        models = _selection(args.models, config.models)
        available_datasets = sorted({task.dataset for task in config.tasks.values()})
        datasets = _selection(args.datasets, available_datasets)
        _extract(config, models, datasets, args.overwrite)
    elif args.command == "probe":
        _probe(
            config,
            _selection(args.models, config.models),
            _selection(args.tasks, config.tasks),
            args.overwrite,
        )
    elif args.command == "cka":
        _cka_all(config, _selection(args.models, config.models))
    elif args.command == "downstream":
        print(run_downstream_emotion(config, args.model, args.overwrite))
    elif args.command == "figures":
        for path in generate_figures(config):
            print(path)
    elif args.command == "reproduce":
        _reproduce(config, args.overwrite)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
