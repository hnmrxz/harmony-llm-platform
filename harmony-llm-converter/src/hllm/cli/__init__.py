"""Command-line interface for the converter."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from hllm import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hllm", description="HarmonyOS LLM Converter")
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command")

    doctor = sub.add_parser("doctor", help="check the Ubuntu conversion environment")
    doctor.set_defaults(handler=_doctor)

    inspect = sub.add_parser("inspect", help="inspect a local Hugging Face model")
    inspect.add_argument("model", type=Path)
    inspect.set_defaults(handler=_inspect)

    download = sub.add_parser("download", help="download a model from Hugging Face")
    download.add_argument("repo")
    download.add_argument("--output", type=Path, default=Path("./models"))
    download.add_argument("--revision")
    download.add_argument("--token")
    download.set_defaults(handler=_download)

    build = sub.add_parser("build", help="build a deployable .hllm package")
    build.add_argument("model", help="local model directory or Hugging Face repo id")
    build.add_argument("--target", required=True)
    build.add_argument("--quant", default="int4")
    build.add_argument("--context", type=int)
    build.add_argument("--output", type=Path, default=Path("./build"))
    build.add_argument("--model-cache", type=Path, default=Path("./models"))
    build.add_argument("--profile", type=Path)
    build.add_argument("--dry-run", action="store_true")
    build.set_defaults(handler=_build)

    validate = sub.add_parser("validate", help="validate a .hllm package")
    validate.add_argument("package", type=Path)
    validate.set_defaults(handler=_validate)
    return parser


def _doctor(_: argparse.Namespace) -> int:
    from hllm.doctor import run_checks
    failed_required = False
    for check in run_checks():
        status = "OK" if check.ok else "FAIL"
        suffix = "" if check.required else " (optional)"
        print(f"{status:4} {check.name}: {check.detail}{suffix}")
        failed_required |= check.required and not check.ok
    return int(failed_required)


def _inspect(args: argparse.Namespace) -> int:
    from dataclasses import asdict
    from hllm.models.detector import inspect_model
    print(json.dumps(asdict(inspect_model(args.model)), ensure_ascii=False, indent=2))
    return 0


def _download(args: argparse.Namespace) -> int:
    from hllm.download.huggingface import download_model
    path = download_model(args.repo, args.output / args.repo.replace("/", "__"), revision=args.revision, token=args.token)
    print(path)
    return 0


def _build(args: argparse.Namespace) -> int:
    from hllm.pipeline.build import BuildRequest, build
    result = build(BuildRequest(source=args.model, target_chip=args.target, quantization=args.quant,
                                context_length=args.context, output_dir=args.output, profile=args.profile,
                                dry_run=args.dry_run, model_cache_dir=args.model_cache))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "success" else 2


def _validate(args: argparse.Namespace) -> int:
    from hllm.validation import validate_hllm
    print(json.dumps(validate_hllm(args.package), ensure_ascii=False, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "handler"):
        parser.print_help()
        return 0
    try:
        return int(args.handler(args))
    except Exception as exc:
        print(f"ERROR {type(exc).__name__}: {exc}")
        return 2
