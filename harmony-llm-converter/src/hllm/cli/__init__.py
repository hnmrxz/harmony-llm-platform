"""Command-line interface for hllm."""

from __future__ import annotations

import argparse

from hllm import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hllm",
        description="Build deployable HarmonyOS LLM packages from model sources.",
    )
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("doctor", help="Check the local conversion environment.")
    subparsers.add_parser("inspect", help="Inspect a local model directory.")
    subparsers.add_parser("download", help="Download a model from Hugging Face.")
    subparsers.add_parser("build", help="Run the end-to-end conversion pipeline.")
    subparsers.add_parser("validate", help="Validate a conversion artifact or .hllm package.")
    subparsers.add_parser("package", help="Package validated artifacts as .hllm.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    parser.error(f"command '{args.command}' is not implemented yet")
    return 2
