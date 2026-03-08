"""CLI interface for the MAGI research pipeline."""

import argparse
import sys


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser for the CLI."""
    parser = argparse.ArgumentParser(
        prog="magi-system",
        description="MAGI Multi-Agent Research Pipeline",
    )
    parser.add_argument(
        "query",
        nargs="?",
        help="Research query to process",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Save report to file path",
    )
    parser.add_argument(
        "--config",
        "-c",
        help="Path to TOML configuration file",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable debug logging",
    )
    return parser


def main(args=None):
    """Main entry point for the CLI."""
    parser = create_parser()
    parsed = parser.parse_args(args)

    if not parsed.query:
        parser.print_help()
        sys.exit(1)

    # Import here to avoid slow imports when just showing help
    from .config import load_config

    if parsed.verbose:
        import logging

        logging.basicConfig(level=logging.DEBUG)

    print("MAGI Research Pipeline")
    print(f"Query: {parsed.query}")
    print(
        "Pipeline execution requires ADK runtime. "
        "Use `adk web python/magi_system` for full execution."
    )

    if parsed.output:
        print(f"Output will be saved to: {parsed.output}")

    config = load_config()
    print(
        f"Config loaded: {config.max_research_iterations} max iterations, "
        f"{config.max_sub_questions} max sub-questions"
    )
