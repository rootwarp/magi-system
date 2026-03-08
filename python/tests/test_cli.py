"""Tests for MAGI-036: CLI interface."""

import argparse
import importlib


class TestCreateParser:
    """Tests for the create_parser function."""

    def test_create_parser_returns_argument_parser(self) -> None:
        from magi_system.cli import create_parser

        parser = create_parser()
        assert isinstance(parser, argparse.ArgumentParser)

    def test_parser_prog_name(self) -> None:
        from magi_system.cli import create_parser

        parser = create_parser()
        assert parser.prog == "magi-system"

    def test_parser_accepts_query_argument(self) -> None:
        from magi_system.cli import create_parser

        parser = create_parser()
        parsed = parser.parse_args(["test query"])
        assert parsed.query == "test query"

    def test_parser_query_is_optional(self) -> None:
        from magi_system.cli import create_parser

        parser = create_parser()
        parsed = parser.parse_args([])
        assert parsed.query is None

    def test_parser_accepts_output_flag(self) -> None:
        from magi_system.cli import create_parser

        parser = create_parser()
        parsed = parser.parse_args(["some query", "--output", "report.md"])
        assert parsed.output == "report.md"

    def test_parser_accepts_output_short_flag(self) -> None:
        from magi_system.cli import create_parser

        parser = create_parser()
        parsed = parser.parse_args(["some query", "-o", "report.md"])
        assert parsed.output == "report.md"

    def test_parser_accepts_config_flag(self) -> None:
        from magi_system.cli import create_parser

        parser = create_parser()
        parsed = parser.parse_args(["some query", "--config", "config.toml"])
        assert parsed.config == "config.toml"

    def test_parser_accepts_config_short_flag(self) -> None:
        from magi_system.cli import create_parser

        parser = create_parser()
        parsed = parser.parse_args(["some query", "-c", "config.toml"])
        assert parsed.config == "config.toml"

    def test_parser_accepts_verbose_flag(self) -> None:
        from magi_system.cli import create_parser

        parser = create_parser()
        parsed = parser.parse_args(["some query", "--verbose"])
        assert parsed.verbose is True

    def test_parser_accepts_verbose_short_flag(self) -> None:
        from magi_system.cli import create_parser

        parser = create_parser()
        parsed = parser.parse_args(["some query", "-v"])
        assert parsed.verbose is True

    def test_parser_verbose_defaults_false(self) -> None:
        from magi_system.cli import create_parser

        parser = create_parser()
        parsed = parser.parse_args(["some query"])
        assert parsed.verbose is False

    def test_parser_output_defaults_none(self) -> None:
        from magi_system.cli import create_parser

        parser = create_parser()
        parsed = parser.parse_args(["some query"])
        assert parsed.output is None

    def test_parser_config_defaults_none(self) -> None:
        from magi_system.cli import create_parser

        parser = create_parser()
        parsed = parser.parse_args(["some query"])
        assert parsed.config is None

    def test_parser_description_contains_magi(self) -> None:
        from magi_system.cli import create_parser

        parser = create_parser()
        assert "MAGI" in parser.description


class TestMain:
    """Tests for the main entry point."""

    def test_main_no_query_exits_with_error(self) -> None:
        import pytest

        from magi_system.cli import main

        with pytest.raises(SystemExit) as exc_info:
            main([])
        assert exc_info.value.code == 1

    def test_main_with_query_prints_info(self, capsys) -> None:
        from magi_system.cli import main

        main(["What is quantum computing?"])
        captured = capsys.readouterr()
        assert "MAGI Research Pipeline" in captured.out
        assert "What is quantum computing?" in captured.out

    def test_main_with_verbose_enables_debug(self, capsys) -> None:
        from magi_system.cli import main

        main(["test query", "--verbose"])
        captured = capsys.readouterr()
        assert "MAGI Research Pipeline" in captured.out

    def test_main_with_output_shows_path(self, capsys) -> None:
        from magi_system.cli import main

        main(["test query", "--output", "report.md"])
        captured = capsys.readouterr()
        assert "report.md" in captured.out

    def test_main_shows_config_info(self, capsys) -> None:
        from magi_system.cli import main

        main(["test query"])
        captured = capsys.readouterr()
        assert "Config loaded" in captured.out


class TestMainModule:
    """Tests for __main__.py entry point."""

    def test_main_module_file_exists(self) -> None:
        import pathlib

        import magi_system

        pkg_dir = pathlib.Path(magi_system.__file__).parent
        main_file = pkg_dir / "__main__.py"
        assert main_file.exists()

    def test_main_module_imports_main(self) -> None:
        import pathlib

        import magi_system

        pkg_dir = pathlib.Path(magi_system.__file__).parent
        main_file = pkg_dir / "__main__.py"
        content = main_file.read_text()
        assert "from .cli import main" in content

    def test_main_module_can_be_found(self) -> None:
        spec = importlib.util.find_spec("magi_system.__main__")
        assert spec is not None
