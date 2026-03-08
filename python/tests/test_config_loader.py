"""Tests for MAGI-035: config_loader.py module."""

import os
import textwrap
from pathlib import Path
from unittest.mock import patch


class TestLoadConfigFromFileDefaults:
    """Tests for loading config with no file (defaults only)."""

    def test_returns_pipeline_config(self) -> None:
        from magi_system.config import PipelineConfig
        from magi_system.config_loader import load_config_from_file

        config = load_config_from_file()
        assert isinstance(config, PipelineConfig)

    def test_returns_default_values_when_no_file(self) -> None:
        from magi_system.config_loader import load_config_from_file

        config = load_config_from_file()
        assert config.max_research_iterations == 3
        assert config.max_discussion_rounds == 3
        assert config.max_sub_questions == 15
        assert config.divergence_threshold == 0.7

    def test_none_path_returns_defaults(self) -> None:
        from magi_system.config_loader import load_config_from_file

        config = load_config_from_file(config_path=None)
        assert config.max_research_iterations == 3

    def test_nonexistent_path_returns_defaults(self) -> None:
        from magi_system.config_loader import load_config_from_file

        config = load_config_from_file(config_path="/nonexistent/path.toml")
        assert config.max_research_iterations == 3


class TestLoadConfigFromTomlFile:
    """Tests for loading config from TOML files."""

    def test_overrides_pipeline_values(self, tmp_path: Path) -> None:
        from magi_system.config_loader import load_config_from_file

        config_file = tmp_path / "config.toml"
        config_file.write_text(textwrap.dedent("""\
            [pipeline]
            max_research_iterations = 10
            max_discussion_rounds = 5
            max_sub_questions = 20
            divergence_threshold = 0.5
        """))

        config = load_config_from_file(str(config_file))
        assert config.max_research_iterations == 10
        assert config.max_discussion_rounds == 5
        assert config.max_sub_questions == 20
        assert config.divergence_threshold == 0.5

    def test_partial_config_only_overrides_specified(self, tmp_path: Path) -> None:
        from magi_system.config_loader import load_config_from_file

        config_file = tmp_path / "config.toml"
        config_file.write_text(textwrap.dedent("""\
            [pipeline]
            max_research_iterations = 7
        """))

        config = load_config_from_file(str(config_file))
        assert config.max_research_iterations == 7
        # Defaults preserved
        assert config.max_discussion_rounds == 3
        assert config.max_sub_questions == 15
        assert config.divergence_threshold == 0.7

    def test_overrides_orchestrator_model(self, tmp_path: Path) -> None:
        from magi_system.config_loader import load_config_from_file

        config_file = tmp_path / "config.toml"
        config_file.write_text('orchestrator_model = "custom-orchestrator"\n')

        config = load_config_from_file(str(config_file))
        assert config.orchestrator.model == "custom-orchestrator"

    def test_overrides_synthesis_model(self, tmp_path: Path) -> None:
        from magi_system.config_loader import load_config_from_file

        config_file = tmp_path / "config.toml"
        config_file.write_text('synthesis_model = "custom-synthesis"\n')

        config = load_config_from_file(str(config_file))
        assert config.synthesis.model == "custom-synthesis"

    def test_empty_toml_file_returns_defaults(self, tmp_path: Path) -> None:
        from magi_system.config_loader import load_config_from_file

        config_file = tmp_path / "config.toml"
        config_file.write_text("")

        config = load_config_from_file(str(config_file))
        assert config.max_research_iterations == 3
        assert config.max_discussion_rounds == 3


class TestEnvVarOverrides:
    """Tests for environment variable overrides."""

    def test_env_overrides_defaults(self) -> None:
        from magi_system.config_loader import load_config_from_file

        with patch.dict(os.environ, {"MAGI_MAX_ITERATIONS": "8"}):
            config = load_config_from_file()
        assert config.max_research_iterations == 8

    def test_env_overrides_file_config(self, tmp_path: Path) -> None:
        from magi_system.config_loader import load_config_from_file

        config_file = tmp_path / "config.toml"
        config_file.write_text(textwrap.dedent("""\
            [pipeline]
            max_research_iterations = 10
        """))

        with patch.dict(os.environ, {"MAGI_MAX_ITERATIONS": "99"}):
            config = load_config_from_file(str(config_file))
        assert config.max_research_iterations == 99

    def test_env_orchestrator_model(self) -> None:
        from magi_system.config_loader import load_config_from_file

        with patch.dict(os.environ, {"MAGI_ORCHESTRATOR_MODEL": "custom-model"}):
            config = load_config_from_file()
        assert config.orchestrator.model == "custom-model"

    def test_env_max_sub_questions(self) -> None:
        from magi_system.config_loader import load_config_from_file

        with patch.dict(os.environ, {"MAGI_MAX_SUB_QUESTIONS": "25"}):
            config = load_config_from_file()
        assert config.max_sub_questions == 25

    def test_env_divergence_threshold(self) -> None:
        from magi_system.config_loader import load_config_from_file

        with patch.dict(os.environ, {"MAGI_DIVERGENCE_THRESHOLD": "0.9"}):
            config = load_config_from_file()
        assert config.divergence_threshold == 0.9

    def test_multiple_env_vars(self) -> None:
        from magi_system.config_loader import load_config_from_file

        env = {
            "MAGI_MAX_ITERATIONS": "5",
            "MAGI_MAX_SUB_QUESTIONS": "30",
            "MAGI_DIVERGENCE_THRESHOLD": "0.3",
        }
        with patch.dict(os.environ, env):
            config = load_config_from_file()
        assert config.max_research_iterations == 5
        assert config.max_sub_questions == 30
        assert config.divergence_threshold == 0.3


class TestValidation:
    """Tests for config validation."""

    def test_invalid_max_research_iterations(self, tmp_path: Path) -> None:
        import pytest

        from magi_system.config_loader import load_config_from_file

        config_file = tmp_path / "config.toml"
        config_file.write_text(textwrap.dedent("""\
            [pipeline]
            max_research_iterations = 0
        """))

        with pytest.raises(ValueError, match="max_research_iterations must be >= 1"):
            load_config_from_file(str(config_file))

    def test_invalid_max_sub_questions(self, tmp_path: Path) -> None:
        import pytest

        from magi_system.config_loader import load_config_from_file

        config_file = tmp_path / "config.toml"
        config_file.write_text(textwrap.dedent("""\
            [pipeline]
            max_sub_questions = -1
        """))

        with pytest.raises(ValueError, match="max_sub_questions must be >= 1"):
            load_config_from_file(str(config_file))

    def test_divergence_threshold_too_high(self, tmp_path: Path) -> None:
        import pytest

        from magi_system.config_loader import load_config_from_file

        config_file = tmp_path / "config.toml"
        config_file.write_text(textwrap.dedent("""\
            [pipeline]
            divergence_threshold = 1.5
        """))

        with pytest.raises(ValueError, match="divergence_threshold must be between"):
            load_config_from_file(str(config_file))

    def test_divergence_threshold_negative(self, tmp_path: Path) -> None:
        import pytest

        from magi_system.config_loader import load_config_from_file

        config_file = tmp_path / "config.toml"
        config_file.write_text(textwrap.dedent("""\
            [pipeline]
            divergence_threshold = -0.1
        """))

        with pytest.raises(ValueError, match="divergence_threshold must be between"):
            load_config_from_file(str(config_file))

    def test_env_var_invalid_value_raises(self) -> None:
        import pytest

        from magi_system.config_loader import load_config_from_file

        with patch.dict(os.environ, {"MAGI_MAX_ITERATIONS": "0"}):
            with pytest.raises(ValueError, match="max_research_iterations must be >= 1"):
                load_config_from_file()

    def test_boundary_values_valid(self, tmp_path: Path) -> None:
        from magi_system.config_loader import load_config_from_file

        config_file = tmp_path / "config.toml"
        config_file.write_text(textwrap.dedent("""\
            [pipeline]
            max_research_iterations = 1
            max_sub_questions = 1
            divergence_threshold = 0.0
        """))

        config = load_config_from_file(str(config_file))
        assert config.max_research_iterations == 1
        assert config.max_sub_questions == 1
        assert config.divergence_threshold == 0.0

    def test_threshold_at_one_is_valid(self, tmp_path: Path) -> None:
        from magi_system.config_loader import load_config_from_file

        config_file = tmp_path / "config.toml"
        config_file.write_text(textwrap.dedent("""\
            [pipeline]
            divergence_threshold = 1.0
        """))

        config = load_config_from_file(str(config_file))
        assert config.divergence_threshold == 1.0
