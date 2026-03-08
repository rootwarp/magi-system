"""Tests for MAGI-001: Module directory structure."""

import importlib
import os

import pytest

MAGI_SYSTEM_DIR = os.path.join(os.path.dirname(__file__), "..", "magi_system")

REQUIRED_MODULES = [
    "clarification",
    "planning",
    "search",
    "reflection",
    "consensus",
    "synthesis",
]


class TestModuleDirectoryStructure:
    """Verify that all required module directories exist with __init__.py files."""

    @pytest.mark.parametrize("module_name", REQUIRED_MODULES)
    def test_module_directory_exists(self, module_name: str) -> None:
        module_dir = os.path.join(MAGI_SYSTEM_DIR, module_name)
        assert os.path.isdir(
            module_dir
        ), f"Module directory '{module_name}' does not exist"

    @pytest.mark.parametrize("module_name", REQUIRED_MODULES)
    def test_module_has_init_file(self, module_name: str) -> None:
        init_file = os.path.join(MAGI_SYSTEM_DIR, module_name, "__init__.py")
        assert os.path.isfile(
            init_file
        ), f"Module '{module_name}' is missing __init__.py"

    @pytest.mark.parametrize("module_name", REQUIRED_MODULES)
    def test_module_is_importable(self, module_name: str) -> None:
        """Each new module should be importable as magi_system.<module>."""
        mod = importlib.import_module(f"magi_system.{module_name}")
        assert mod is not None


class TestExistingImportsUnbroken:
    """Verify that existing imports continue to work after adding new modules."""

    def test_root_package_importable(self) -> None:
        import magi_system

        assert hasattr(magi_system, "root_agent")

    def test_discussion_module_importable(self) -> None:
        from magi_system.discussion import loop_agent, discussion_round

        assert loop_agent is not None
        assert discussion_round is not None

    def test_sub_agents_importable(self) -> None:
        from magi_system import sub_agents

        assert sub_agents is not None

    def test_tools_importable(self) -> None:
        from magi_system import tools

        assert tools is not None
