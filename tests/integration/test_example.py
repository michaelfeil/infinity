"""
Example integration tests for the Infinity project.

These tests demonstrate how to write integration tests that verify
multiple components working together.
"""

import pytest
from pathlib import Path
from typing import Dict, Any


@pytest.mark.integration
class TestIntegrationExample:
    """Example integration test suite."""

    def test_integration_fixture_access(
        self, temp_dir: Path, mock_config: Dict[str, Any]
    ) -> None:
        """Verify that integration tests can access shared fixtures."""
        assert temp_dir.exists()
        assert mock_config["debug"] is True

        # Simulate integration between file system and configuration
        config_file = temp_dir / "config.json"
        config_file.write_text('{"key": "value"}')
        assert config_file.exists()

    def test_project_structure_integration(self, project_root: Path) -> None:
        """Verify that project structure is correctly integrated."""
        # Check that main project files exist
        assert (project_root / "README.md").exists()
        assert (project_root / "pyproject.toml").exists()
        assert (project_root / "tests").is_dir()

        # Check that libs directory exists
        libs_dir = project_root / "libs"
        if libs_dir.exists():
            # Verify the monorepo structure
            assert libs_dir.is_dir()
            # Common lib directories in Infinity project
            expected_libs = ["infinity_emb", "client_infinity", "embed_package"]
            for lib in expected_libs:
                lib_path = libs_dir / lib
                if lib_path.exists():
                    assert lib_path.is_dir()
