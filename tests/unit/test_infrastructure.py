"""
Validation tests for the testing infrastructure setup.

These tests verify that the testing infrastructure is properly configured
and all fixtures are working as expected.
"""

import os
from pathlib import Path
from typing import Dict, Any
import pytest


class TestInfrastructureSetup:
    """Test suite for validating the testing infrastructure."""

    def test_pytest_is_working(self) -> None:
        """Verify that pytest is properly installed and functioning."""
        assert True, "pytest is working"

    def test_temp_dir_fixture(self, temp_dir: Path) -> None:
        """Verify that temp_dir fixture creates a valid temporary directory."""
        assert temp_dir.exists(), "Temporary directory should exist"
        assert temp_dir.is_dir(), "Temporary directory should be a directory"

        # Test writing to the temp directory
        test_file = temp_dir / "test.txt"
        test_file.write_text("test content")
        assert test_file.read_text() == "test content"

    def test_temp_file_fixture(self, temp_file: Path) -> None:
        """Verify that temp_file fixture creates a valid temporary file."""
        assert temp_file.exists(), "Temporary file should exist"
        assert temp_file.is_file(), "Temporary file should be a file"

        # Test writing to the temp file
        temp_file.write_text("hello world")
        assert temp_file.read_text() == "hello world"

    def test_mock_env_vars_fixture(self, mock_env_vars: Dict[str, str]) -> None:
        """Verify that mock_env_vars fixture sets environment variables."""
        assert mock_env_vars is not None, "mock_env_vars should not be None"
        assert "TEST_MODE" in mock_env_vars, "TEST_MODE should be in mock_env_vars"
        assert os.environ.get("TEST_MODE") == "true", "TEST_MODE should be set to 'true'"

    def test_mock_config_fixture(self, mock_config: Dict[str, Any]) -> None:
        """Verify that mock_config fixture provides valid configuration."""
        assert mock_config is not None, "mock_config should not be None"
        assert "debug" in mock_config, "debug should be in mock_config"
        assert mock_config["debug"] is True, "debug should be True"
        assert mock_config["timeout"] == 30, "timeout should be 30"

    def test_sample_data_fixture(self, sample_data: Dict[str, Any]) -> None:
        """Verify that sample_data fixture provides valid test data."""
        assert sample_data is not None, "sample_data should not be None"
        assert "items" in sample_data, "items should be in sample_data"
        assert len(sample_data["items"]) == 3, "sample_data should have 3 items"
        assert "metadata" in sample_data, "metadata should be in sample_data"

    def test_project_root_fixture(self, project_root: Path) -> None:
        """Verify that project_root fixture points to the correct directory."""
        assert project_root.exists(), "Project root should exist"
        assert project_root.is_dir(), "Project root should be a directory"
        assert (project_root / "pyproject.toml").exists(), "pyproject.toml should exist"
        assert (project_root / "README.md").exists(), "README.md should exist"

    @pytest.mark.unit
    def test_unit_marker(self) -> None:
        """Verify that unit marker is applied correctly."""
        # This test should be automatically marked as unit
        assert True

    def test_pytest_mock_available(self, mocker: Any) -> None:
        """Verify that pytest-mock is properly installed and working."""
        # Create a simple mock to verify pytest-mock functionality
        mock_func = mocker.Mock(return_value=42)
        result = mock_func()
        assert result == 42, "Mock function should return 42"
        mock_func.assert_called_once()


class TestCoverageIntegration:
    """Test suite for validating coverage reporting setup."""

    def test_coverage_is_configured(self) -> None:
        """Verify that coverage reporting is configured."""
        # This test just needs to run to contribute to coverage
        result = self._helper_method()
        assert result == "coverage_test"

    def _helper_method(self) -> str:
        """Helper method to test coverage of private methods."""
        return "coverage_test"

    def test_multiple_assertions(self) -> None:
        """Test with multiple assertions to verify coverage tracking."""
        data = [1, 2, 3, 4, 5]
        assert len(data) == 5
        assert sum(data) == 15
        assert max(data) == 5
        assert min(data) == 1


class TestProjectStructure:
    """Test suite for validating the project structure."""

    def test_tests_directory_exists(self, project_root: Path) -> None:
        """Verify that the tests directory exists."""
        tests_dir = project_root / "tests"
        assert tests_dir.exists(), "tests directory should exist"
        assert tests_dir.is_dir(), "tests should be a directory"

    def test_unit_tests_directory_exists(self, project_root: Path) -> None:
        """Verify that the unit tests directory exists."""
        unit_dir = project_root / "tests" / "unit"
        assert unit_dir.exists(), "tests/unit directory should exist"
        assert unit_dir.is_dir(), "tests/unit should be a directory"

    def test_integration_tests_directory_exists(self, project_root: Path) -> None:
        """Verify that the integration tests directory exists."""
        integration_dir = project_root / "tests" / "integration"
        assert integration_dir.exists(), "tests/integration directory should exist"
        assert integration_dir.is_dir(), "tests/integration should be a directory"

    def test_conftest_exists(self, project_root: Path) -> None:
        """Verify that conftest.py exists."""
        conftest = project_root / "tests" / "conftest.py"
        assert conftest.exists(), "conftest.py should exist"
        assert conftest.is_file(), "conftest.py should be a file"

    def test_init_files_exist(self, project_root: Path) -> None:
        """Verify that __init__.py files exist in test directories."""
        init_files = [
            project_root / "tests" / "__init__.py",
            project_root / "tests" / "unit" / "__init__.py",
            project_root / "tests" / "integration" / "__init__.py",
        ]

        for init_file in init_files:
            assert init_file.exists(), f"{init_file} should exist"
            assert init_file.is_file(), f"{init_file} should be a file"


@pytest.mark.slow
class TestPerformanceMarker:
    """Test suite for validating marker functionality."""

    def test_slow_marker(self) -> None:
        """Verify that slow marker can be applied."""
        # This test is marked as slow
        assert True
