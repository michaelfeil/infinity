"""
Shared pytest fixtures and configuration for Infinity testing infrastructure.

This module provides common test fixtures that can be used across all tests
in the Infinity monorepo, including temporary directories, mock configurations,
and other shared test utilities.
"""

import os
import tempfile
from pathlib import Path
from typing import Dict, Any, Generator
import pytest


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """
    Provide a temporary directory for tests.

    The directory is automatically cleaned up after the test completes.

    Yields:
        Path: Temporary directory path

    Example:
        def test_file_operations(temp_dir):
            test_file = temp_dir / "test.txt"
            test_file.write_text("hello")
            assert test_file.read_text() == "hello"
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def temp_file(temp_dir: Path) -> Generator[Path, None, None]:
    """
    Provide a temporary file for tests.

    Args:
        temp_dir: Temporary directory fixture

    Yields:
        Path: Temporary file path

    Example:
        def test_read_file(temp_file):
            temp_file.write_text("test content")
            assert temp_file.read_text() == "test content"
    """
    temp_file_path = temp_dir / "test_file.txt"
    temp_file_path.touch()
    yield temp_file_path


@pytest.fixture
def mock_env_vars(monkeypatch: pytest.MonkeyPatch) -> Dict[str, str]:
    """
    Provide mock environment variables for tests.

    Args:
        monkeypatch: Pytest monkeypatch fixture

    Returns:
        Dict[str, str]: Dictionary of mock environment variables

    Example:
        def test_env_config(mock_env_vars):
            assert os.environ.get("TEST_MODE") == "true"
    """
    env_vars = {
        "TEST_MODE": "true",
        "PYTHONPATH": str(Path(__file__).parent.parent),
    }

    for key, value in env_vars.items():
        monkeypatch.setenv(key, value)

    return env_vars


@pytest.fixture
def mock_config() -> Dict[str, Any]:
    """
    Provide a mock configuration dictionary for tests.

    Returns:
        Dict[str, Any]: Mock configuration with common test settings

    Example:
        def test_config_loading(mock_config):
            assert mock_config["debug"] is True
            assert mock_config["timeout"] == 30
    """
    return {
        "debug": True,
        "timeout": 30,
        "max_retries": 3,
        "batch_size": 10,
        "cache_enabled": False,
    }


@pytest.fixture
def sample_data() -> Dict[str, Any]:
    """
    Provide sample data for testing.

    Returns:
        Dict[str, Any]: Sample data structure for tests

    Example:
        def test_data_processing(sample_data):
            assert len(sample_data["items"]) == 3
            assert sample_data["metadata"]["version"] == "1.0"
    """
    return {
        "items": [
            {"id": 1, "name": "item1", "value": 100},
            {"id": 2, "name": "item2", "value": 200},
            {"id": 3, "name": "item3", "value": 300},
        ],
        "metadata": {
            "version": "1.0",
            "created": "2025-01-01",
            "author": "test",
        },
    }


@pytest.fixture(scope="session")
def project_root() -> Path:
    """
    Provide the project root directory path.

    Returns:
        Path: Project root directory

    Example:
        def test_project_structure(project_root):
            assert (project_root / "README.md").exists()
            assert (project_root / "pyproject.toml").exists()
    """
    return Path(__file__).parent.parent


@pytest.fixture
def change_test_dir(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Change to the test's directory for the duration of the test.

    This is useful for tests that depend on relative file paths.

    Args:
        request: Pytest request fixture
        monkeypatch: Pytest monkeypatch fixture

    Example:
        def test_local_file(change_test_dir):
            # Now in the same directory as the test file
            with open("local_file.txt") as f:
                content = f.read()
    """
    monkeypatch.chdir(request.fspath.dirname)


# Pytest configuration hooks
def pytest_configure(config: pytest.Config) -> None:
    """
    Configure pytest with custom settings.

    Args:
        config: Pytest configuration object
    """
    # Add custom markers
    config.addinivalue_line(
        "markers", "unit: Unit tests that test individual components"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests for multiple components"
    )
    config.addinivalue_line(
        "markers", "slow: Tests that take significant time to execute"
    )
    config.addinivalue_line(
        "markers", "performance: Performance benchmarking tests"
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """
    Modify test collection to add markers based on test location.

    Args:
        config: Pytest configuration object
        items: List of collected test items
    """
    for item in items:
        # Add unit marker to tests in unit_test directories
        if "unit_test" in str(item.fspath) or "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)

        # Add integration marker to tests in integration directories
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)

        # Add slow marker to tests in end_to_end directories
        if "end_to_end" in str(item.fspath):
            item.add_marker(pytest.mark.slow)
