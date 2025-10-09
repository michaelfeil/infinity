# Testing Infrastructure

This directory contains the root-level testing infrastructure for the Infinity monorepo.

## Overview

The testing infrastructure provides a comprehensive setup for running tests across the entire Infinity project, including all sub-packages in the `libs/` directory.

## Structure

```
tests/
├── __init__.py              # Package initialization
├── conftest.py              # Shared pytest fixtures and configuration
├── README.md                # This file
├── unit/                    # Unit tests
│   ├── __init__.py
│   └── test_infrastructure.py
└── integration/             # Integration tests
    ├── __init__.py
    └── test_example.py
```

## Running Tests

### Run All Tests

```bash
poetry run pytest
```

### Run Tests with Coverage

```bash
poetry run pytest --cov=libs --cov-report=term-missing --cov-report=html
```

### Run Only Unit Tests

```bash
poetry run pytest -m unit
```

### Run Only Integration Tests

```bash
poetry run pytest -m integration
```

### Exclude Slow Tests

```bash
poetry run pytest -m "not slow"
```

### Run Tests in a Specific Directory

```bash
poetry run pytest tests/unit/
poetry run pytest libs/infinity_emb/tests/
```

## Test Markers

The following markers are available for organizing tests:

- `@pytest.mark.unit` - Unit tests that test individual components in isolation
- `@pytest.mark.integration` - Integration tests that test multiple components together
- `@pytest.mark.slow` - Tests that take significant time to run
- `@pytest.mark.performance` - Performance benchmarking tests

Example usage:

```python
import pytest

@pytest.mark.unit
def test_something():
    assert True

@pytest.mark.integration
@pytest.mark.slow
def test_complex_integration():
    # This test is marked as both integration and slow
    pass
```

## Available Fixtures

The following fixtures are available in all tests via `conftest.py`:

### `temp_dir`
Provides a temporary directory that is automatically cleaned up after the test.

```python
def test_example(temp_dir):
    test_file = temp_dir / "test.txt"
    test_file.write_text("hello")
    assert test_file.read_text() == "hello"
```

### `temp_file`
Provides a temporary file that is automatically cleaned up after the test.

```python
def test_example(temp_file):
    temp_file.write_text("content")
    assert temp_file.read_text() == "content"
```

### `mock_env_vars`
Sets up mock environment variables for testing.

```python
def test_example(mock_env_vars):
    assert os.environ.get("TEST_MODE") == "true"
```

### `mock_config`
Provides a mock configuration dictionary.

```python
def test_example(mock_config):
    assert mock_config["debug"] is True
```

### `sample_data`
Provides sample data for testing.

```python
def test_example(sample_data):
    assert len(sample_data["items"]) == 3
```

### `project_root`
Provides the project root directory path.

```python
def test_example(project_root):
    assert (project_root / "README.md").exists()
```

## Coverage Reports

Coverage reports are automatically generated when running tests with coverage enabled:

- **Terminal Report**: Shows coverage summary with missing lines
- **HTML Report**: Generated in `htmlcov/` directory - open `htmlcov/index.html` in a browser
- **XML Report**: Generated as `coverage.xml` for CI/CD integration

The project is configured to require 80% code coverage by default. This can be adjusted in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
addopts = [
    "--cov-fail-under=80",  # Adjust this value
]
```

## Writing Tests

### Unit Test Example

```python
# tests/unit/test_my_module.py
import pytest

class TestMyModule:
    def test_basic_functionality(self):
        """Test basic functionality."""
        result = 2 + 2
        assert result == 4

    @pytest.mark.unit
    def test_with_fixture(self, mock_config):
        """Test using a fixture."""
        assert mock_config["timeout"] == 30
```

### Integration Test Example

```python
# tests/integration/test_my_integration.py
import pytest

@pytest.mark.integration
class TestMyIntegration:
    def test_components_together(self, temp_dir, project_root):
        """Test multiple components working together."""
        # Integration test logic here
        pass
```

## CI/CD Integration

The testing infrastructure is designed to work seamlessly with CI/CD pipelines:

```yaml
# Example GitHub Actions workflow
- name: Run Tests
  run: |
    poetry install
    poetry run pytest --cov=libs --cov-report=xml

- name: Upload Coverage
  uses: codecov/codecov-action@v3
  with:
    file: ./coverage.xml
```

## Configuration

All test configuration is centralized in the root `pyproject.toml` file:

- `[tool.pytest.ini_options]` - Pytest settings
- `[tool.coverage.run]` - Coverage collection settings
- `[tool.coverage.report]` - Coverage reporting settings

## Troubleshooting

### Tests Not Found

If pytest cannot find your tests, ensure:
1. Test files are named `test_*.py` or `*_test.py`
2. Test functions start with `test_`
3. Test classes start with `Test`

### Import Errors

If you encounter import errors:
1. Ensure Poetry environment is activated: `poetry shell`
2. Install dependencies: `poetry install`
3. Check that `PYTHONPATH` is set correctly

### Coverage Issues

If coverage reports are not generated:
1. Ensure pytest-cov is installed: `poetry install`
2. Check coverage configuration in `pyproject.toml`
3. Verify source paths are correct

## Additional Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [pytest-cov Documentation](https://pytest-cov.readthedocs.io/)
- [pytest-mock Documentation](https://pytest-mock.readthedocs.io/)
- [Coverage.py Documentation](https://coverage.readthedocs.io/)
