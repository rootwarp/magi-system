# Project Overview

This is a Python project.

## Development Setup

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate  # On Windows

# Install dependencies
pip install -r requirements.txt
```

## Project Structure

```
.
├── src/           # Source code
├── tests/         # Test files
├── requirements.txt
└── README.md
```

## Code Style

- Follow PEP 8 conventions
- Use type hints for function signatures
- Write docstrings for public functions and classes

## Testing

```bash
pytest tests/
```

## Common Commands

- Run tests: `pytest`
- Format code: `black .`
- Lint code: `ruff check .`
