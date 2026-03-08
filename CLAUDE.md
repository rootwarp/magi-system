# Project Overview

This is a multi-language project containing Python and Go code.

## Project Structure

```
.
├── python/            # Python codebase
│   ├── magi_system/   # Main Python package
│   ├── tests/         # Python test files
│   ├── requirements.txt
│   └── pyproject.toml
├── golang/            # Go codebase
│   ├── cmd/           # Command entrypoints
│   ├── internal/      # Private application code
│   ├── pkg/           # Public library code
│   └── go.mod
├── CLAUDE.md
├── LICENSE
└── README.md
```

## Python

### Development Setup

```bash
cd python

# Create virtual environment
python -m venv venv

# Activate virtual environment
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate  # On Windows

# Install dependencies
pip install -r requirements.txt
```

### Code Style

- Follow PEP 8 conventions
- Use type hints for function signatures
- Write docstrings for public functions and classes

### Testing

```bash
cd python
pytest tests/
```

### Common Commands

- Run tests: `cd python && pytest`
- Format code: `cd python && black .`
- Lint code: `cd python && ruff check .`

## Go

### Development Setup

```bash
cd golang
go mod tidy
```

### Code Style

- Follow standard Go conventions
- Use `gofmt` / `goimports` for formatting
- Write GoDoc comments for exported functions and types

### Testing

```bash
cd golang
go test ./...
```

### Common Commands

- Run tests: `cd golang && go test ./...`
- Build: `cd golang && go build ./...`
- Lint code: `cd golang && golangci-lint run`
