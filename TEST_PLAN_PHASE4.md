# Phase 4: CI/CD Setup

## Overview
Phase 4 implements automated testing, linting, and deployment pipelines using GitHub Actions and pre-commit hooks.

## Completed Components

### ✅ GitHub Actions Workflows

#### 1. test.yml (`.github/workflows/test.yml`)
- **Trigger**: Push to `main` or `develop`, Pull Requests to `main` or `develop`
- **Matrix Testing**: Tests on Python 3.9, 3.10, 3.11
- **Steps**:
  1. Install dependencies from requirements.txt and requirements-test.txt
  2. Lint with flake8 (PEP 8 compliance)
  3. Run pytest with coverage reporting
  4. Upload coverage to Codecov
  5. Archive HTML coverage reports as artifacts
- **Ensures**: Code quality and test coverage on every push

#### 2. nightly-tests.yml (`.github/workflows/nightly-tests.yml`)
- **Trigger**: Daily at 2 AM UTC (can be triggered manually)
- **Steps**:
  1. Full test suite with verbose output
  2. Compatibility tests with different numpy/pandas versions
  3. Slack notification on failure
- **Ensures**: Compatibility across dependency versions, catches regressions

#### 3. deploy.yml (`.github/workflows/deploy.yml`)
- **Trigger**: Push to `main` with changes in critical files
- **Steps**:
  1. Run full test suite
  2. Build Docker image
  3. Push to registry (placeholder)
  4. Deployment notification
- **Ensures**: Only tested code is deployed, Docker-ready

### ✅ Pre-commit Hooks (`.pre-commit-config.yaml`)

#### Git Hygiene Hooks
- `trailing-whitespace`: Remove trailing spaces
- `end-of-file-fixer`: Ensure single newline at end of files
- `check-yaml`: Validate YAML syntax
- `check-added-large-files`: Prevent large files (>1MB)
- `detect-private-key`: Catch hardcoded secrets

#### Python Code Quality
- **black**: Auto-format code (Python 3.11 compatible)
- **isort**: Sort imports consistently with black profile
- **flake8**: Lint against PEP 8 (max line 127, ignores E203/W503)
- **mypy**: Type checking with library stubs
- **ruff**: Fast linting with auto-fix

#### CI Configuration
- Auto-commit fixes from hooks
- Auto-update PR with fixes
- Weekly hook updates
- Auto-generation of commit messages

### ✅ Docker Configuration (`Dockerfile`)
- Base: `python:3.11-slim` (minimal footprint)
- Installs dependencies from requirements.txt
- Exposes port 8501 for Streamlit
- Healthcheck endpoint configured
- Entry: Runs `streamlit run app.py`

### ✅ Dependency Management (`.github/dependabot.yml`)
- **pip**: Weekly updates, Monday 3 AM UTC
- **GitHub Actions**: Weekly updates, Monday 3 AM UTC
- PR limit: 5 concurrent dependency PRs
- Labels: `dependencies` and `ci` for easy filtering

## How to Use

### Initial Setup
```bash
# Install pre-commit hooks
pip install pre-commit
pre-commit install

# Run hooks manually (before CI runs them)
pre-commit run --all-files
```

### Local Development
```bash
# Make changes to Python files
git add .

# Pre-commit hooks will run automatically
# If they fail, fix the issues and commit again
git commit -m "feat: add new feature"
```

### CI/CD Flow
1. **Push to GitHub**: Triggers `test.yml`
2. **Tests Pass**: Workflow completes with coverage report
3. **Nightly Run**: 2 AM UTC run full suite + version matrix
4. **Push to main**: Triggers `deploy.yml` if critical files changed
5. **Dependabot**: Opens PRs for dependency updates weekly

## Verification Checklist

- [x] `.github/workflows/test.yml` created and contains pytest + coverage
- [x] `.github/workflows/nightly-tests.yml` created with version matrix + Slack
- [x] `.github/workflows/deploy.yml` created with Docker build
- [x] `.pre-commit-config.yaml` created with black, isort, flake8, mypy, ruff
- [x] `Dockerfile` created with Streamlit configuration
- [x] `.github/dependabot.yml` created for dependency updates
- [x] All workflows reference correct file paths
- [x] Python versions match project support (3.9, 3.10, 3.11)

## Next Steps (Phase 5)

Phase 5 will add:
1. **Test Dashboard Tab**: Display test results, coverage trends in Streamlit
2. **Runtime Validation**: Add health checks to app.py
3. **Alert System**: Slack/email notifications for failures
4. **Monitoring**: Track model accuracy, API response times

## Notes

- Pre-commit hooks run **before** CI, catching issues locally
- Nightly tests catch version compatibility issues that CI may miss
- Deploy workflow is Docker-ready but requires actual registry (ECR, Docker Hub, etc.)
- Dependabot PRs are auto-created; merge at your discretion
- All workflows have failure notifications for visibility
