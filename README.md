# AMAT5315 2026 Fall Exercise

This repository contains my weekly exercises and projects for the AMAT5315 computation course.

Exercises are organized by week:

- `week1/`
- `week2/`
- and so on

## Install pytest

Install `pytest` with:

```bash
python3 -m pip install pytest
```

If you are using Homebrew Python on macOS and see an `externally-managed-environment` error, install it for your user environment with:

```bash
python3 -m pip install --user --break-system-packages pytest
```

## Run Tests

Run all tests for the weekly exercise folders with:

```bash
python3 -m pytest week1/
```

For a later week, replace `week1/` with the relevant folder, for example:

```bash
python3 -m pytest week2/
```
