"""
Skip the tests that need the private dataset, instead of erroring on it.

The Softwood deal records are confidential and are not distributed with this
repository (see ../data/README.md). Most of `test_pipeline.py` reads
`data/processed/deals_clean.csv`, so on a fresh clone those tests would fail at
fixture setup with a FileNotFoundError — seventeen red lines that say nothing
about the code.

This file turns them into skips with a reason attached. Nothing in
`test_pipeline.py` is modified, and when the dataset is present this file has no
effect at all: every test collects and runs exactly as it did before.

    with data:     19 passed
    without data:   2 passed, 17 skipped

The two that still run are the pure-logic ones — spreadsheet placeholder
parsing, and the product-line scope rule.
"""

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

CLEAN = ROOT / "data" / "processed" / "deals_clean.csv"
RAW = ROOT / "data" / "raw" / "textile_sales.csv"

# Fixtures that cannot be built without the cleaned dataset. `client` boots the
# FastAPI app, which fits the benchmark from the same CSV in its lifespan
# handler, so it depends on the file just as directly as `deals` does.
DATA_BACKED_FIXTURES = {"deals", "client"}

# One test reads the raw export directly rather than through a fixture.
NEEDS_RAW = {"test_cleaning_never_silently_loses_rows"}

_REASON = (
    "needs the confidential Softwood dataset at {path}, which is not "
    "distributed with this repository — see data/README.md"
)


def pytest_collection_modifyitems(config, items):
    missing_clean = not CLEAN.exists()
    missing_raw = not RAW.exists()
    if not (missing_clean or missing_raw):
        return

    for item in items:
        wants_clean = missing_clean and (
            DATA_BACKED_FIXTURES & set(getattr(item, "fixturenames", ()))
        )
        wants_raw = missing_raw and item.name in NEEDS_RAW
        if wants_clean:
            item.add_marker(pytest.mark.skip(
                reason=_REASON.format(path="data/processed/deals_clean.csv")))
        elif wants_raw:
            item.add_marker(pytest.mark.skip(
                reason=_REASON.format(path="data/raw/textile_sales.csv")))
