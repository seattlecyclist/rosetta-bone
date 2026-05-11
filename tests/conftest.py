import os

import pytest


def pytest_collection_modifyitems(config, items):
    if config.getoption("-m") == "slow":
        return
    skip_slow = pytest.mark.skip(reason="needs -m slow")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


@pytest.fixture
def require_anthropic_key():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set")
