"""Check test configuration with released and newer optional tooling."""

import importlib.util
import runpy
from pathlib import Path
from unittest.mock import patch

import pytest
from hypothesis import settings


@pytest.mark.parametrize("has_budget", [False, True])
@pytest.mark.parametrize(
    "profile, examples", [(None, 25), ("quick", 25), ("nightly", 200)]
)
def test_conftest_tooling(monkeypatch, has_budget, profile, examples):
    if profile is None:
        monkeypatch.delenv("HYPOTHESIS_PROFILE", raising=False)
    else:
        monkeypatch.setenv("HYPOTHESIS_PROFILE", profile)
    monkeypatch.setenv("JAXEDITH_TYPECHECK", "0")
    previous_profile = settings.get_current_profile_name()
    original_find_spec = importlib.util.find_spec

    def find_spec(name, *args, **kwargs):
        if name == "hwoutils.testbudget":
            return object() if has_budget else None
        return original_find_spec(name, *args, **kwargs)

    try:
        with patch("importlib.util.find_spec", side_effect=find_spec):
            config = runpy.run_path(str(Path(__file__).with_name("conftest.py")))
        assert config["pytest_plugins"] == (
            ["hwoutils.testbudget.plugin"] if has_budget else []
        )
        assert settings.get_current_profile_name() == (profile or "quick")
        assert settings().max_examples == examples
        assert settings().deadline is None
    finally:
        settings.load_profile(previous_profile)
