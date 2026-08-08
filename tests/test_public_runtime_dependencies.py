"""Regression tests for public runtime dependency boundaries."""

import builtins
import importlib
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_object_constants_import_does_not_require_prior(monkeypatch):
    """Static object constants must not pull in Prior or its remote datasets."""
    module_name = "sims.utils.constants.object_constants"
    sys.modules.pop(module_name, None)

    original_import = builtins.__import__

    def import_without_prior(name, *args, **kwargs):
        if name == "prior" or name.startswith("prior."):
            raise AssertionError("object_constants unexpectedly imported prior")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_prior)
    module = importlib.import_module(module_name)

    assert not hasattr(module, "bad_asset_ids")


def test_unused_wandb_dependency_is_not_declared():
    """The paper data pipeline must not restore the unused W&B runtime stack."""
    project_config = (REPOSITORY_ROOT / "pyproject.toml").read_text()

    assert '"wandb' not in project_config
