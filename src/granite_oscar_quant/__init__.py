"""Public package API for the IBM Granite OScaR adapter.

The package root exposes the lightweight Pydantic models immediately and lazily
loads the torch-heavy patching functions only when callers ask for them. That
keeps configuration and schema imports cheap in tools, tests, and notebooks that
do not need to instantiate a model.
"""

from .config import OscarKVConfig
from .schemas import BenchmarkReport, BenchmarkRun

__all__ = [
    "BenchmarkReport",
    "BenchmarkRun",
    "OscarKVConfig",
    "apply_oscar_to_granite",
    "restore_granite_attention",
]


def __getattr__(name: str):
    """Lazily import torch-backed patch helpers.

    What it does:
        Resolves `apply_oscar_to_granite` and `restore_granite_attention` on
        demand from `granite_patch`.

    Why it exists:
        Importing the patch module imports torch and later touches Hugging Face
        model symbols. Delaying that work keeps simple config/schema imports
        fast and avoids surprising import failures in environments that are only
        inspecting metadata.

    How it helps:
        Downstream code can use `from granite_oscar_quant import OscarKVConfig`
        without paying the model-runtime import cost, while still getting a
        convenient package-root import for the patching entry points.
    """
    if name in {"apply_oscar_to_granite", "restore_granite_attention"}:
        from .granite_patch import apply_oscar_to_granite, restore_granite_attention

        return {
            "apply_oscar_to_granite": apply_oscar_to_granite,
            "restore_granite_attention": restore_granite_attention,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
