from __future__ import annotations

import contextvars
import importlib.machinery
import re
import sys
import types


def apply_runtime_compatibility_patches() -> None:
    """Patch optional integrations that are broken in this container image.

    The installed `kernels` package raises during import with the current
    `huggingface_hub` dataclass validator, and the installed `trackio` package
    imports a hub symbol that is not present. Both are optional for these
    experiments, but TRL/Transformers probe them at import time.
    """

    _patch_transformers_hub_kernels()
    _patch_trackio()


def _patch_transformers_hub_kernels() -> None:
    module_name = "transformers.integrations.hub_kernels"
    if module_name in sys.modules:
        return

    try:
        from kernels import (  # noqa: F401
            Device,
            LayerRepository,
            Mode,
            get_kernel,
            register_kernel_mapping,
            replace_kernel_forward_from_hub,
            use_kernel_forward_from_hub,
        )

        return
    except Exception:
        pass

    module = types.ModuleType(module_name)

    def use_kernel_forward_from_hub(*args, **kwargs):
        def decorator(cls):
            return cls

        return decorator

    class LayerRepository:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("The optional `kernels` package is unavailable in this runtime.")

    def replace_kernel_forward_from_hub(*args, **kwargs):
        raise RuntimeError("The optional `kernels` package is unavailable in this runtime.")

    def register_kernel_mapping(*args, **kwargs):
        raise RuntimeError("The optional `kernels` package is unavailable in this runtime.")

    def is_kernel(attn_implementation):
        return (
            attn_implementation is not None
            and re.search(r"^[^/:]+/[^/:]+(?:@[^/:]+)?(?::[^/:]+)?$", attn_implementation) is not None
        )

    def load_and_register_kernel(attn_implementation: str) -> None:
        if is_kernel(attn_implementation):
            raise ImportError("The optional `kernels` package is unavailable in this runtime.")

    module.use_kernel_forward_from_hub = use_kernel_forward_from_hub
    module.LayerRepository = LayerRepository
    module.replace_kernel_forward_from_hub = replace_kernel_forward_from_hub
    module.register_kernel_mapping = register_kernel_mapping
    module.is_kernel = is_kernel
    module.load_and_register_kernel = load_and_register_kernel
    module._kernels_available = False
    sys.modules[module_name] = module

    try:
        import transformers.utils.import_utils as import_utils

        import_utils._kernels_available = False
    except Exception:
        pass


def _patch_trackio() -> None:
    if "trackio" in sys.modules:
        return

    try:
        import trackio  # noqa: F401

        return
    except Exception:
        pass

    module = types.ModuleType("trackio")
    module.__spec__ = importlib.machinery.ModuleSpec("trackio", loader=None)
    module.context_vars = types.SimpleNamespace(current_run=contextvars.ContextVar("current_run", default=None))
    sys.modules["trackio"] = module
