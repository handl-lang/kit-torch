"""kit-torch — the PyTorch integration adapters + backend.

``torch.export`` → IR (:mod:`~kit_torch.torch_bridge`), the
executable ``IRModule`` + weight-chain folding, the batched executor
machinery (:mod:`~kit_torch.executors`), the small model zoo
(:mod:`~kit_torch.models`), and the typed reports + equivalence
gate (:mod:`~kit_torch.report`).

Plan 0007 adds the backend ports — :func:`~kit_torch.backend.TorchBackend`
(the :class:`~mlopt_core.pipeline.Backend` value:
:class:`TorchSource` + :class:`TorchSink` +
:class:`~kit_torch.composer.TorchComposer` +
:class:`~kit_torch.meter.TorchMeter`), the torch delivery runner
(:class:`~kit_torch.runners.TorchCompileRunner`), the torch halves
of the regime machinery (:mod:`~kit_torch.regime`) and calibration
(:mod:`~kit_torch.calibrate`), the production export helpers
(:mod:`~kit_torch.export`), and the torch autotune candidate
builders (:mod:`~kit_torch.autotune`).  The public names below
resolve lazily — importing ``kit_torch`` pulls no pipeline
machinery until the names are touched.
"""

import importlib
from typing import Any

# Importing the concrete-eval backend registers it with mlopt-core
# (:func:`mlopt_core.meta.register_concrete_eval`): any import of the
# torch adapter makes core's numeric candidate check work, and core
# itself imports neither ``torch`` nor ``kit_torch``.
from kit_torch import meta_eval as _meta_eval  # noqa: F401

# ---------------------------------------------------------------------------
# Lazy public surface — names resolve on first attribute access.
# ---------------------------------------------------------------------------

_DELEGATED = {
    "TorchBackend": "kit_torch.backend",
    "TorchComposer": "kit_torch.composer",
    "TorchMeter": "kit_torch.meter",
    "TorchCompileRunner": "kit_torch.runners",
    "TORCH_BUILDERS": "kit_torch.autotune",
    "RegimeDispatch": "kit_torch.regime",
    "regime_dispatch": "kit_torch.regime",
    "calibrate": "kit_torch.calibrate",
    "TargetProfile": "kit_torch.calibrate",
    "load_profile": "kit_torch.calibrate",
    "ContractionPolicy": "kit_torch.contraction_policy",
    "load_contraction_policy": "kit_torch.contraction_policy",
    "DevicePolicy": "kit_torch.device",
    "DeviceReport": "kit_torch.device",
    "DevicePlacementError": "kit_torch.device",
    "UnsupportedDistributedError": "kit_torch.device",
    "device_graph_key": "kit_torch.device",
    "cuda_device_indices": "kit_torch.device",
    "distributed_wrapper_name": "kit_torch.device",
    "is_collective_op": "kit_torch.device",
    "ShardedAwareDevicePolicy": "kit_torch.device_placement",
    "ShardedParamError": "kit_torch.device_placement",
    "check_sharded_params": "kit_torch.device_placement",
    "sharded_params_in": "kit_torch.device_placement",
    "is_sharded_param": "kit_torch.device_placement",
    "is_dtensor": "kit_torch.device_placement",
    "DeviceSlot": "kit_torch.device_placement",
    "PlacementPlan": "kit_torch.device_placement",
    "plan_placement": "kit_torch.device_placement",
    "group_by_device": "kit_torch.device_placement",
    "DeviceCheckedSource": "kit_torch.device_placement",
    "DevicePolicyRunner": "kit_torch.device_placement",
    "RegionFuseConfig": "kit_torch.region_fuse",
    "RegionFusionRunner": "kit_torch.runners",
}


def __getattr__(name: str) -> Any:
    """Resolve the adapter-surface names lazily."""
    mod = _DELEGATED.get(name)
    if mod is not None:
        return getattr(importlib.import_module(mod), name)
    raise AttributeError(
        f"module {__name__!r} has no attribute {name!r}"
    )


def register() -> None:
    """Seat the torch backend on ``mlopt_core.backends.DEFAULT_REGISTRY``.

    Explicit seam — importing ``kit_torch`` registers nothing; call
    ``kit_torch.register()`` at the entry point (the same pattern as
    ``mlopt_kernel.play.register()`` / ``kit_onnx.register()``).
    Registers ``"torch"`` with :class:`TorchSource` + ``TorchSink``
    (``"torch"`` target), ``TorchComposer`` and ``TorchMeter`` as lazy
    factories — ``build_backend`` assembles the full port set.  A
    custom ``device_policy`` is not registered (it is per-deployment);
    pass one via ``TorchBackend(device_policy=...)`` directly.
    """
    from mlopt_core import backends

    from kit_torch.adapters import TorchSink, TorchSource
    from kit_torch.composer import TorchComposer
    from kit_torch.meter import TorchMeter

    backends.register_backend(
        "torch",
        source=TorchSource,
        targets={"torch": TorchSink},
        composer=TorchComposer,
        meter=TorchMeter,
        notes="PyTorch backend; device_policy per-deployment",
    )
