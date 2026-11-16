"""Storage-safety gates on ``BatchedExecutorBase``'s CUDA-graph path.

CPU coverage for the I-002 fix: a captured graph bakes raw storage
pointers for every tensor its kernels read (the ``eval_mod``
parameter tables above all), so ``_graph_replay`` must verify the
:func:`kit_torch.device.storage_fingerprint` snapshot taken at
capture before every replay — a rebind (``.to()``, ``_align_to``,
``_park_params``) triggers a recapture against the current storages,
and a failed recapture falls back to eager *permanently* with the
reason recorded.  Mirrors ``_GraphReplayModule``'s contract in
``kit_cuda.runners``; the real ``torch.cuda.CUDAGraph`` record path
stays CUDA-only (``requires_cuda``), so these tests arm the graph
slot with fakes exactly like the concrete-executor suites do.
"""

from __future__ import annotations

import warnings

import pytest
import torch

from kit_torch.executors import BatchedExecutorBase
from kit_torch.executors.base import (
    _empty_capture_warning,
    _reemit_capture_warnings,
)


class _FakeEval:
    """Stand-in for ``IRModule`` — the tensor tables fingerprinted.

    Not an ``nn.Module`` on purpose: ``storage_fingerprint`` reads
    ``parameters()``/``buffers()``/``_param_map``/``_param_values``
    by ``getattr``, so a plain object exercises the dict-table half
    (the ``_param_values`` entries that never registered).
    """

    def __init__(self) -> None:
        self._root = None
        self._param_map = {"w": torch.nn.Parameter(torch.randn(2, 2))}
        # Unregistered tensor — only reachable via the _param_values
        # table; this is the coverage the top-level fingerprint misses.
        self._param_values = {"v": torch.randn(3)}


class _DummyExecutor(BatchedExecutorBase, torch.nn.Module):
    """Minimal concrete executor: plan set, trivial forward."""

    def __init__(self) -> None:
        super().__init__()
        self._inputs = []
        self.eval_mod = _FakeEval()
        self._plan = {}
        self._const_cache = {}
        self._init_graph_state()
        self.eager_calls = 0

    def _forward_impl(self, *xs: torch.Tensor) -> torch.Tensor:
        self.eager_calls += 1
        return xs[0] * 2 if xs else torch.zeros(1)

    def forward(self, *xs: torch.Tensor) -> torch.Tensor:
        out = self._graph_replay(xs)
        if out is not None:
            return out
        return self._forward_impl(*xs)


class _FakeGraph:
    def __init__(self) -> None:
        self.replays = 0

    def replay(self) -> None:
        self.replays += 1


def _arm(mod: _DummyExecutor, x: torch.Tensor, out: torch.Tensor):
    """Populate the graph slot as ``capture_cuda_graph`` would."""
    g = _FakeGraph()
    mod._graph = g
    mod._graph_inputs = [x.detach().clone()]
    mod._graph_out = out
    mod._param_snap = mod._storage_snapshot()
    mod._graph_warmup = 3
    return g


@pytest.fixture
def mod() -> _DummyExecutor:
    return _DummyExecutor()


def _x() -> torch.Tensor:
    return torch.randn(4)


# ---------------------------------------------------------------------------
#  Replay gate — stable storages
# ---------------------------------------------------------------------------


def test_stable_params_replay(mod: _DummyExecutor):
    """(a) An unchanged fingerprint replays the captured graph."""
    x = _x()
    sentinel = torch.full((1,), 3.0)
    g = _arm(mod, x, sentinel)

    assert mod._graph_replay((x.clone(),)) is sentinel
    assert g.replays == 1
    assert mod.eager_calls == 0
    # in-place mutation keeps the same storage — replayed, per the
    # documented "in-place updates are replayed" caveat.
    with torch.no_grad():
        mod.eval_mod._param_values["v"].add_(1.0)
    assert mod._params_stable()
    assert mod._graph_replay((x.clone(),)) is sentinel
    assert g.replays == 2


def test_signature_mismatch_runs_eager(mod: _DummyExecutor):
    """Wrong arity/shape falls back to ``_forward_impl`` untouched."""
    x = _x()
    sentinel = torch.full((1,), 3.0)
    g = _arm(mod, x, sentinel)

    assert mod._graph_replay((torch.randn(7),)) is None
    assert mod._graph_replay(()) is None
    assert g.replays == 0
    assert mod.eager_calls == 0  # _graph_replay itself never evals
    out = mod(torch.randn(7))
    assert out is not sentinel
    assert mod.eager_calls == 1


def test_unfingerprinted_slot_still_replays(mod: _DummyExecutor):
    """``_param_snap None`` (hand-injected ``_graph``, no capture)
    keeps the historical replay behaviour — documented in
    ``_params_stable``."""
    x = _x()
    sentinel = torch.full((1,), 3.0)
    g = _FakeGraph()
    mod._graph = g
    mod._graph_inputs = [x.clone()]
    mod._graph_out = sentinel
    assert mod._param_snap is None
    assert mod._params_stable()
    assert mod._graph_replay((x.clone(),)) is sentinel
    assert g.replays == 1


# ---------------------------------------------------------------------------
#  Fingerprint coverage
# ---------------------------------------------------------------------------


def test_snapshot_covers_param_map_param_values_and_const_cache(
    mod: _DummyExecutor,
):
    """Each tensor family the graph reads moves the fingerprint."""
    snap = mod._storage_snapshot()
    # registered Parameter rebind
    mod.eval_mod._param_map["w"] = torch.nn.Parameter(torch.randn(2, 2))
    assert mod._storage_snapshot() != snap
    snap = mod._storage_snapshot()
    # unregistered _param_values tensor rebind (the eval_mod half of
    # the snapshot — invisible to storage_fingerprint(self) alone)
    mod.eval_mod._param_values["v"] = torch.randn(3)
    assert mod._storage_snapshot() != snap
    snap = mod._storage_snapshot()
    # _const_cache entry (index/bottom-row tensors baked at record)
    mod._const_cache[("idx",)] = torch.arange(4)
    assert mod._storage_snapshot() != snap


# ---------------------------------------------------------------------------
#  Rebind → recapture / eager-forever
# ---------------------------------------------------------------------------


def test_rebound_param_recapture_unavailable_eager_forever(
    mod: _DummyExecutor,
):
    """(b-fail) Rebound params + no CUDA → recapture declines →
    permanent eager with the reason recorded, never a stale replay."""
    x = _x()
    sentinel = torch.full((1,), 3.0)
    g = _arm(mod, x, sentinel)
    mod.eval_mod._param_values["v"] = torch.randn(3)  # rebind

    assert not mod._params_stable()
    out = mod(x.clone())
    # eager result, not the static buffer; old graph never replayed
    assert out is not sentinel
    assert g.replays == 0
    assert mod.eager_calls == 1
    # honest state: graph dropped, eager-forever recorded
    assert not mod.is_graph_captured
    assert mod._graph_eager_forever
    # the reason is recorded verbatim — "recapture declined" when
    # capture_cuda_graph no-ops (no CUDA), the exception text when it
    # raises (e.g. CUDA present but inputs CPU)
    assert mod._recapture_error
    assert (
        "recapture" in mod._recapture_error
        or ":" in mod._recapture_error
    )

    # subsequent calls stay eager — no repeated recapture attempts
    mod(x.clone())
    assert mod.eager_calls == 2
    assert g.replays == 0


def test_rebound_param_recapture_success_swaps_graph(
    mod: _DummyExecutor, monkeypatch
):
    """(b-ok) A successful recapture swaps graph AND static buffers —
    the replay must use the NEW objects, not the pre-check locals."""
    x = _x()
    old_out = torch.full((1,), 1.0)
    old_g = _arm(mod, x, old_out)
    mod.eval_mod._param_map["w"] = torch.nn.Parameter(torch.randn(2, 2))

    new_g = _FakeGraph()
    new_out = torch.full((1,), 2.0)
    seen: dict = {}

    def fake_capture(*args, warmup=3, **_kw):
        seen["warmup"] = warmup
        mod._graph = new_g
        mod._graph_inputs = [a.detach().clone() for a in args]
        mod._graph_out = new_out
        mod._param_snap = mod._storage_snapshot()
        mod._graph_eager_forever = False
        mod._recapture_error = None
        mod._recapture_count = 0
        mod._graph_warmup = warmup
        return mod

    monkeypatch.setattr(mod, "capture_cuda_graph", fake_capture)
    # non-default warmup proves _recapture forwards the captured value
    mod._graph_warmup = 7

    assert mod._graph_replay((x.clone(),)) is new_out
    assert seen["warmup"] == 7
    assert new_g.replays == 1
    assert old_g.replays == 0  # stale graph never replayed
    assert mod._recapture_count == 1
    assert mod._recapture_error is None
    assert mod.is_graph_captured


def test_recapture_declined_when_cuda_unavailable(
    mod: _DummyExecutor, monkeypatch
):
    """``capture_cuda_graph`` no-oping (no CUDA) is a declined
    recapture — same eager-forever outcome as a raise, with the
    distinct recorded reason."""
    x = _x()
    sentinel = torch.full((1,), 3.0)
    _arm(mod, x, sentinel)
    mod.eval_mod._param_values["v"] = torch.randn(3)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    assert mod._graph_replay((x.clone(),)) is None
    assert mod._graph_eager_forever
    assert "recapture declined" in (mod._recapture_error or "")


def test_recapture_exception_marks_eager_forever(
    mod: _DummyExecutor, monkeypatch
):
    """A raising recapture records the classified reason and never
    retries — the stale graph is dropped first."""
    x = _x()
    sentinel = torch.full((1,), 3.0)
    g = _arm(mod, x, sentinel)
    mod.eval_mod._param_values["v"] = torch.randn(3)

    def boom(*_a, **_k):
        raise RuntimeError("cuda graph capture blew up")

    monkeypatch.setattr(mod, "capture_cuda_graph", boom)
    assert mod._graph_replay((x.clone(),)) is None
    assert mod._graph_eager_forever
    assert "RuntimeError" in (mod._recapture_error or "")
    assert "blew up" in (mod._recapture_error or "")
    assert g.replays == 0
    # eager-forever short-circuits before any further capture attempt
    assert mod._graph_replay((x.clone(),)) is None


# ---------------------------------------------------------------------------
#  Empty-capture decline — zero recorded kernels is a capture FAILURE
# ---------------------------------------------------------------------------
#
# torch warns "The CUDA Graph is empty ..." (CUDAGraph.cpp) when a
# capture records no kernels; arming that graph would replay stale
# memory while ``is_graph_captured`` reads True.  ``capture_cuda_graph``
# raises on the warning — the same failure path every other capture
# error takes — so ``CudaGraphRunner`` records ``cuda_graph_failure``
# and delivery continues eager.  The CUDA record path itself stays
# ``requires_cuda``; these tests pin the detection helper directly.


def _warnmsg(text: str) -> warnings.WarningMessage:
    return warnings.WarningMessage(
        message=text, category=UserWarning, filename="f.py", lineno=1
    )


def test_empty_capture_warning_detects_torch_message():
    """Torch's verbatim warning (aten CUDAGraph.cpp) is detected."""
    ws = [
        _warnmsg(
            "The CUDA Graph is empty. This usually means that the "
            "graph was attempted to be captured on wrong device or "
            "stream."
        )
    ]
    assert _empty_capture_warning(ws) is ws[0]


def test_empty_capture_warning_variants_case_insensitive():
    """The older "CUDA Graph is empty" phrasing and case variants
    match; unrelated warnings do not."""
    assert (
        _empty_capture_warning(
            [_warnmsg("UserWarning: CUDA GRAPH IS EMPTY")]
        )
        is not None
    )
    assert _empty_capture_warning([_warnmsg("an empty graph ahead")])
    assert (
        _empty_capture_warning([_warnmsg("some other warning")]) is None
    )
    assert _empty_capture_warning([]) is None


def test_reemit_capture_warnings_skips_only_the_trigger():
    """catch_warnings(record=True) must not swallow unrelated
    warnings: everything except the matched empty-capture warning is
    re-emitted verbatim."""
    empty = _warnmsg("The CUDA Graph is empty ...")
    other = _warnmsg("unrelated capture warning")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _reemit_capture_warnings([empty, other], empty)
    assert len(caught) == 1
    # warn_explicit wraps the str in the category — compare text
    assert str(caught[0].message) == other.message
    # skipping None re-emits everything
    with warnings.catch_warnings(record=True) as caught2:
        warnings.simplefilter("always")
        _reemit_capture_warnings([empty, other], None)
    assert len(caught2) == 2
