"""Shared machinery for the planned batched executors.

The three batched lowering adapters —
:class:`kit_carriers.scan_lower.BatchedScanModule`,
:class:`kit_carriers.om_lower.BatchedOMModule`, and
:class:`kit_carriers.omd_lower.BatchedOmdModule` — all embed a plain
``IRModule`` as submodule ``eval_mod``, analyse their root term into
``self._plan`` once in ``__init__``, and carry byte-identical
CUDA-graph capture state plus input-env/eval-closure plumbing.  This
mixin (extracted in plan 0002 phase A) single-sources that shared
machinery; each concrete class keeps its own ``__init__`` and its own
``_forward_impl`` plan section, which genuinely differ per carrier.

The module also hosts the plan-time **level-schedule skeleton**
(extracted in plan 0002 phase D) — :func:`level_schedule` (post-order
``visit()`` assigning tree levels) and :func:`slot_gathers`
(leaf-first slot numbering + per-level operand gathers) — shared by
``build_scan_plan``/``build_om_plan``/``build_omd_plan``.  The
per-domain seeding stays at the sites: om weights leaf slots by DAG
multiplicity and omd's map forest walks iteratively per domain, so
the helpers expose the invariant, not a policy.

Ports layer: the concrete classes conform to
:class:`mlopt_core.ports.BatchedExecutor` (``forward`` + ``_plan`` +
``is_batched``); the members here stay class-level API, not port
members.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Any, Self

import torch
from mlopt_core.term import Op

from kit_torch.device import storage_fingerprint

__all__ = ["BatchedExecutorBase", "level_schedule", "slot_gathers"]


def level_schedule(
    root: Any,
    children_of: Callable[[Any], Any],
    is_leaf: Callable[[Any], bool],
    *,
    key: Callable[[Any], Any] = lambda t: t,
) -> tuple[list, list[list]]:
    """Post-order level assignment over a compose tree (or DAG).

    ``visit(t)`` returns ``t``'s level — 0 when ``is_leaf(t)``, else
    ``max(children's levels) + 1`` — appending leaves to ``leaves``
    in first-encounter order and internal nodes to ``levels[lv-1]``,
    so every node's operands sit at strictly earlier levels.  Shared
    subtrees are visited once via the ``key``-projected memo: the
    default is the term itself (post-hash-consing terms are interned
    — content keys); ``id`` keeps strict per-object identity.

    Returns ``(leaves, levels)``; pair with :func:`slot_gathers` for
    the leaf-first slot numbering + per-level operand gathers.
    ``children_of``/``is_leaf`` carry the per-domain vocabulary
    (``aff_compose`` / ``om_compose`` / ``omd_compose``).
    """
    leaves: list = []
    levels: list[list] = []
    level_of: dict[Any, int] = {}

    def visit(t: Any) -> int:
        k = key(t)
        lv = level_of.get(k)
        if lv is not None:
            return lv
        if is_leaf(t):
            level_of[k] = 0
            leaves.append(t)
            return 0
        lv = max(visit(a) for a in children_of(t)) + 1
        level_of[k] = lv
        while len(levels) < lv:
            levels.append([])
        levels[lv - 1].append(t)
        return lv

    visit(root)
    return leaves, levels


def slot_gathers(
    levels: list[list],
    children_of: Callable[[Any], Any],
    slot: dict,
    next_slot: int,
    *,
    key: Callable[[Any], Any] = lambda t: t,
) -> list[tuple[list[int], list[int]]]:
    """Leaf-first slot numbering + per-level operand gathers.

    ``slot`` arrives seeded with the leaf slots — contiguous
    ``0..n-1`` at most sites, or om_lower's multiplicity-weighted
    layout (DAG-shared leaves occupy one slot per occurrence — pass
    the real ``next_slot``).  Each level's outputs then take
    consecutive slots in order, so every operand slot is < its own
    level's outputs.  Returns ``gathers[k] == (f_slots, g_slots)``,
    the index_select lists each level consumes.  ``slot`` is mutated
    (internal nodes added) so the caller reads the root's slot
    afterwards.
    """
    gathers: list[tuple[list[int], list[int]]] = []
    for nodes in levels:
        gathers.append(
            (
                [slot[key(children_of(t)[0])] for t in nodes],
                [slot[key(children_of(t)[1])] for t in nodes],
            )
        )
        for t in nodes:
            slot[key(t)] = next_slot
            next_slot += 1
    return gathers


def _tensor_storage(t: torch.Tensor) -> Any:
    """Untyped-storage identity for one tensor (fingerprint leaf)."""
    try:
        return t.untyped_storage()
    except Exception:  # pragma: no cover — exotic tensor types
        return t


#: Substrings of torch's empty-capture warning — "The CUDA Graph is
#: empty. This usually means that the graph was attempted to be
#: captured on wrong device or stream." (aten/src/ATen/cuda/
#: CUDAGraph.cpp), matched case-insensitively with the older
#: "CUDA Graph is empty" phrasing covered by the same substrings.
_EMPTY_GRAPH_WARNING = ("graph is empty", "empty graph")


def _empty_capture_warning(
    ws: list[warnings.WarningMessage],
) -> warnings.WarningMessage | None:
    """Return the empty-capture warning in *ws*, when present.

    An empty ``torch.cuda.graph`` records zero kernels — replaying it
    would serve stale memory while ``is_graph_captured`` still reads
    True, so capture treats the warning as a failure (the same raise
    path every other capture error takes) rather than arming a graph
    that does nothing.
    """
    for w in ws:
        text = str(w.message).lower()
        if any(p in text for p in _EMPTY_GRAPH_WARNING):
            return w
    return None


def _reemit_capture_warnings(
    ws: list[warnings.WarningMessage],
    skip: warnings.WarningMessage | None,
) -> None:
    """Re-emit every recorded warning except *skip*.

    ``warnings.catch_warnings(record=True)`` diverts warnings from
    stderr into the list; the empty-capture check must not swallow
    unrelated warnings raised inside the record, so non-triggering
    ones are re-emitted verbatim after the capture block exits.
    """
    for w in ws:
        if w is not skip:
            warnings.warn_explicit(
                w.message, w.category, w.filename, w.lineno
            )


class BatchedExecutorBase:
    """Mixin holding the batched executors' shared machinery.

    Use as ``class BatchedXModule(BatchedExecutorBase, nn.Module)`` —
    the mixin defines no ``__init__``, so ``super().__init__()`` in
    the concrete class still reaches ``nn.Module.__init__`` through
    the MRO, and ``nn.Module``'s attribute hooks apply normally.

    The concrete ``__init__`` must set ``_inputs`` (``ir.inputs``),
    ``eval_mod`` (a plain ``IRModule``), ``_plan`` (``None`` when the
    root doesn't match the planned pattern), and ``_const_cache``,
    then call :meth:`_init_graph_state`.  ``_forward_impl`` itself is
    supplied by the concrete class.
    """

    # Field contract (annotations only — the concrete __init__
    # initialises the values; nothing here touches them).
    _inputs: list[Any]
    _plan: dict | None
    eval_mod: Any
    _const_cache: dict[tuple, torch.Tensor]
    _graph: torch.cuda.CUDAGraph | None
    _graph_inputs: list[torch.Tensor]
    _graph_out: torch.Tensor | None
    _forward_impl: Callable[..., torch.Tensor]
    _param_snap: tuple | None
    _graph_warmup: int
    _graph_eager_forever: bool
    _recapture_count: int
    _recapture_error: str | None

    def _init_graph_state(self) -> None:
        """(Re)set the CUDA-graph capture fields.

        Concrete ``__init__``s call this once ``eval_mod``/``_plan``
        exist; :meth:`drop_cuda_graph` reuses it.  Also resets the
        storage-safety state — a dropped graph carries no fingerprint,
        and a fresh :meth:`capture_cuda_graph` starts with a clean
        eager-forever/recapture record.
        """
        self._graph = None
        self._graph_inputs = []
        self._graph_out = None
        self._param_snap = None
        self._graph_warmup = 3
        self._graph_eager_forever = False
        self._recapture_count = 0
        self._recapture_error = None

    @property
    def is_graph_captured(self) -> bool:
        """True after :meth:`capture_cuda_graph` succeeded."""
        return self._graph is not None

    @property
    def _root(self) -> Any:
        """The folded root term the executor lowers.

        Delegates to the wrapped ``IRModule`` so consumers
        inspecting the lowered term (reports, benches) see the same
        root whichever executor routing picked.
        """
        return self.eval_mod._root

    @property
    def _param_map(self) -> Any:
        """The materialised parameter map.

        Delegates to the wrapped ``IRModule`` (fused_* leaves live
        there whichever executor routing picked).
        """
        return self.eval_mod._param_map

    def capture_cuda_graph(  # pragma: no cover — CUDA-only body
        self,
        *example_inputs: torch.Tensor,
        warmup: int = 3,
    ) -> Self:
        """Capture the batched forward into a CUDA graph.

        After capture, ``forward`` copies each input into static
        buffers and replays the graph — one launch total.  Returns
        ``self`` for chaining; a no-op when the module is not batched
        or CUDA is unavailable.

        Caveats (standard CUDA-graph rules): inputs must be CUDA
        tensors; calls with a different shape/dtype/device fall back
        to the eager path; parameters are baked in by pointer —
        in-place updates are replayed, replacing a parameter is not
        (a rebind is caught by the storage fingerprint and triggers a
        recapture, or an honest eager-forever fallback — see
        :meth:`_graph_replay`); the returned tensor is the static
        output buffer, overwritten by the next call (clone it to keep
        it).
        """
        if self._plan is None or not torch.cuda.is_available():
            return self
        if not all(t.is_cuda for t in example_inputs):
            raise ValueError("capture_cuda_graph requires CUDA inputs")
        static_ins = [t.detach().clone() for t in example_inputs]
        # Warm up on a side stream (also populates _const_cache so no
        # host-side allocation happens during capture).
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(warmup):
                self._forward_impl(*static_ins)
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with (
            warnings.catch_warnings(record=True) as ws,
            torch.cuda.graph(g),
        ):
            out = self._forward_impl(*static_ins)
        empty = _empty_capture_warning(ws)
        _reemit_capture_warnings(ws, empty)
        if empty is not None:
            # Zero recorded kernels — the same failure path as any
            # other capture error: the raise propagates to
            # CudaGraphRunner (``cuda_graph_failure`` + drop) or to
            # ``_recapture`` (eager-forever), never an armed no-op
            # graph that would replay stale memory.
            raise RuntimeError(
                "CUDA-graph capture recorded no kernels: "
                f"{empty.message}"
            )
        # Storage snapshot taken AFTER warmup+record so it covers the
        # fully-populated _const_cache; assigned atomically with the
        # graph so a fingerprint failure can never leave a graph
        # replaying un-fingerprinted storages (the exception
        # propagates — CudaGraphRunner records it as
        # ``cuda_graph_failure`` and delivery continues eager).
        snap = self._storage_snapshot()
        self._graph, self._graph_inputs, self._graph_out = (
            g,
            static_ins,
            out,
        )
        self._param_snap = snap
        self._graph_warmup = warmup
        self._graph_eager_forever = False
        self._recapture_count = 0
        self._recapture_error = None
        return self

    def drop_cuda_graph(self) -> None:
        """Release the captured graph, returning to eager execution."""
        self._init_graph_state()

    # -- storage safety -----------------------------------------------

    def _storage_snapshot(self) -> tuple:
        """Untyped-storage identity over everything the graph reads.

        A captured graph holds raw pointers to the storages its
        kernels touched — the wrapped parameters above all — and does
        NOT keep them alive: ``IRModule._park_params`` (the
        low-memory ``verify_module`` dance) drops the live weight
        tables to CPU, freeing exactly those storages, and a sibling
        forward can then reuse the memory so replay reads garbage
        (the I-002 CUDA poison).  The snapshot mirrors
        ``_GraphReplayModule._param_snap`` in ``kit_cuda.runners``:

        * ``storage_fingerprint(self)`` — registered
          parameters/buffers (recursing into ``eval_mod``) plus this
          module's ``_param_map`` view (the property delegates to
          ``eval_mod._param_map``);
        * ``storage_fingerprint(self.eval_mod)`` — the embedded
          ``IRModule``'s ``_param_values`` table, which holds tensors
          that never registered as Parameters (``_align_to`` still
          migrates them);
        * ``_const_cache`` values — the index/bottom-row tensors
          ``_forward_impl`` bakes into the recording; a device/dtype
          realignment recreates them, and the graph must not keep
          reading the stale storage.

        Holding the untyped storages keeps the captured memory alive
        (no free-then-realloc ABA); comparing identities before every
        replay catches every real rebind — ``.to()``, ``_align_to``,
        ``load_state_dict``-adjacent moves — since none of them
        preserves storage identity.
        """
        out: list[Any] = []
        out.extend(storage_fingerprint(self))
        ev = getattr(self, "eval_mod", None)
        if ev is not None and ev is not self:
            out.extend(storage_fingerprint(ev))
        cache = getattr(self, "_const_cache", None)
        if isinstance(cache, dict):
            out.extend(
                _tensor_storage(t)
                for t in cache.values()
                if isinstance(t, torch.Tensor)
            )
        return tuple(out)

    def _params_stable(self) -> bool:
        """Return True while the captured storages ARE the live ones.

        ``_param_snap is None`` means the graph slot was populated
        without :meth:`capture_cuda_graph` (tests inject ``_graph``
        directly) — there is no fingerprint to check against, so the
        historical replay-the-slot behaviour stands.  Every graph the
        capture path itself records carries a snapshot.
        """
        snap = self._param_snap
        if snap is None:
            return True
        cur = self._storage_snapshot()
        return len(cur) == len(snap) and all(
            a is b for a, b in zip(cur, snap, strict=True)
        )

    def _recapture(self, xs: tuple[torch.Tensor, ...]) -> bool:
        """Re-record against the current storages; ``False`` on failure.

        Mirrors ``_GraphReplayModule._recapture`` /
        ``_DeviceGraphModule``'s drop-and-recapture: drop the stale
        graph FIRST (a failed capture must never leave the old
        pointers replayable), then run the same
        :meth:`capture_cuda_graph` recipe on the caller's inputs —
        which also re-snapshots the now-current storages.  A failed
        or declined recapture marks the module eager-forever (the
        reason recorded on ``_recapture_error``) rather than
        retrying — and never replaying — a stale graph.
        """
        count = self._recapture_count
        warmup = self._graph_warmup
        try:
            self.drop_cuda_graph()
            self.capture_cuda_graph(*xs, warmup=warmup)
        except Exception as exc:
            self._graph_eager_forever = True
            self._recapture_error = f"{type(exc).__name__}: {exc}"
            return False
        if self._graph is None:
            # capture_cuda_graph declined (plan gone / CUDA vanished)
            # — same honest eager-forever outcome as a raise.
            self._graph_eager_forever = True
            self._recapture_error = (
                "recapture declined: capture_cuda_graph did not "
                "record a graph"
            )
            return False
        self._recapture_count = count + 1
        self._recapture_error = None
        return True

    def _graph_replay_ready(self, xs: tuple[torch.Tensor, ...]) -> bool:
        """Return True when the captured graph may safely replay ``xs``.

        The per-call storage check: eager-forever short-circuits,
        a stable fingerprint replays, and a mismatch attempts ONE
        recapture against the current storages — ``False`` means the
        caller must run ``_forward_impl`` instead (never a stale
        replay).
        """
        if self._graph_eager_forever:
            return False
        if self._params_stable():
            return True
        return self._recapture(xs)

    def _graph_replay(self, xs: tuple[torch.Tensor, ...]) -> Any:
        """Replay the captured graph for ``xs``; ``None`` → run eager.

        Single-sources the forward fast-path the concrete executors
        used to inline: signature match (arity + shape/dtype/device
        against the static input buffers), the storage-safety gate
        (:meth:`_graph_replay_ready` — which may recapture, swapping
        ``_graph``/``_graph_inputs``/``_graph_out``), input copy, one
        ``replay()``, and the static output buffer.  ``None`` means
        "not served by the graph" — signature mismatch, no capture,
        or eager-forever — and the caller falls back to
        ``_forward_impl`` exactly as before.  The returned tensor is
        the static output buffer, overwritten by the next replay
        (clone it to keep it).
        """
        g = self._graph
        if g is None or len(xs) != len(self._graph_inputs):
            return None
        if not all(
            t.shape == b.shape
            and t.dtype == b.dtype
            and t.device == b.device
            for t, b in zip(xs, self._graph_inputs, strict=True)
        ):
            return None
        if not self._graph_replay_ready(xs):
            return None
        # A successful recapture swapped the graph AND the static
        # buffers — re-read both rather than replaying the stale
        # pre-check objects.
        g = self._graph
        if g is None:  # pragma: no cover — ready implies captured
            return None
        for buf, t in zip(self._graph_inputs, xs, strict=True):
            buf.copy_(t, non_blocking=True)
        g.replay()
        return self._graph_out

    # -- eval plumbing ------------------------------------------------

    def _input_env(
        self, xs: tuple[torch.Tensor, ...]
    ) -> tuple[Any, dict[str, Any]]:
        """Return the shared ``_forward_impl`` prologue: ``(x, env)``.

        ``x`` is the first positional arg (``None`` on zero args —
        the serial evaluator's "self" fallback); ``env`` maps
        ``"self"`` and every named IR input to the matching
        positional arg, defaulting missing ones to ``x``.
        """
        x = xs[0] if xs else None
        env: dict[str, Any] = {"self": x}
        for i, inp in enumerate(self._inputs):
            env[inp.name] = xs[i] if i < len(xs) else x
        return x, env

    def ev_factory(
        self,
        env: dict[str, Any],
        x: Any,
        memo_env: dict,
    ) -> Callable[[Any], Any]:
        """Build the ``ev(t)`` leaf-evaluation closure.

        Semantics of ``eval_mod._eval(t, env, x, memo_env)``: ``Op``
        results dedup through (and are written back into) the caller's
        ``memo_env``, and a hit returns the stored object verbatim.

        The fast route — ``eval_mod._eval_fast`` (the slot-tape plan,
        no memo walk) — applies while ``memo_env`` holds only entries
        this closure wrote: ``_eval_fast``'s internal dedup is
        structural, and the top-level write/hit above reproduces the
        observable memo contract.  Once the caller seeds memo with
        batched subterm results (omd's ``*_seeds`` — they must hit at
        ANY depth, not just the leaf root), the recursive evaluator
        takes over permanently for this env.  ``isinstance(t, Op)``
        gates the memo touchpoints exactly like ``eval_term`` (leaves
        never consult it).
        """
        eval_mod = self.eval_mod
        memo: dict[Any, Any] = memo_env if memo_env is not None else {}
        ours: set[Any] = set()

        def ev(t: Any) -> Any:
            if isinstance(t, Op):
                hit = memo.get(t)
                if hit is not None:
                    return hit
            if len(memo) == len(ours):
                out = eval_mod._eval_fast(t, env, x)
            else:
                out = eval_mod._eval(t, env, x, memo)
            if isinstance(t, Op):
                ours.add(t)
                memo[t] = out
            return out

        return ev

    def _cached(
        self,
        key: tuple,
        like: torch.Tensor,
        make,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        """Device/dtype-aware tensor cache (index tensors, pad rows).

        ``like`` supplies the device; ``dtype`` defaults to ``like``'s
        but is fixed ``torch.long`` for index tensors.
        """
        want = dtype or like.dtype
        t = self._const_cache.get(key)
        if t is None or t.device != like.device or t.dtype != want:
            t = make(like)
            self._const_cache[key] = t
        return t
