"""Parallel/concurrent fitness evaluators (V0–V3)."""
from __future__ import annotations

from typing import Any, Callable

from pso.parallel.base import FitnessEvaluator
from pso.parallel.v0_sequential import SequentialEvaluator
from pso.parallel.v1_threading import ThreadingEvaluator
from pso.parallel.v2_multiprocessing import MultiprocessingEvaluator
from pso.parallel.v3_asyncio import AsyncioEvaluator

__all__ = [
    "FitnessEvaluator",
    "SequentialEvaluator",
    "ThreadingEvaluator",
    "MultiprocessingEvaluator",
    "AsyncioEvaluator",
    "build_evaluator",
]

_EVALUATORS = {
    "v0": SequentialEvaluator,
    "v1": ThreadingEvaluator,
    "v2": MultiprocessingEvaluator,
    "v3": AsyncioEvaluator,
}


def build_evaluator(
    name: str,
    objective: Callable,
    **kwargs: Any,
) -> FitnessEvaluator:
    """Factory — *name* is ``'v0'``, ``'v1'``, ``'v2'``, or ``'v3'``."""
    try:
        cls = _EVALUATORS[name]
    except KeyError:
        raise ValueError(f"Unknown strategy '{name}'. Choose from {list(_EVALUATORS)}")
    return cls(objective, **kwargs)
