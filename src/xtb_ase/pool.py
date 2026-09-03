"""Process-level execution for independent ASE calculations."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor
import copy
import itertools
import multiprocessing
import pickle
from typing import Any

from ase.calculators.calculator import Calculator, PropertyNotImplementedError, all_changes

from ._runtime import normalize_optional_int


CalculatorFactory = Callable[[], Calculator]


def _calculate_one(
    calculator_factory: CalculatorFactory,
    atoms,
    properties: tuple[str, ...],
) -> dict[str, Any]:
    calculator = calculator_factory()
    atoms_copy = atoms.copy()
    calculator.calculate(
        atoms_copy,
        properties=properties,
        system_changes=all_changes,
    )
    missing = [name for name in properties if name not in calculator.results]
    if missing:
        raise PropertyNotImplementedError(
            f"calculator did not return requested properties: {missing}"
        )
    return {name: copy.deepcopy(calculator.results[name]) for name in properties}


class CalculatorPool:
    """Run independent ASE calculations in isolated worker processes.

    ``calculator_factory`` must be a zero-argument callable.  For more than
    one worker it must be pickleable, so a module-level function or
    ``functools.partial`` is preferred over a lambda.
    """

    def __init__(
        self,
        calculator_factory: CalculatorFactory,
        *,
        processes: int = 1,
        mp_context: str | None = None,
    ) -> None:
        if not callable(calculator_factory):
            raise TypeError("calculator_factory must be callable")
        self.calculator_factory = calculator_factory
        self.processes = normalize_optional_int(processes, "processes", minimum=1)
        if self.processes is None:
            raise ValueError("processes must be a positive integer")
        self.mp_context = "spawn" if mp_context is None else mp_context
        try:
            multiprocessing.get_context(self.mp_context)
        except ValueError as exc:
            raise ValueError(
                f"unknown multiprocessing context: {self.mp_context!r}"
            ) from exc
        if self.processes > 1:
            try:
                pickle.dumps(calculator_factory)
            except (pickle.PickleError, TypeError, AttributeError) as exc:
                raise TypeError(
                    "calculator_factory must be pickleable when processes > 1; "
                    "use a module-level function or functools.partial"
                ) from exc

    @staticmethod
    def _normalize_properties(properties: Sequence[str] | str) -> tuple[str, ...]:
        if isinstance(properties, str):
            normalized = (properties,)
        else:
            normalized = tuple(properties)
        if not normalized or any(not isinstance(name, str) or not name for name in normalized):
            raise ValueError("properties must contain at least one non-empty name")
        return normalized

    def map(
        self,
        atoms: Iterable,
        properties: Sequence[str] | str = ("energy",),
    ) -> list[dict[str, Any]]:
        """Calculate each structure and return results in input order."""

        normalized_properties = self._normalize_properties(properties)
        if self.processes == 1:
            return [
                _calculate_one(self.calculator_factory, structure, normalized_properties)
                for structure in atoms
            ]

        context = multiprocessing.get_context(self.mp_context)
        with ProcessPoolExecutor(
            max_workers=self.processes,
            mp_context=context,
        ) as executor:
            return list(
                executor.map(
                    _calculate_one,
                    itertools.repeat(self.calculator_factory),
                    atoms,
                    itertools.repeat(normalized_properties),
                )
            )
