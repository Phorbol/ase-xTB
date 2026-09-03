"""Strict parsers for the machine-readable files emitted by xTB."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np


class OutputParseError(ValueError):
    """Raised when an xTB output file is incomplete or malformed."""


@dataclass(frozen=True)
class ParsedGradient:
    energy_hartree: float
    gradient_hartree_per_bohr: np.ndarray


@dataclass(frozen=True)
class ParsedProperties:
    total_energy_hartree: float | None = None
    homo_lumo_gap_ev: float | None = None
    homo_lumo_gap_hartree: float | None = None
    dipole_au: np.ndarray | None = None
    dipole_debye: float | None = None
    dipole_debye_vector: np.ndarray | None = None
    quadrupole_au: np.ndarray | None = None


@dataclass(frozen=True)
class ParsedOrbitals:
    energies_hartree: np.ndarray
    occupations: np.ndarray


_FLOAT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][-+]?\d+)?"
_FLOAT_RE = re.compile(rf"^{_FLOAT}$")


def _float(token: str) -> float:
    try:
        return float(token.replace("D", "E").replace("d", "e"))
    except ValueError as exc:
        raise OutputParseError(f"invalid numeric value {token!r}") from exc


def _first_numeric_line(lines: list[str], start: int, description: str) -> float:
    for line in lines[start:]:
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        fields = line.split()
        if len(fields) != 1 or not _FLOAT_RE.fullmatch(fields[0]):
            raise OutputParseError(f"malformed {description} line: {line!r}")
        return _float(fields[0])
    raise OutputParseError(f"missing {description}")


def _marker_index(lines: list[str], marker: str) -> int:
    for index, line in enumerate(lines):
        if marker in line:
            return index
    raise OutputParseError(f"missing section marker {marker!r}")


def parse_engrad(path: str | Path, natoms: int) -> ParsedGradient:
    """Parse xTB's ``.engrad`` energy and gradient file."""

    if natoms <= 0:
        raise ValueError("natoms must be positive")
    lines = Path(path).read_text().splitlines()
    energy_marker = _marker_index(lines, "The current total energy in Eh")
    gradient_marker = _marker_index(lines, "The current gradient in Eh/bohr")
    energy = _first_numeric_line(lines, energy_marker + 1, "total energy")

    values: list[float] = []
    started = False
    for line in lines[gradient_marker + 1 :]:
        if line.lstrip().startswith("#"):
            if started:
                break
            continue
        if not line.strip():
            continue
        fields = line.split()
        if any(not _FLOAT_RE.fullmatch(field) for field in fields):
            raise OutputParseError(f"malformed gradient line: {line!r}")
        started = True
        values.extend(_float(field) for field in fields)

    expected = 3 * natoms
    if len(values) != expected:
        raise OutputParseError(
            f"gradient has {len(values)} values; expected {expected}"
        )
    return ParsedGradient(energy, np.asarray(values, dtype=float).reshape(natoms, 3))


def parse_charges(path: str | Path, natoms: int) -> np.ndarray:
    """Parse one atomic charge per line from xTB's ``charges`` file."""

    if natoms <= 0:
        raise ValueError("natoms must be positive")
    values: list[float] = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 1 or not _FLOAT_RE.fullmatch(fields[0]):
            raise OutputParseError(f"malformed charge line: {line!r}")
        values.append(_float(fields[0]))
    if len(values) != natoms:
        raise OutputParseError(
            f"charges has {len(values)} values; expected {natoms}"
        )
    return np.asarray(values, dtype=float)


def parse_stdout_properties(text: str, natoms: int) -> ParsedProperties:
    """Parse stable scalar/vector properties from the normal xTB printout."""

    if natoms <= 0:
        raise ValueError("natoms must be positive")
    energy_match = re.search(
        rf"::\s*total energy\s+({_FLOAT})\s+Eh",
        text,
        flags=re.IGNORECASE,
    )
    gap_match = re.search(
        rf"(?:HOMO-LUMO gap|HL-Gap)\s+({_FLOAT})\s+Eh\s+({_FLOAT})\s+eV",
        text,
        flags=re.IGNORECASE,
    )

    dipole_au = None
    dipole_debye = None
    dipole_debye_vector = None
    dipole_section = text.split("Atomic dipole moments", maxsplit=1)
    if len(dipole_section) == 2:
        total_match = re.search(
            rf"^\s*total\s+({_FLOAT})\s+({_FLOAT})\s+({_FLOAT})\s*$",
            dipole_section[1],
            flags=re.IGNORECASE | re.MULTILINE,
        )
        if total_match:
            dipole_au = np.asarray(
                [_float(value) for value in total_match.groups()], dtype=float
            )
        magnitude_match = re.search(
            rf"\|total\|\s+({_FLOAT})\s+({_FLOAT})\s+Debye",
            dipole_section[1],
            flags=re.IGNORECASE,
        )
        if magnitude_match:
            dipole_debye = _float(magnitude_match.group(2))
    else:
        dipole_sections = text.split("molecular dipole", maxsplit=1)
        if len(dipole_sections) == 2:
            full_match = re.search(
                rf"^\s*full:?\s+({_FLOAT})\s+({_FLOAT})\s+({_FLOAT})\s+({_FLOAT})\s*$",
                dipole_sections[1],
                flags=re.IGNORECASE | re.MULTILINE,
            )
            if full_match:
                values = [_float(value) for value in full_match.groups()]
                dipole_debye_vector = np.asarray(values[:3], dtype=float)
                dipole_debye = values[3]

    quadrupole_au = None
    quadrupole_sections = text.split("Atomic quadrupole moments", maxsplit=1)
    if len(quadrupole_sections) == 2:
        quadrupole_match = re.search(
            rf"^\s*total\s+({_FLOAT})\s+({_FLOAT})\s+({_FLOAT})\s+"
            rf"({_FLOAT})\s+({_FLOAT})\s+({_FLOAT})\s*$",
            quadrupole_sections[1],
            flags=re.IGNORECASE | re.MULTILINE,
        )
        if quadrupole_match:
            quadrupole_au = np.asarray(
                [_float(value) for value in quadrupole_match.groups()], dtype=float
            )
    else:
        quadrupole_sections = text.split(
            "molecular quadrupole (traceless)", maxsplit=1
        )
        if len(quadrupole_sections) == 2:
            quadrupole_match = re.search(
                rf"^\s*full:?\s+({_FLOAT})\s+({_FLOAT})\s+({_FLOAT})\s+"
                rf"({_FLOAT})\s+({_FLOAT})\s+({_FLOAT})\s*$",
                quadrupole_sections[1],
                flags=re.IGNORECASE | re.MULTILINE,
            )
            if quadrupole_match:
                quadrupole_au = np.asarray(
                    [_float(value) for value in quadrupole_match.groups()],
                    dtype=float,
                )

    return ParsedProperties(
        total_energy_hartree=(
            _float(energy_match.group(1)) if energy_match else None
        ),
        homo_lumo_gap_ev=(_float(gap_match.group(2)) if gap_match else None),
        homo_lumo_gap_hartree=(
            _float(gap_match.group(1)) if gap_match else None
        ),
        dipole_au=dipole_au,
        dipole_debye=dipole_debye,
        dipole_debye_vector=dipole_debye_vector,
        quadrupole_au=quadrupole_au,
    )


def parse_wbo(path: str | Path, natoms: int) -> np.ndarray:
    """Parse xTB's emitted atom-pair WBO list into a symmetric matrix.

    xTB writes only the largest WBOs above its printout threshold.  Missing
    pairs are therefore represented by zero and do not imply an exact zero
    in the full AO bond-order matrix.
    """

    if natoms <= 0:
        raise ValueError("natoms must be positive")
    result = np.zeros((natoms, natoms), dtype=float)
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 3 or not fields[0].isdigit() or not fields[1].isdigit():
            raise OutputParseError(f"malformed WBO line: {line!r}")
        first, second = int(fields[0]), int(fields[1])
        if not (1 <= first <= natoms and 1 <= second <= natoms):
            raise OutputParseError(f"WBO atom index out of range: {line!r}")
        value = _float(fields[2])
        result[first - 1, second - 1] = value
        result[second - 1, first - 1] = value
    return result


def parse_molden(path: str | Path) -> ParsedOrbitals:
    """Parse orbital energies and occupations from a Molden file."""

    energies: list[float] = []
    occupations: list[float] = []
    current_energy: float | None = None
    in_mo_section = False
    for line in Path(path).read_text().splitlines():
        stripped = line.strip()
        if stripped.upper() == "[MO]":
            in_mo_section = True
            continue
        if not in_mo_section:
            continue
        if stripped.startswith("Ene="):
            current_energy = _float(stripped.split("=", 1)[1].strip())
        elif stripped.startswith("Occup="):
            if current_energy is None:
                raise OutputParseError("Molden occupation appears before energy")
            occupations.append(_float(stripped.split("=", 1)[1].strip()))
            energies.append(current_energy)
            current_energy = None
    if not energies:
        raise OutputParseError("Molden file contains no orbital records")
    if current_energy is not None:
        raise OutputParseError("Molden orbital is missing occupation")
    return ParsedOrbitals(
        np.asarray(energies, dtype=float), np.asarray(occupations, dtype=float)
    )


def parse_hessian(path: str | Path, natoms: int) -> np.ndarray:
    """Parse the flattened non-mass-weighted xTB Hessian."""

    if natoms <= 0:
        raise ValueError("natoms must be positive")
    lines = Path(path).read_text().splitlines()
    start = _marker_index(lines, "$hessian") + 1
    values: list[float] = []
    for line in lines[start:]:
        if line.lstrip().startswith("$"):
            break
        for field in line.split():
            if not _FLOAT_RE.fullmatch(field):
                raise OutputParseError(f"malformed Hessian value: {field!r}")
            values.append(_float(field))
    dimension = 3 * natoms
    expected = dimension * dimension
    if len(values) != expected:
        raise OutputParseError(
            f"Hessian has {len(values)} values; expected {expected}"
        )
    return np.asarray(values, dtype=float).reshape(dimension, dimension)


def parse_vibspectrum(path: str | Path) -> np.ndarray:
    """Parse mode frequencies in cm^-1 from xTB's ``vibspectrum`` file."""

    lines = Path(path).read_text().splitlines()
    start = _marker_index(lines, "$vibrational spectrum") + 1
    frequencies: list[float] = []
    for line in lines[start:]:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("$end"):
            break
        if stripped.startswith("#"):
            continue
        fields = stripped.split()
        if not fields or not fields[0].isdigit():
            raise OutputParseError(f"malformed vibrational mode line: {line!r}")
        if len(fields) < 2:
            raise OutputParseError(f"missing vibrational frequency: {line!r}")
        frequency_index = 1
        if not _FLOAT_RE.fullmatch(fields[frequency_index]):
            frequency_index = 2
        if frequency_index >= len(fields) or not _FLOAT_RE.fullmatch(
            fields[frequency_index]
        ):
            raise OutputParseError(f"missing vibrational frequency: {line!r}")
        frequencies.append(_float(fields[frequency_index]))
    if not frequencies:
        raise OutputParseError("vibrational spectrum contains no modes")
    return np.asarray(frequencies, dtype=float)
