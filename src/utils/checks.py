"""Pass/fail record of numerical verification checks, saved with every result."""

from __future__ import annotations


class Checks:
    """Comparisons of computed values with expected values.

    Each check passes if its value (an error, so non-negative) is at most its
    tolerance. A NaN value fails, so a broken computation cannot pass silently.
    """

    def __init__(self) -> None:
        self.records: list[dict] = []

    def add(self, name: str, value: float, tolerance: float) -> None:
        value = float(value)
        self.records.append({"name": name, "value": value, "tolerance": tolerance, "passed": value <= tolerance})

    @property
    def all_passed(self) -> bool:
        return all(record["passed"] for record in self.records)

    def report(self) -> str:
        """One line with the pass count, plus one line per failed check."""
        n_passed = sum(record["passed"] for record in self.records)
        lines = [f"Checks passed: {n_passed}/{len(self.records)}"]
        for record in self.records:
            if not record["passed"]:
                lines.append(f"  FAILED  {record['name']}: {record['value']:.3e} > {record['tolerance']:.1e}")
        return "\n".join(lines)
