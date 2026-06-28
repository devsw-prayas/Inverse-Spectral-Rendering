"""Shared test harness: result types, runners, reporter.

Usage:
    from tests.harness import GradResult, StructResult, run_three_way, Reporter
"""
from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import torch

torch.set_default_dtype(torch.float64)

from src.gradient import fd_gradient


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class GradResult:
    test_id:   str
    label:     str
    param:     str
    autograd:  float
    analytic:  Optional[float]
    fd:        float
    rel_ag_fd: float
    rel_an_fd: Optional[float]
    tol:       float

    @property
    def passed(self) -> bool:
        if self.rel_ag_fd > self.tol:
            return False
        if self.analytic is not None and self.rel_an_fd > self.tol:
            return False
        return True


@dataclass
class StructResult:
    test_id:  str
    quantity: str
    measured: float
    expected: Optional[float]
    rel_err:  Optional[float]
    tol:      float
    passed:   bool
    note:     str = ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rel(a: float, b: float) -> float:
    return abs(a - b) / max(abs(b), 1e-30)


def run_three_way(
    fn:         Callable[[], torch.Tensor],
    param:      torch.Tensor,
    analytic:   Optional[float],
    test_id:    str,
    label:      str,
    param_name: str,
    tol:        float = 5e-8,
) -> GradResult:
    """autograd / analytic / central-FD for a scalar loss closing over param."""
    if param.grad is not None:
        param.grad.zero_()
    loss = fn()
    loss.backward()
    ag = param.grad.item()
    param.grad.zero_()

    fd = fd_gradient(fn, param).item()

    return GradResult(
        test_id=test_id,
        label=label,
        param=param_name,
        autograd=ag,
        analytic=analytic,
        fd=fd,
        rel_ag_fd=_rel(ag, fd),
        rel_an_fd=_rel(analytic, fd) if analytic is not None else None,
        tol=tol,
    )


# ---------------------------------------------------------------------------
# Reporter
# ---------------------------------------------------------------------------

class Reporter:
    def __init__(self) -> None:
        self._grad:   list[GradResult]   = []
        self._struct: list[StructResult] = []

    def add(self, result: GradResult | StructResult) -> None:
        if isinstance(result, GradResult):
            self._grad.append(result)
        else:
            self._struct.append(result)

    def print_grad_table(self) -> None:
        HDR = ("Test", "Label", "Param", "Autograd", "Analytic", "FD (ref)",
               "Rel AG/FD", "Rel An/FD", "Pass")
        W   = (6, 22, 10, 13, 13, 13, 11, 11, 5)

        def row(vals: tuple) -> str:
            return "  ".join(str(v).ljust(w) for v, w in zip(vals, W))

        print("\n=== Gradient Validation ===")
        print(row(HDR))
        print("-" * (sum(W) + 2 * len(W)))
        for r in self._grad:
            an  = f"{r.analytic:.6e}"  if r.analytic  is not None else "n/a"
            ran = f"{r.rel_an_fd:.2e}" if r.rel_an_fd is not None else "n/a"
            print(row((
                r.test_id, r.label, r.param,
                f"{r.autograd:.6e}", an, f"{r.fd:.6e}",
                f"{r.rel_ag_fd:.2e}", ran,
                "PASS" if r.passed else "FAIL",
            )))
        n = sum(r.passed for r in self._grad)
        print(f"\n{n}/{len(self._grad)} gradient tests passed.")

    def print_struct_table(self) -> None:
        HDR = ("Test", "Quantity", "Measured", "Expected", "Rel Err", "Pass", "Note")
        W   = (6, 32, 13, 13, 11, 5, 40)

        def row(vals: tuple) -> str:
            return "  ".join(str(v).ljust(w) for v, w in zip(vals, W))

        print("\n=== Structural / Analytic Checks ===")
        print(row(HDR))
        print("-" * (sum(W) + 2 * len(W)))
        for r in self._struct:
            exp = f"{r.expected:.4e}" if r.expected is not None else "n/a"
            re  = f"{r.rel_err:.2e}"  if r.rel_err  is not None else "n/a"
            print(row((
                r.test_id, r.quantity,
                f"{r.measured:.6e}", exp, re,
                "PASS" if r.passed else "FAIL",
                r.note,
            )))
        n = sum(r.passed for r in self._struct)
        print(f"\n{n}/{len(self._struct)} structural tests passed.")

    def print_all(self) -> None:
        self.print_grad_table()
        self.print_struct_table()

    def save_csv(self, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        with (out_dir / "grad_table.csv").open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["Test", "Label", "Param", "Autograd", "Analytic", "FD",
                        "Rel_AG_FD", "Rel_An_FD", "Pass"])
            for r in self._grad:
                w.writerow([r.test_id, r.label, r.param, r.autograd,
                            r.analytic if r.analytic is not None else "",
                            r.fd, r.rel_ag_fd,
                            r.rel_an_fd if r.rel_an_fd is not None else "",
                            "PASS" if r.passed else "FAIL"])
        with (out_dir / "struct_table.csv").open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["Test", "Quantity", "Measured", "Expected", "RelErr", "Pass", "Note"])
            for r in self._struct:
                w.writerow([r.test_id, r.quantity, r.measured,
                            r.expected if r.expected is not None else "",
                            r.rel_err  if r.rel_err  is not None else "",
                            "PASS" if r.passed else "FAIL",
                            r.note])

    def all_passed(self) -> bool:
        return (all(r.passed for r in self._grad)
                and all(r.passed for r in self._struct))
