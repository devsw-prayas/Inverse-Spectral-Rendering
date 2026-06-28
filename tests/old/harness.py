"""Shared gradient comparison harness.

Emits GradResult rows and StructResult rows; auto-regenerates the
Section 3 validation table (CSV + LaTeX) and a structural invariants table.

Usage:
    from tests.harness import GradResult, StructResult, run_three_way, Reporter
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

import torch

from src.gradient import fd_gradient


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class GradResult:
    test_id:   str
    scene:     str
    param:     str
    autograd:  float
    analytic:  float | None
    fd:        float
    rel_ag_fd: float
    rel_an_fd: float | None
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
    expected: float | None
    rel_err:  float | None
    tol:      float
    passed:   bool
    note:     str = ""


# ---------------------------------------------------------------------------
# Three-way gradient runner
# ---------------------------------------------------------------------------

def _rel(a: float, b: float) -> float:
    return abs(a - b) / max(abs(b), 1e-30)


def run_three_way(
    fn:         "callable[[], torch.Tensor]",
    param:      torch.Tensor,
    analytic:   float | None,
    test_id:    str,
    scene:      str,
    param_name: str,
    tol:        float = 5e-8,
) -> GradResult:
    """Run autograd / analytic / FD for a scalar loss that closes over param.

    fn:      () -> scalar tensor; param must be a leaf with requires_grad=True.
    param:   0-dim tensor, requires_grad=True.
    analytic: pre-computed analytic gradient (float), or None.
    """
    if param.grad is not None:
        param.grad.zero_()
    loss = fn()
    loss.backward()
    ag = param.grad.item()
    param.grad.zero_()

    fd = fd_gradient(fn, param).item()

    return GradResult(
        test_id=test_id,
        scene=scene,
        param=param_name,
        autograd=ag,
        analytic=analytic,
        fd=fd,
        rel_ag_fd=_rel(ag, fd),
        rel_an_fd=_rel(analytic, fd) if analytic is not None else None,
        tol=tol,
    )


# ---------------------------------------------------------------------------
# Reporter: collects results and emits tables
# ---------------------------------------------------------------------------

class Reporter:
    def __init__(self) -> None:
        self._grad: list[GradResult]   = []
        self._struct: list[StructResult] = []

    def add(self, result: GradResult | StructResult) -> None:
        if isinstance(result, GradResult):
            self._grad.append(result)
        else:
            self._struct.append(result)

    # --- gradient table ---

    def print_grad_table(self) -> None:
        _G_HDR = ("Test", "Scene", "Param", "Autograd", "Analytic", "FD (ref)",
                  "Rel AG/FD", "Rel An/FD", "Pass")
        _G_W   = (6, 20, 10, 13, 13, 13, 11, 11, 5)

        def _row(vals):
            return "  ".join(str(v).ljust(w) for v, w in zip(vals, _G_W))

        print("\n=== Section 3 Gradient Validation ===")
        print(_row(_G_HDR))
        print("-" * (sum(_G_W) + 2 * len(_G_W)))
        for r in self._grad:
            an  = f"{r.analytic:.6e}"  if r.analytic  is not None else "n/a"
            ran = f"{r.rel_an_fd:.2e}" if r.rel_an_fd is not None else "n/a"
            print(_row((
                r.test_id, r.scene, r.param,
                f"{r.autograd:.6e}", an, f"{r.fd:.6e}",
                f"{r.rel_ag_fd:.2e}", ran,
                "PASS" if r.passed else "FAIL",
            )))
        n_pass = sum(r.passed for r in self._grad)
        print(f"\n{n_pass}/{len(self._grad)} gradient tests passed.")

    # --- structural table ---

    def print_struct_table(self) -> None:
        _S_HDR = ("Test", "Quantity", "Measured", "Expected", "Rel Err", "Pass", "Note")
        _S_W   = (6, 30, 13, 13, 11, 5, 40)

        def _row(vals):
            return "  ".join(str(v).ljust(w) for v, w in zip(vals, _S_W))

        print("\n=== Structural Invariants ===")
        print(_row(_S_HDR))
        print("-" * (sum(_S_W) + 2 * len(_S_W)))
        for r in self._struct:
            exp = f"{r.expected:.4e}" if r.expected is not None else "n/a"
            re  = f"{r.rel_err:.2e}"  if r.rel_err  is not None else "n/a"
            print(_row((
                r.test_id, r.quantity,
                f"{r.measured:.6e}", exp, re,
                "PASS" if r.passed else "FAIL",
                r.note,
            )))
        n_pass = sum(r.passed for r in self._struct)
        print(f"\n{n_pass}/{len(self._struct)} structural tests passed.")

    def print_all(self) -> None:
        self.print_grad_table()
        self.print_struct_table()

    # --- file export ---

    def save_csv(self, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)

        with (out_dir / "validation_table.csv").open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["Test", "Scene", "Param", "Autograd", "Analytic", "FD",
                        "Rel_AG_FD", "Rel_An_FD", "Pass"])
            for r in self._grad:
                w.writerow([
                    r.test_id, r.scene, r.param,
                    r.autograd,
                    r.analytic if r.analytic is not None else "",
                    r.fd, r.rel_ag_fd,
                    r.rel_an_fd if r.rel_an_fd is not None else "",
                    "PASS" if r.passed else "FAIL",
                ])

        with (out_dir / "structural_table.csv").open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["Test", "Quantity", "Measured", "Expected", "RelErr", "Pass", "Note"])
            for r in self._struct:
                w.writerow([
                    r.test_id, r.quantity, r.measured,
                    r.expected if r.expected is not None else "",
                    r.rel_err  if r.rel_err  is not None else "",
                    "PASS" if r.passed else "FAIL",
                    r.note,
                ])

    def save_latex(self, out_dir: Path) -> None:
        """LaTeX tabular for Section 3 (gradient validation only)."""
        out_dir.mkdir(parents=True, exist_ok=True)
        lines = [
            r"\begin{tabular}{lllllll}",
            r"\toprule",
            r"Test & Param & Autograd & Analytic & FD (ref)"
            r" & $\epsilon_\mathrm{AG}$ & $\epsilon_\mathrm{an}$ \\",
            r"\midrule",
        ]
        for r in self._grad:
            an  = f"{r.analytic:.3e}" if r.analytic  is not None else "--"
            ran = f"{r.rel_an_fd:.0e}" if r.rel_an_fd is not None else "--"
            lines.append(
                f"\\texttt{{{r.test_id}}} & ${r.param}$ "
                f"& {r.autograd:.3e} & {an} & {r.fd:.3e}"
                f" & {r.rel_ag_fd:.0e} & {ran} \\\\"
            )
        lines += [r"\bottomrule", r"\end{tabular}"]
        (out_dir / "validation_table.tex").write_text("\n".join(lines))

    def all_passed(self) -> bool:
        return (all(r.passed for r in self._grad)
                and all(r.passed for r in self._struct))
