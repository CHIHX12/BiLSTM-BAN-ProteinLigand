#!/usr/bin/env python3
"""
Test Runner — HTS Drug-Protein Screening System
================================================
Runs the complete test suite and generates a structured report.

Usage:
    python run_tests.py [options]

Options:
    --fast       Skip slow model benchmark tests (unit tests only)
    --report     Save HTML report  (default: test_report.html)
    --verbose    Show individual test output
    --no-bio     Skip biological plausibility tests

Exit codes:
    0 = All tests passed
    1 = One or more tests failed
    2 = Dependency/import error
"""

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime

_HERE = os.path.dirname(os.path.abspath(__file__))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run HTS screening test suite")
    p.add_argument("--fast",    action="store_true", help="Skip model benchmark tests")
    p.add_argument("--no-bio",  action="store_true", help="Skip biological plausibility tests")
    p.add_argument("--verbose", action="store_true", help="Show verbose test output")
    p.add_argument("--report",  default=os.path.join(_HERE, "test_report.txt"),
                   help="Path to save test report")
    return p.parse_args()


def check_pytest() -> bool:
    try:
        import pytest
        return True
    except ImportError:
        print("[ERROR] pytest not installed. Install with: pip install pytest scipy")
        return False


def run_test_module(module_path: str, label: str, verbose: bool, extra_args=None) -> dict:
    """Run a single pytest module, return result dict."""
    cmd = [sys.executable, "-m", "pytest", module_path,
           "-v" if verbose else "-q",
           "--tb=short",
           "--no-header",
           f"--color=no"]
    if extra_args:
        cmd += extra_args

    t0 = time.perf_counter()
    result = subprocess.run(cmd, capture_output=not verbose,
                            text=True, cwd=_HERE)
    elapsed = time.perf_counter() - t0

    stdout = result.stdout if not verbose else ""
    stderr = result.stderr if not verbose else ""
    output = stdout + stderr

    # Parse pytest output
    passed = failed = errors = skipped = 0
    for line in output.splitlines():
        if " passed" in line:
            import re
            m = re.search(r"(\d+) passed", line)
            if m: passed = int(m.group(1))
        if " failed" in line:
            import re
            m = re.search(r"(\d+) failed", line)
            if m: failed = int(m.group(1))
        if " error" in line:
            import re
            m = re.search(r"(\d+) error", line)
            if m: errors = int(m.group(1))
        if " skipped" in line:
            import re
            m = re.search(r"(\d+) skipped", line)
            if m: skipped = int(m.group(1))

    return {
        "label":   label,
        "passed":  passed,
        "failed":  failed,
        "errors":  errors,
        "skipped": skipped,
        "elapsed": elapsed,
        "returncode": result.returncode,
        "output": output,
    }


def _status_icon(r: dict) -> str:
    if r["returncode"] == 0:
        return "PASS"
    if r["failed"] == 0 and r["errors"] == 0:
        return "SKIP"
    return "FAIL"


def main() -> int:
    args = parse_args()

    if not check_pytest():
        return 2

    print("=" * 68)
    print("  HTS Drug-Protein Screening — Test Suite")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 68)

    # ── Define test modules ────────────────────────────────────────────
    tests_dir = os.path.join(_HERE, "tests")
    modules = [
        (os.path.join(tests_dir, "test_smiles_validator.py"),
         "SMILES Validator (unit tests)"),
        (os.path.join(tests_dir, "test_protein_validator.py"),
         "Protein Validator (unit tests)"),
        (os.path.join(tests_dir, "test_input_loader.py"),
         "Input Loader (unit tests)"),
    ]

    if not args.fast:
        modules.append((
            os.path.join(tests_dir, "test_model_benchmark.py"),
            "Model Benchmark (AUROC verification on BindingDB)"
        ))

    if not args.no_bio:
        modules.append((
            os.path.join(tests_dir, "test_biological_plausibility.py"),
            "Biological Plausibility (known drug-target pairs)"
        ))

    # ── Run each module ────────────────────────────────────────────────
    results = []
    total_start = time.perf_counter()

    for module_path, label in modules:
        print(f"\n  Running: {label}")
        print(f"  {'─' * 60}")
        r = run_test_module(module_path, label, args.verbose)
        results.append(r)

        icon = _status_icon(r)
        print(f"  [{icon}]  passed={r['passed']}  failed={r['failed']}  "
              f"skipped={r['skipped']}  ({r['elapsed']:.1f}s)")

        # Show failures immediately
        if r["failed"] > 0 or r["errors"] > 0:
            print("\n  ── FAILURE DETAILS ──────────────────────────────────")
            for line in r["output"].splitlines():
                if any(kw in line for kw in ["FAILED", "ERROR", "assert", "AssertionError", "E  "]):
                    print(f"  {line}")
            print()

    total_elapsed = time.perf_counter() - total_start

    # ── Summary ───────────────────────────────────────────────────────
    all_passed  = sum(r["passed"]  for r in results)
    all_failed  = sum(r["failed"]  for r in results)
    all_errors  = sum(r["errors"]  for r in results)
    all_skipped = sum(r["skipped"] for r in results)
    overall_ok  = all(r["returncode"] == 0 for r in results)

    print()
    print("=" * 68)
    print("  SUMMARY")
    print("=" * 68)
    print(f"  {'Module':<45} {'Status':>6}  {'T(s)':>5}")
    print(f"  {'─' * 60}")
    for r in results:
        icon = _status_icon(r)
        print(f"  {r['label']:<45} {icon:>6}  {r['elapsed']:>4.1f}s")

    print(f"  {'─' * 60}")
    print(f"  Total: {all_passed} passed, {all_failed} failed, "
          f"{all_skipped} skipped  ({total_elapsed:.1f}s)")
    print()

    if overall_ok:
        print("  ✓  ALL TESTS PASSED")
    else:
        print("  ✗  SOME TESTS FAILED — review details above")

    # ── Write report ──────────────────────────────────────────────────
    report_lines = [
        f"HTS Screening — Test Report",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"",
        f"{'Module':<50} Status  Pass  Fail  Skip  Time",
        f"{'─'*80}",
    ]
    for r in results:
        icon = _status_icon(r)
        report_lines.append(
            f"{r['label']:<50} {icon:>6}  {r['passed']:>4}  "
            f"{r['failed']:>4}  {r['skipped']:>4}  {r['elapsed']:>4.1f}s"
        )
    report_lines += [
        f"{'─'*80}",
        f"TOTAL: {all_passed} passed, {all_failed} failed, "
        f"{all_errors} errors, {all_skipped} skipped",
        f"RESULT: {'PASS' if overall_ok else 'FAIL'}",
        f"",
        "=" * 80,
        "DETAILED OUTPUT",
        "=" * 80,
    ]
    for r in results:
        report_lines += ["", f"── {r['label']} ──", r["output"] or "(verbose mode — output to console)"]

    with open(args.report, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"\n  Report saved to: {args.report}")
    print("=" * 68)

    return 0 if overall_ok else 1


if __name__ == "__main__":
    sys.exit(main())
