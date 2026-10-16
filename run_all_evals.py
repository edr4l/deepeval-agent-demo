#!/usr/bin/env python3
"""
run_all_evals.py — CI-shaped eval runner for deepeval-agent-demo
================================================================

Discovers every test_*.py in evals/, runs each as a subprocess,
parses DeepEval's terminal output for per-case pass/fail/score,
and produces:
  1. A JSON summary (eval_results.json) — machine-readable for CI
  2. A terminal report — human-readable for review

Exit code:
  0 = all files passed (every test case in every file passed)
  1 = at least one file had a failure
  2 = at least one file errored (crashed, timed out, or unparseable output)

Usage:
  python run_all_evals.py                    # run all test files
  python run_all_evals.py test_rag_agent.py  # run one specific file
  python run_all_evals.py --timeout 120      # custom timeout per file (default: 180s)
  python run_all_evals.py --json-only        # suppress terminal report, print JSON to stdout

Designed to run from the repo root: ~/Workspace/evals/deepeval-agent-demo/
"""

import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TestCaseResult:
    """One test case inside one file."""
    case_id: str                      # e.g. "test_case_0"
    passed: bool
    metrics: list = field(default_factory=list)   # [{name, score, threshold, status, reason}]
    input_text: Optional[str] = None
    actual_output: Optional[str] = None
    expected_output: Optional[str] = None


@dataclass
class FileResult:
    """Results for one test file."""
    file: str
    status: str            # "passed", "failed", "error"
    total_cases: int = 0
    passed_cases: int = 0
    failed_cases: int = 0
    pass_rate: float = 0.0
    duration_seconds: float = 0.0
    cases: list = field(default_factory=list)     # [TestCaseResult]
    error_message: Optional[str] = None
    token_cost_usd: Optional[float] = None
    raw_summary: Optional[str] = None  # DeepEval's own summary line


@dataclass
class RunSummary:
    """Aggregate across all files."""
    timestamp: str
    total_files: int = 0
    passed_files: int = 0
    failed_files: int = 0
    errored_files: int = 0
    total_cases: int = 0
    passed_cases: int = 0
    failed_cases: int = 0
    overall_pass_rate: float = 0.0
    total_duration_seconds: float = 0.0
    files: list = field(default_factory=list)     # [FileResult]


# ---------------------------------------------------------------------------
# Output parser — extracts results from DeepEval's terminal output
# ---------------------------------------------------------------------------

def _clean_box_chars(text: str) -> str:
    """Strip DeepEval's box-drawing characters and trailing whitespace."""
    # Remove box-drawing chars: │ (U+2502), ┃ (U+2503), ╭╮╰╯─━╶╴ etc.
    cleaned = re.sub(r'[│┃╭╮╰╯─━╶╴┌┐└┘├┤┬┴┼╋]+', '', text)
    # Collapse multiple spaces and strip
    cleaned = re.sub(r' {2,}', ' ', cleaned).strip()
    return cleaned


def parse_deepeval_output(stdout: str, stderr: str) -> FileResult:
    """
    Parse DeepEval's terminal output into structured results.

    DeepEval prints two formats depending on execution pattern:

    1. evaluate() pattern (Sprint 3-4 files):
       ✅ test_case_0 (Passed N metrics)
       ❌ test_case_1
         ... Input / Actual / Expected ...
         Status | Metric | Score | Threshold | Reason
         FAIL   | Name   | 0.59  | 0.70      | ...

    2. evals_iterator pattern (Sprint 1-2 files):
       Same output format — DeepEval standardises the terminal rendering.

    Both end with:
       Test Results (N total tests):
         Pass Rate: X% | Passed: Y | Failed: Z
    """
    result = FileResult(file="", status="error")
    combined = stdout + "\n" + stderr
    cases = []

    # --- Parse individual test cases ---

    # Pattern 1: Passed test case — "✅ test_case_N (Passed M metrics)"
    pass_pattern = re.compile(
        r'[✅]\s*(test_case_\d+)\s*\(Passed\s+(\d+)\s+metrics?\)',
        re.UNICODE
    )
    for match in pass_pattern.finditer(combined):
        case_id = match.group(1)
        cases.append(TestCaseResult(
            case_id=case_id,
            passed=True,
            metrics=[{"status": "PASS", "count": int(match.group(2))}]
        ))

    # Pattern 2: Failed test case — "❌ test_case_N" followed by metric details
    # We need to extract the metric rows that follow each failed case
    fail_blocks = re.split(r'[❌]\s*(test_case_\d+)', combined)
    # fail_blocks alternates: [preamble, case_id, block, case_id, block, ...]
    for i in range(1, len(fail_blocks), 2):
        case_id = fail_blocks[i].strip()
        block = fail_blocks[i + 1] if i + 1 < len(fail_blocks) else ""

        # Extract input
        input_match = re.search(r'Input:\s*(.+?)(?:\n|$)', block)
        input_text = _clean_box_chars(input_match.group(1)) if input_match else None

        # Extract actual output (may be multi-line)
        actual_match = re.search(r'Actual Output:\s*(.+?)(?=Expected Output:|$)', block, re.DOTALL)
        actual_text = _clean_box_chars(actual_match.group(1))[:200] if actual_match else None

        # Extract expected output
        expected_match = re.search(r'Expected Output:\s*(.+?)(?=Metrics|$)', block, re.DOTALL)
        expected_text = _clean_box_chars(expected_match.group(1))[:200] if expected_match else None

        # Extract metric rows — look for "FAIL │ Name │ Score │ Threshold │ Reason"
        # DeepEval uses box-drawing characters: │ (U+2502) and ┃ (U+2503)
        metric_rows = re.findall(
            r'(PASS|FAIL)\s*[│┃]\s*(.+?)\s*[│┃]\s*([0-9.]+)\s*[│┃]\s*([0-9.]+)\s*[│┃]\s*(.+?)(?:\n|$)',
            block
        )
        metrics = []
        for status, name, score, threshold, reason in metric_rows:
            metrics.append({
                "name": name.strip(),
                "score": float(score),
                "threshold": float(threshold),
                "status": status.strip(),
                "reason": _clean_box_chars(reason)[:300]
            })

        cases.append(TestCaseResult(
            case_id=case_id,
            passed=False,
            metrics=metrics,
            input_text=input_text,
            actual_output=actual_text,
            expected_output=expected_text,
        ))

    # --- Parse the summary line ---
    summary_match = re.search(
        r'Test Results\s*\((\d+)\s*total\s*tests?\).*?'
        r'Pass Rate:\s*([0-9.]+)%\s*\|\s*Passed:\s*(\d+)\s*\|\s*Failed:\s*(\d+)',
        combined, re.DOTALL
    )
    if summary_match:
        result.total_cases = int(summary_match.group(1))
        result.pass_rate = float(summary_match.group(2))
        result.passed_cases = int(summary_match.group(3))
        result.failed_cases = int(summary_match.group(4))
        result.raw_summary = summary_match.group(0).strip()

    # --- Parse token cost ---
    cost_match = re.search(r'token cost:\s*([0-9.]+)\s*USD', combined)
    if cost_match:
        result.token_cost_usd = float(cost_match.group(1))

    # --- If we got no cases from parsing but got a summary, trust the summary ---
    if not cases and summary_match:
        # Create placeholder cases from the summary counts
        for i in range(result.passed_cases):
            cases.append(TestCaseResult(case_id=f"test_case_{i}", passed=True))
        for i in range(result.failed_cases):
            cases.append(TestCaseResult(
                case_id=f"test_case_{result.passed_cases + i}", passed=False
            ))

    # --- If we got cases from parsing, use those counts ---
    if cases and not summary_match:
        result.total_cases = len(cases)
        result.passed_cases = sum(1 for c in cases if c.passed)
        result.failed_cases = sum(1 for c in cases if not c.passed)
        result.pass_rate = (
            round(result.passed_cases / result.total_cases * 100, 1)
            if result.total_cases > 0 else 0.0
        )

    result.cases = cases

    # --- Determine status ---
    if result.total_cases == 0:
        # Couldn't parse anything — check for common error patterns
        if "ModuleNotFoundError" in combined or "ImportError" in combined:
            result.status = "error"
            err_match = re.search(r'(ModuleNotFoundError|ImportError):\s*(.+)', combined)
            result.error_message = err_match.group(0) if err_match else "Import error"
        elif "Traceback" in combined:
            result.status = "error"
            result.error_message = "Python traceback — see raw output"
        else:
            result.status = "error"
            result.error_message = "No test results parsed from output"
    elif result.failed_cases == 0:
        result.status = "passed"
    else:
        result.status = "failed"

    return result


# ---------------------------------------------------------------------------
# Runner — discovers and executes test files
# ---------------------------------------------------------------------------

def discover_test_files(evals_dir: str, specific_file: Optional[str] = None) -> list[str]:
    """Find all test_*.py files in the evals directory."""
    if specific_file:
        path = os.path.join(evals_dir, specific_file)
        if not os.path.exists(path):
            print(f"ERROR: {path} does not exist", file=sys.stderr)
            sys.exit(2)
        return [path]

    pattern = os.path.join(evals_dir, "test_*.py")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"ERROR: No test_*.py files found in {evals_dir}", file=sys.stderr)
        sys.exit(2)
    return files


def run_test_file(filepath: str, timeout: int, repo_root: str) -> FileResult:
    """Run a single test file as a subprocess and parse results."""
    filename = os.path.basename(filepath)
    print(f"\n{'=' * 60}")
    print(f"  Running: {filename}")
    print(f"{'=' * 60}")

    start = time.time()
    try:
        proc = subprocess.run(
            [sys.executable, filepath],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=repo_root,
            env={
                **os.environ,
                "DEEPEVAL_NO_INSPECT_PROMPT": "1",
                "OPIK_USAGE_REPORT_ENABLED": "false",
            }
        )
        duration = time.time() - start

        # Print stdout in real-time style (it's already captured, but show it)
        if proc.stdout:
            for line in proc.stdout.strip().split("\n")[-10:]:  # last 10 lines
                print(f"  {line}")

        result = parse_deepeval_output(proc.stdout, proc.stderr)
        result.file = filename
        result.duration_seconds = round(duration, 1)

        # If subprocess returned non-zero and we couldn't parse results
        if proc.returncode != 0 and result.total_cases == 0:
            result.status = "error"
            result.error_message = f"Exit code {proc.returncode}"
            if proc.stderr:
                # Grab last meaningful line from stderr
                err_lines = [l for l in proc.stderr.strip().split("\n") if l.strip()]
                if err_lines:
                    result.error_message += f": {err_lines[-1][:200]}"

    except subprocess.TimeoutExpired:
        duration = time.time() - start
        result = FileResult(
            file=filename,
            status="error",
            duration_seconds=round(duration, 1),
            error_message=f"Timed out after {timeout}s"
        )
    except Exception as e:
        duration = time.time() - start
        result = FileResult(
            file=filename,
            status="error",
            duration_seconds=round(duration, 1),
            error_message=str(e)
        )

    # Print per-file verdict
    icon = {"passed": "PASS", "failed": "FAIL", "error": "ERR!"}[result.status]
    print(f"\n  [{icon}] {filename}: {result.passed_cases}/{result.total_cases} "
          f"({result.pass_rate}%) in {result.duration_seconds}s")
    if result.error_message:
        print(f"  ERROR: {result.error_message}")

    return result


# ---------------------------------------------------------------------------
# Report — terminal and JSON output
# ---------------------------------------------------------------------------

def print_terminal_report(summary: RunSummary):
    """Print a human-readable summary to the terminal."""
    print("\n")
    print("=" * 70)
    print("  EVAL SUITE RESULTS")
    print("=" * 70)
    print(f"  Timestamp:  {summary.timestamp}")
    print(f"  Duration:   {summary.total_duration_seconds}s")
    print()

    # Per-file table
    print(f"  {'File':<45} {'Status':<8} {'Pass':>4} {'Fail':>4} {'Rate':>7} {'Time':>6}")
    print(f"  {'-'*45} {'-'*8} {'-'*4} {'-'*4} {'-'*7} {'-'*6}")
    for f in summary.files:
        icon = {"passed": " OK ", "failed": "FAIL", "error": "ERR!"}[f.status]
        rate = f"{f.pass_rate:.0f}%" if f.total_cases > 0 else "N/A"
        print(f"  {f.file:<45} [{icon}]  {f.passed_cases:>4} {f.failed_cases:>4} {rate:>7} {f.duration_seconds:>5.1f}s")
        if f.error_message:
            print(f"    -> {f.error_message[:70]}")

    print(f"  {'-'*45} {'-'*8} {'-'*4} {'-'*4} {'-'*7} {'-'*6}")
    print(f"  {'TOTAL':<45} {'':8} {summary.passed_cases:>4} {summary.failed_cases:>4} "
          f"{summary.overall_pass_rate:.0f}%".rstrip())
    print()

    # Failed test case details
    failed_cases = []
    for f in summary.files:
        for c in f.cases:
            if not c.passed:
                failed_cases.append((f.file, c))

    if failed_cases:
        print(f"  FAILED TEST CASES ({len(failed_cases)}):")
        print(f"  {'-'*68}")
        for filename, case in failed_cases:
            print(f"  {filename} > {case.case_id}")
            if case.input_text:
                print(f"    Input: {case.input_text[:80]}")
            for m in case.metrics:
                if isinstance(m, dict) and "name" in m:
                    print(f"    {m['status']}: {m['name']} "
                          f"(score={m['score']}, threshold={m['threshold']})")
                    if m.get("reason"):
                        print(f"      Reason: {m['reason'][:100]}")
        print()

    # Errored files
    errored = [f for f in summary.files if f.status == "error"]
    if errored:
        print(f"  ERRORED FILES ({len(errored)}):")
        print(f"  {'-'*68}")
        for f in errored:
            print(f"  {f.file}: {f.error_message}")
        print()

    # Verdict
    if summary.errored_files > 0:
        verdict = "ERROR — some files could not run"
    elif summary.failed_files > 0:
        verdict = f"FAILED — {summary.failed_cases} test case(s) across {summary.failed_files} file(s)"
    else:
        verdict = f"PASSED — all {summary.total_cases} test cases across {summary.total_files} files"

    print(f"  VERDICT: {verdict}")
    print("=" * 70)


def write_json_summary(summary: RunSummary, output_path: str):
    """Write JSON summary to file."""

    def serialise(obj):
        """Convert dataclass instances to dicts."""
        if hasattr(obj, '__dataclass_fields__'):
            return asdict(obj)
        return obj

    with open(output_path, "w") as f:
        json.dump(asdict(summary), f, indent=2, default=str)

    print(f"\n  JSON summary written to: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run all DeepEval test files and produce a summary."
    )
    parser.add_argument(
        "file", nargs="?", default=None,
        help="Run a specific test file (e.g. test_rag_agent.py)"
    )
    parser.add_argument(
        "--timeout", type=int, default=180,
        help="Timeout per file in seconds (default: 180)"
    )
    parser.add_argument(
        "--json-only", action="store_true",
        help="Suppress terminal report, print JSON to stdout"
    )
    parser.add_argument(
        "--output", type=str, default="eval_results.json",
        help="JSON output file path (default: eval_results.json)"
    )
    args = parser.parse_args()

    # Determine repo root — script should be run from repo root
    # or we find it relative to this script's location
    repo_root = os.getcwd()
    evals_dir = os.path.join(repo_root, "evals")
    if not os.path.isdir(evals_dir):
        # Try relative to script location
        script_dir = os.path.dirname(os.path.abspath(__file__))
        evals_dir = os.path.join(script_dir, "evals")
        repo_root = script_dir
    if not os.path.isdir(evals_dir):
        print(f"ERROR: Cannot find evals/ directory. Run from repo root.", file=sys.stderr)
        sys.exit(2)

    # Discover test files
    test_files = discover_test_files(evals_dir, args.file)

    print(f"\nDiscovered {len(test_files)} test file(s):")
    for f in test_files:
        print(f"  - {os.path.basename(f)}")

    # Run each file
    summary = RunSummary(
        timestamp=datetime.now(timezone.utc).isoformat(),
        total_files=len(test_files),
    )

    suite_start = time.time()

    for filepath in test_files:
        file_result = run_test_file(filepath, args.timeout, repo_root)
        summary.files.append(file_result)

        # Accumulate totals
        summary.total_cases += file_result.total_cases
        summary.passed_cases += file_result.passed_cases
        summary.failed_cases += file_result.failed_cases

        if file_result.status == "passed":
            summary.passed_files += 1
        elif file_result.status == "failed":
            summary.failed_files += 1
        else:
            summary.errored_files += 1

    summary.total_duration_seconds = round(time.time() - suite_start, 1)
    summary.overall_pass_rate = (
        round(summary.passed_cases / summary.total_cases * 100, 1)
        if summary.total_cases > 0 else 0.0
    )

    # Output
    if args.json_only:
        print(json.dumps(asdict(summary), indent=2, default=str))
    else:
        print_terminal_report(summary)
        write_json_summary(summary, args.output)

    # Exit code
    if summary.errored_files > 0:
        sys.exit(2)
    elif summary.failed_files > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
