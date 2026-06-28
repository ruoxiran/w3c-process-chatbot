#!/usr/bin/env python3
"""Deterministic structural eval — runs in CI on every PR.

Uses the ``template`` LLM provider so the workflow's answer is the
hand-coded fallback, not a real LLM call. That makes the run fast
(seconds, not minutes), free, and reproducible across CI agents —
the score depends ONLY on the retrieval, planning, and scoring
code, not on a stochastic generation step.

Exit code:
  0 — all cases passed
  1 — structural regression (at least one case failed)

If ``$GITHUB_STEP_SUMMARY`` is set (i.e. running inside a GitHub
Actions job), a markdown summary of the score + any failures is
appended there so it surfaces in the workflow run UI without
operators needing to dig into the logs.

Usage:
    python scripts/run_deterministic_eval.py                  # core eval only
    python scripts/run_deterministic_eval.py --include-adversarial
    python scripts/run_deterministic_eval.py --adversarial-threshold 0.66
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


# Make ``app`` importable regardless of cwd.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "apps" / "api"))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--include-adversarial",
        action="store_true",
        help="Also score the extended adversarial / compound cases. "
        "These are harder and the pass bar is a ratio, not 100%%.",
    )
    parser.add_argument(
        "--adversarial-threshold",
        type=float,
        default=0.66,
        help="Minimum adversarial pass ratio (default: 0.66 = 20/30).",
    )
    return parser.parse_args()


def _write_summary(text: str) -> None:
    """Append to the GitHub Actions step summary if available."""
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not path:
        return
    try:
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(text)
            if not text.endswith("\n"):
                handle.write("\n")
    except OSError:
        pass  # Summary is informational; never block the run on its failure.


def main() -> int:
    args = _parse_args()

    # Import here so the script's --help works even if app/ has a
    # heavy import-time side effect (logging setup, etc.).
    from app.core.config import Settings
    from app.evals.adversarial_cases import ADVERSARIAL_CASES
    from app.evals.cases import EVAL_CASES
    from app.evals.runner import run_eval_cases
    from app.evals.workflow import build_eval_workflow

    workflow = build_eval_workflow(Settings())

    # ---- Core structural eval — must be 100% ----
    core_report = run_eval_cases(EVAL_CASES, workflow.run)
    core_pass = core_report.passed_count == core_report.total_count

    lines = [
        "## Deterministic eval",
        "",
        f"**Core: {core_report.passed_count}/{core_report.total_count}** "
        f"({core_report.score * 100:.0f}%)",
    ]
    if not core_pass:
        failed = [r for r in core_report.results if not r.passed]
        lines.append("")
        lines.append(f"<details><summary>{len(failed)} failing core cases</summary>")
        lines.append("")
        for result in failed:
            lines.append(f"- **{result.name}** — {result.details or 'no detail'}")
        lines.append("</details>")

    adversarial_pass = True
    if args.include_adversarial:
        adv_report = run_eval_cases(ADVERSARIAL_CASES, workflow.run)
        adv_score = adv_report.score
        adversarial_pass = adv_score >= args.adversarial_threshold
        lines.append("")
        lines.append(
            f"**Adversarial: {adv_report.passed_count}/{adv_report.total_count}** "
            f"({adv_score * 100:.0f}%, threshold {args.adversarial_threshold * 100:.0f}%) — "
            f"{'PASS' if adversarial_pass else 'FAIL'}"
        )
        if not adversarial_pass:
            failed = [r for r in adv_report.results if not r.passed]
            lines.append("")
            lines.append(f"<details><summary>{len(failed)} failing adversarial cases</summary>")
            lines.append("")
            for result in failed:
                reasons = "; ".join(result.failures) or "no detail"
                lines.append(f"- **{result.case_name}** — {reasons}")
            lines.append("</details>")

    summary = "\n".join(lines) + "\n"
    print(summary)
    _write_summary(summary)

    return 0 if (core_pass and adversarial_pass) else 1


if __name__ == "__main__":
    sys.exit(main())
