#!/usr/bin/env python
"""CLI for the LLM-as-judge evaluator.

Usage:
    python scripts/run_llm_judge.py
    python scripts/run_llm_judge.py --no-adversarial
    python scripts/run_llm_judge.py --judge-model qwen3:8b --output reports/judge.json
    python scripts/run_llm_judge.py --tag adversarial --tag charter

Pass ``--print-failures`` to see only the cases whose average judge score is
below the pass threshold (3.5 / 5).

This script runs the LIVE chat workflow with the configured LLM provider
(``ollama`` by default), so it requires Ollama to be running and the configured
model to be pulled. It will take 1-3 minutes per case.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Make `app` importable regardless of cwd.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "apps" / "api"))

from app.core.config import get_settings  # noqa: E402
from app.evals.adversarial_cases import ADVERSARIAL_CASES  # noqa: E402
from app.evals.cases import EVAL_CASES  # noqa: E402
from app.evals.llm_judge import _JUDGE_PASS_THRESHOLD, run_llm_judge  # noqa: E402
from app.workflows.chat_workflow import ChatWorkflow  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-adversarial",
        action="store_true",
        help="Skip the extended adversarial / compound / detail-correctness cases.",
    )
    parser.add_argument(
        "--judge-model",
        default=None,
        help="Model name to use as the judge (default: project's configured llm_model).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write the full JSON report.",
    )
    parser.add_argument(
        "--tag",
        action="append",
        default=[],
        help="Only run cases that have any of these tags. Repeatable.",
    )
    parser.add_argument(
        "--print-failures",
        action="store_true",
        help="Print only failing cases at the end.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    settings = get_settings()

    cases = list(EVAL_CASES)
    if not args.no_adversarial:
        cases.extend(ADVERSARIAL_CASES)
    if args.tag:
        wanted = {tag.lower() for tag in args.tag}
        cases = [c for c in cases if any(t.lower() in wanted for t in c.tags)]

    if not cases:
        print("No cases matched the requested filters.", file=sys.stderr)
        return 1

    print(
        f"Running LLM-judge over {len(cases)} cases\n"
        f"  workflow LLM provider: {settings.llm_provider}\n"
        f"  judge model: {args.judge_model or settings.llm_model}\n"
        f"  pass threshold (avg score): {_JUDGE_PASS_THRESHOLD} / 5",
        flush=True,
    )

    workflow = ChatWorkflow(settings)
    started = time.monotonic()
    report = run_llm_judge(
        cases=cases,
        workflow=workflow,
        settings=settings,
        judge_model=args.judge_model,
    )
    elapsed = time.monotonic() - started

    print(
        f"\nFinished in {elapsed:.1f}s\n"
        f"  pass rate: {report.pass_rate * 100:.1f}% ({report.passed}/{report.total})\n"
        f"  avg accuracy:      {report.average_accuracy} / 5\n"
        f"  avg groundedness:  {report.average_groundedness} / 5\n"
        f"  avg relevance:     {report.average_relevance} / 5\n"
        f"  avg harm_avoidance:{report.average_harm_avoidance} / 5",
        flush=True,
    )

    if args.print_failures:
        failures = [s for s in report.scores if not s.passed]
        if not failures:
            print("\nAll cases passed.")
        else:
            print(f"\nFailures ({len(failures)}):")
            for s in failures:
                print(
                    f"  - [{','.join(s.tags) or '(no tags)'}] {s.case_name} "
                    f"avg={s.average} acc={s.accuracy} ground={s.groundedness} "
                    f"rel={s.relevance} harm={s.harm_avoidance}"
                )
                print(f"    reasoning: {s.reasoning[:300]}")
                if s.error:
                    print(f"    error: {s.error}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
        print(f"\nFull report written to {args.output}")

    return 0 if report.passed == report.total else 2


if __name__ == "__main__":
    sys.exit(main())
