#!/usr/bin/env python3
import argparse
import json
import os
from collections import Counter


def load_eval(path: str):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def summarize(run_name: str, runs_dir: str):
    eval_path = os.path.join(runs_dir, run_name, "eval_results.json")
    data = load_eval(eval_path)
    if data is None:
        return None

    total = compiled = correct = skipped = 0
    runtime_errors = Counter()
    correctness_issues = Counter()

    for _, samples in data.items():
        for s in samples:
            total += 1
            if s.get("compiled"):
                compiled += 1
            if s.get("correctness"):
                correct += 1
            if s.get("metadata", {}).get("skipped"):
                skipped += 1
            meta = s.get("metadata", {}) or {}
            if meta.get("runtime_error_name"):
                runtime_errors[meta["runtime_error_name"]] += 1
            if meta.get("correctness_issue"):
                correctness_issues[meta["correctness_issue"]] += 1

    return {
        "run_name": run_name,
        "eval_path": eval_path,
        "total": total,
        "compiled": compiled,
        "correct": correct,
        "skipped": skipped,
        "runtime_errors": runtime_errors,
        "correctness_issues": correctness_issues,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runs-dir",
        default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs"),
        help="Runs directory (default: repo/runs)",
    )
    parser.add_argument(
        "--run-names",
        default="mi300_k2_5_level1,mi300_k2_5_level2,mi300_k2_5_level3,mi300_k2_5_level4",
        help="Comma-separated run names",
    )
    args = parser.parse_args()

    run_names = [r.strip() for r in args.run_names.split(",") if r.strip()]
    for run_name in run_names:
        summary = summarize(run_name, args.runs_dir)
        if summary is None:
            print(f"{run_name}: eval_results.json not found")
            continue

        total = summary["total"]
        compiled = summary["compiled"]
        correct = summary["correct"]
        skipped = summary["skipped"]
        compiled_rate = (compiled / total * 100.0) if total else 0.0
        correct_rate = (correct / total * 100.0) if total else 0.0

        print(f"{run_name}")
        print(f"  eval_results: {summary['eval_path']}")
        print(f"  total samples: {total}")
        print(f"  compiled: {compiled} ({compiled_rate:.1f}%)")
        print(f"  correct: {correct} ({correct_rate:.1f}%)")
        print(f"  skipped: {skipped}")

        if summary["runtime_errors"]:
            top_runtime = summary["runtime_errors"].most_common(5)
            print("  top runtime errors:")
            for name, count in top_runtime:
                print(f"    - {name}: {count}")
        if summary["correctness_issues"]:
            top_correct = summary["correctness_issues"].most_common(5)
            print("  top correctness issues:")
            for name, count in top_correct:
                print(f"    - {name}: {count}")
        print("")


if __name__ == "__main__":
    main()
