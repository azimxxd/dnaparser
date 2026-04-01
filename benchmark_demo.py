"""Run the backend on curated demo VCF files and print a short report.

This script is intentionally simple: it exercises the same parser and model
that the API uses, but without requiring you to boot FastAPI first.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import main


DEFAULT_CASES = (
    Path("demo_cases/high_signal.vcf"),
    Path("demo_cases/low_signal.vcf"),
    Path("demo_cases/mixed_signal.vcf"),
)


def _format_result(item: dict[str, object]) -> str:
    disease = str(item["associated_disease"])
    explanation = str(item["explanation"])
    confidence = str(item["confidence_level"])
    conflict = bool(item["reference_model_conflict"])
    return (
        f"{item['chromosome']}:{item['position']} {item['mutation']} | "
        f"score={item['risk_score']:.6f} | confidence={confidence} | "
        f"conflict={str(conflict).lower()} | disease={disease} | why={explanation}"
    )


def analyze_file(path: Path, batch_size: int) -> dict[str, object]:
    started_at = perf_counter()
    report = main.analyze_vcf_path(
        path,
        batch_size=batch_size,
    )
    elapsed = perf_counter() - started_at

    return {
        "path": path,
        "total_scanned": report["total_variants_scanned"],
        "dangerous_count": report["dangerous_variants_found"],
        "threshold_used": report["threshold_used"],
        "max_raw_score": report["max_raw_score"],
        "elapsed_seconds": round(elapsed, 3),
        "top_risks": report["top_risks"],
        "top_candidates": report["top_candidates"],
        "analysis_summary": report["analysis_summary"],
        "follow_up_guidance": report["follow_up_guidance"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run curated backend demo cases through the current model.",
    )
    parser.add_argument(
        "files",
        nargs="*",
        type=Path,
        help="Optional VCF files to analyze. Defaults to the curated demo cases.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50_000,
        help="Number of variants to score per batch.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=5,
        help="How many top-risk rows to print per file.",
    )
    return parser.parse_args()


def main_cli() -> None:
    args = parse_args()
    files = tuple(args.files) if args.files else DEFAULT_CASES

    for path in files:
        report = analyze_file(path, batch_size=args.batch_size)
        print(f"\n=== {report['path']} ===")
        print(f"scanned_variants: {report['total_scanned']}")
        print(f"dangerous_variants: {report['dangerous_count']}")
        print(f"threshold_used: {report['threshold_used']}")
        print(f"max_raw_score: {report['max_raw_score']}")
        print(f"elapsed_seconds: {report['elapsed_seconds']}")
        print(f"summary: {report['analysis_summary']['short_text']}")
        print(f"overall_alert_level: {report['analysis_summary']['overall_alert_level']}")
        print(f"urgency: {report['follow_up_guidance']['urgency']}")
        print("recommended_next_steps:")
        for step in report["follow_up_guidance"]["recommended_next_steps"][:4]:
            print(f"  - {step}")

        top_risks = report["top_risks"]
        if not top_risks:
            print("top_risks: none above the current threshold")
        else:
            print("top_risks:")
            for item in top_risks[: args.top]:
                print(f"  {_format_result(item)}")

        print("top_candidates:")
        for item in report["top_candidates"][: args.top]:
            print(f"  {_format_result(item)}")


if __name__ == "__main__":
    main_cli()
