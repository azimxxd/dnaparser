"""Refresh demo VCF files from the current ClinVar snapshot and model.

The goal is simple: after retraining, pick a few real strong positive rows
and a few real strong negative rows so the demo cases stay aligned with the
latest model instead of drifting out of date.
"""

from __future__ import annotations

import argparse
import gzip
import heapq
from pathlib import Path

import main
import train_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate real demo VCF files from the current ClinVar + model.",
    )
    parser.add_argument(
        "--clinvar-path",
        default="clinvar.vcf.gz",
        help="Path to the ClinVar VCF used for scanning examples.",
    )
    parser.add_argument(
        "--output-dir",
        default="demo_cases",
        help="Directory where high/low/mixed demo VCFs will be written.",
    )
    parser.add_argument(
        "--scan-limit",
        type=int,
        default=500_000,
        help="How many ClinVar rows to scan while mining examples.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=20_000,
        help="Batch size used while scoring candidate examples.",
    )
    parser.add_argument(
        "--rows-per-case",
        type=int,
        default=5,
        help="How many rows to put into each demo file.",
    )
    return parser.parse_args()


def _push_high(
    heap: list[tuple[float, str]],
    score: float,
    line: str,
    limit: int,
) -> None:
    entry = (score, line)
    if len(heap) < limit:
        heapq.heappush(heap, entry)
        return
    if score > heap[0][0]:
        heapq.heapreplace(heap, entry)


def _push_low(
    heap: list[tuple[float, str]],
    score: float,
    line: str,
    limit: int,
) -> None:
    # Keep the smallest scores in a max-like heap via negative values.
    entry = (-score, line)
    if len(heap) < limit:
        heapq.heappush(heap, entry)
        return
    if -score > heap[0][0]:
        heapq.heapreplace(heap, entry)


def _write_vcf(path: Path, lines: list[str]) -> None:
    header = [
        "##fileformat=VCFv4.3",
        "##source=dnaparser_refresh_demo_cases",
        '##INFO=<ID=AF_ESP,Number=1,Type=Float,Description="Population allele frequency from ESP">',
        '##INFO=<ID=AF_EXAC,Number=1,Type=Float,Description="Population allele frequency from ExAC">',
        '##INFO=<ID=AF_TGP,Number=1,Type=Float,Description="Population allele frequency from 1000 Genomes">',
        '##INFO=<ID=CLNVC,Number=1,Type=String,Description="ClinVar variant class">',
        '##INFO=<ID=GENEINFO,Number=1,Type=String,Description="Gene symbol and identifier">',
        '##INFO=<ID=MC,Number=.,Type=String,Description="Molecular consequence">',
        '##INFO=<ID=ORIGIN,Number=.,Type=String,Description="Origin code">',
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO",
    ]
    content = "\n".join(header + lines) + "\n"
    path.write_text(content, encoding="utf-8")


def main_cli() -> None:
    args = parse_args()
    clinvar_path = Path(args.clinvar_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    high_heap: list[tuple[float, str]] = []
    low_heap: list[tuple[float, str]] = []
    batch_variants: list[dict[str, str]] = []
    batch_meta: list[tuple[int, str]] = []
    processed = 0

    with gzip.open(clinvar_path, "rt", encoding="utf-8") as fh:
        for raw_line in fh:
            if raw_line.startswith("#"):
                continue

            processed += 1
            fields = raw_line.rstrip("\n").split("\t", 8)
            info = fields[7]
            info_map = train_model._parse_info(info)
            clnsig = info_map.get("CLNSIG")
            if not clnsig:
                continue

            label = train_model.map_label(clnsig)
            if label is None:
                continue

            ref = fields[3]
            alt = fields[4]
            if len(ref) != 1 or len(alt) != 1:
                continue

            batch_variants.append(
                {
                    "CHROM": fields[0],
                    "POS": fields[1],
                    "REF": ref,
                    "ALT": alt,
                    "INFO": info,
                }
            )
            batch_meta.append((label, raw_line.rstrip("\n")))

            if len(batch_variants) >= args.batch_size:
                scores = main.score_batch(batch_variants)
                for score, (row_label, line) in zip(scores, batch_meta):
                    score = float(score)
                    if row_label == 1:
                        _push_high(high_heap, score, line, args.rows_per_case)
                    else:
                        _push_low(low_heap, score, line, args.rows_per_case)
                batch_variants.clear()
                batch_meta.clear()

            if processed >= args.scan_limit:
                break

    if batch_variants:
        scores = main.score_batch(batch_variants)
        for score, (row_label, line) in zip(scores, batch_meta):
            score = float(score)
            if row_label == 1:
                _push_high(high_heap, score, line, args.rows_per_case)
            else:
                _push_low(low_heap, score, line, args.rows_per_case)

    high_lines = [line for _, line in sorted(high_heap, key=lambda item: item[0], reverse=True)]
    low_lines = [line for neg_score, line in sorted(low_heap, key=lambda item: item[0])]
    mixed_lines = high_lines + low_lines

    if len(high_lines) < args.rows_per_case or len(low_lines) < args.rows_per_case:
        raise RuntimeError(
            "Not enough demo rows were found. Increase --scan-limit and try again."
        )

    _write_vcf(output_dir / "high_signal.vcf", high_lines)
    _write_vcf(output_dir / "low_signal.vcf", low_lines)
    _write_vcf(output_dir / "mixed_signal.vcf", mixed_lines)

    print(f"Scanned rows: {processed}")
    print("High-signal example scores:")
    for score, _ in sorted(high_heap, key=lambda item: item[0], reverse=True):
        print(f"  {score:.6f}")
    print("Low-signal example scores:")
    for neg_score, _ in sorted(low_heap, key=lambda item: item[0]):
        print(f"  {-neg_score:.6f}")
    print(f"Wrote refreshed demo files to {output_dir}")


if __name__ == "__main__":
    main_cli()
