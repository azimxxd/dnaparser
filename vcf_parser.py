"""
High-performance, memory-efficient VCF (Variant Call Format) parser.

Designed for multi-GB VCF files under strict RAM constraints.
Uses generator-based lazy evaluation to maintain O(batch_size) memory complexity.

Optimization choices:
- Generator pattern (yield) avoids loading the entire file into memory.
- Bounded tab splitting (split('\t', N)) stops after the needed columns,
  preventing allocation of substrings for INFO, FORMAT, and sample columns
  which can be hundreds of KB per line.
- Column indices are resolved dynamically from the #CHROM header so the
  parser tolerates non-standard column orderings.
- Batching amortises per-yield overhead and aligns with downstream
  consumers that operate on NumPy arrays or DataFrame chunks.
- Malformed rows are logged and skipped rather than crashing the generator.
"""

from __future__ import annotations

import io
import gzip
import logging
from pathlib import Path
from typing import Generator, TextIO

logger = logging.getLogger(__name__)

# Core columns required by downstream code.
REQUIRED_COLUMNS = ("CHROM", "POS", "REF", "ALT")

# VCF standard names (the header line prefixes CHROM with '#').
_VCF_STANDARD_HEADER = {
    "CHROM": 0,
    "POS": 1,
    "ID": 2,
    "REF": 3,
    "ALT": 4,
    "QUAL": 5,
    "FILTER": 6,
    "INFO": 7,
}

VariantBatch = list[dict[str, str]]


def _resolve_column_indices(
    header_line: str,
    required_columns: tuple[str, ...] = REQUIRED_COLUMNS,
) -> tuple[dict[str, int], int]:
    """Map required column names to their 0-based indices in the header.

    Returns:
        A tuple of (column_map, max_split) where *max_split* is the
        minimum number of splits needed to reach every required column,
        keeping split cost as low as possible.
    """
    # Strip the leading '#' and trailing newline, then split on tabs.
    columns = header_line.lstrip("#").rstrip("\n\r").split("\t")
    name_to_idx: dict[str, int] = {col: idx for idx, col in enumerate(columns)}

    col_map: dict[str, int] = {}
    for col in required_columns:
        if col not in name_to_idx:
            raise ValueError(
                f"Required column '{col}' not found in VCF header. "
                f"Available columns: {columns}"
            )
        col_map[col] = name_to_idx[col]

    # We only need to split up to (max_index + 1) fields; everything
    # after that (INFO, FORMAT, samples) is left as a single unsplit
    # tail string, saving significant allocation on wide VCF lines.
    max_idx = max(col_map.values())
    return col_map, max_idx + 1


def _open_vcf(path: str | Path) -> TextIO:
    """Transparently open plain-text or gzip-compressed VCF files."""
    path = Path(path)
    if path.suffix == ".gz":
        return io.TextIOWrapper(gzip.open(path, "rb"), encoding="utf-8")
    return open(path, encoding="utf-8", buffering=8 * 1024 * 1024)  # 8 MB buffer


def parse_vcf(
    path: str | Path,
    batch_size: int = 50_000,
    include_info: bool = False,
) -> Generator[VariantBatch, None, None]:
    """Lazily parse a VCF file and yield batches of variant records.

    Each yielded batch is a list of dicts with keys CHROM, POS, REF, ALT.
    If include_info=True, each record also includes INFO.
    Memory usage is bounded to O(batch_size) — once a batch is yielded the
    references are released and the GC can reclaim the memory.

    Malformed rows are logged and skipped — the generator never crashes on
    bad data, which is critical for user-uploaded files.

    Args:
        path: Filesystem path to the VCF (or VCF.gz) file.
        batch_size: Number of variant records per batch.
        include_info: Include the INFO column in each record.

    Yields:
        A list of up to *batch_size* dicts, each containing the four
        required fields as strings.

    Raises:
        ValueError: If the #CHROM header line is missing or lacks
            required columns.
    """
    required_columns = REQUIRED_COLUMNS + (("INFO",) if include_info else ())

    col_map: dict[str, int] | None = None
    max_split: int = 0
    skipped: int = 0
    line_num: int = 0

    batch: VariantBatch = []

    with _open_vcf(path) as fh:
        for line in fh:
            line_num += 1

            # --- Skip meta-information lines (##) instantly. ---
            # Single char check is faster than startswith for hot path.
            if line[0] == "#":
                if line[1] == "#":
                    # Meta-info line (e.g. ##INFO=<…>). Skip.
                    continue
                # This is the #CHROM header line — parse column positions.
                col_map, max_split = _resolve_column_indices(line, required_columns)
                continue

            if col_map is None:
                raise ValueError(
                    "Encountered data lines before #CHROM header. "
                    "The file does not appear to be a valid VCF."
                )

            # --- Data line — extract only the columns we need. ---
            # Wrapped in try/except so a single corrupted row doesn't
            # kill the entire multi-GB parse.
            try:
                # split('\t', max_split) creates at most (max_split + 1)
                # substrings instead of splitting the entire line which,
                # for lines with a large INFO field, avoids allocating
                # hundreds of kilobytes of throwaway strings.
                fields = line.split("\t", max_split)

                record = {
                    "CHROM": fields[col_map["CHROM"]],
                    "POS": fields[col_map["POS"]],
                    "REF": fields[col_map["REF"]],
                    "ALT": fields[col_map["ALT"]],
                }
                if include_info:
                    record["INFO"] = fields[col_map["INFO"]].rstrip("\n\r")

                batch.append(record)
            except (IndexError, KeyError) as exc:
                skipped += 1
                logger.warning(
                    "Skipping malformed row %d: %s — %s",
                    line_num,
                    line[:120].rstrip(),
                    exc,
                )
                continue

            if len(batch) >= batch_size:
                yield batch
                batch = []  # release references for GC

        # Flush remaining records.
        if batch:
            yield batch

    if skipped:
        logger.warning("Finished parsing: %d malformed rows skipped", skipped)
