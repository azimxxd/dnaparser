"""
Integration examples: FastAPI streaming endpoint and multiprocessing pipeline.

Both patterns keep the producer (VCF parsing) and consumer (ML prediction)
decoupled via a bounded queue / async generator, preserving the O(batch_size)
memory guarantee of the parser.
"""

from __future__ import annotations

import json
import multiprocessing as mp
from pathlib import Path
from typing import Any

from vcf_parser import VariantBatch, parse_vcf

# ---------------------------------------------------------------------------
# 1.  FastAPI streaming response
# ---------------------------------------------------------------------------
# Run with:  uvicorn pipeline:app --host 0.0.0.0 --port 8000
#
# The endpoint streams prediction results as newline-delimited JSON (NDJSON)
# so the client can start consuming results before the entire file is parsed.

from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse

app = FastAPI(title="VCF Variant Predictor")

# Replace with your actual model inference function.
def predict_batch(batch: VariantBatch) -> list[dict[str, Any]]:
    """Stub: run ML inference on a batch of variants."""
    return [
        {
            "chrom": rec["CHROM"],
            "pos": rec["POS"],
            "ref": rec["REF"],
            "alt": rec["ALT"],
            "pathogenicity_score": 0.42,  # placeholder
        }
        for rec in batch
    ]


async def _stream_predictions(vcf_path: str, batch_size: int):
    """Async generator that yields NDJSON lines of prediction results."""
    for batch in parse_vcf(vcf_path, batch_size=batch_size):
        results = predict_batch(batch)
        for result in results:
            yield json.dumps(result) + "\n"


@app.get("/predict")
async def predict_variants(
    vcf_path: str = Query(..., description="Server-local path to VCF file"),
    batch_size: int = Query(50_000, ge=1, le=500_000),
):
    """Stream ML predictions for every variant in a VCF file."""
    return StreamingResponse(
        _stream_predictions(vcf_path, batch_size),
        media_type="application/x-ndjson",
    )


# ---------------------------------------------------------------------------
# 2.  Multiprocessing pipeline (producer/consumer via bounded queue)
# ---------------------------------------------------------------------------
# The bounded queue (maxsize) acts as backpressure: if the consumer is slow,
# the producer blocks instead of buffering unlimited batches in memory.

_SENTINEL = None  # signals the consumer that parsing is done


def _producer(vcf_path: str, batch_size: int, queue: mp.Queue) -> None:
    """Read VCF in one process and push batches onto the queue."""
    for batch in parse_vcf(vcf_path, batch_size=batch_size):
        queue.put(batch)
    queue.put(_SENTINEL)


def _consumer(queue: mp.Queue) -> None:
    """Pull batches from the queue and run predictions."""
    while True:
        batch = queue.get()
        if batch is _SENTINEL:
            break
        results = predict_batch(batch)
        # In production: write results to DB, publish to Kafka, etc.
        print(f"Predicted {len(results)} variants")


def run_multiprocess_pipeline(
    vcf_path: str | Path,
    batch_size: int = 50_000,
    queue_maxsize: int = 4,
) -> None:
    """Launch a two-process pipeline: VCF reader -> ML predictor.

    Args:
        vcf_path: Path to the VCF file.
        batch_size: Variants per batch (controls per-batch memory).
        queue_maxsize: Max batches buffered between processes.
            Total peak memory ~ queue_maxsize * batch_size * ~record_size.
    """
    queue: mp.Queue = mp.Queue(maxsize=queue_maxsize)

    producer = mp.Process(target=_producer, args=(str(vcf_path), batch_size, queue))
    consumer = mp.Process(target=_consumer, args=(queue,))

    producer.start()
    consumer.start()

    producer.join()
    consumer.join()


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python pipeline.py <path_to.vcf>")
        sys.exit(1)

    run_multiprocess_pipeline(sys.argv[1])
