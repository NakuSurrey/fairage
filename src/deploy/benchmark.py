"""
CPU latency benchmark for the exported ONNX models.

runs the model on a single 224x224x3 image many times, records latency
per inference, reports p50 / p95 / p99 percentiles. covers both the
float32 and int8 versions so the README can show a real speedup number.

why percentiles, not averages:
    average latency hides tail latency. p99 is what a user actually
    feels — the worst 1% of requests. for a sub-200ms latency claim,
    p99 is the number that has to stay under 200ms, not the mean.

how the benchmark loop works:
    1. warm up — run 10 inferences and discard the timings. first runs
       are slow because the runtime caches kernels and allocators.
    2. measure — run N inferences, record each one. perf_counter() gives
       sub-microsecond resolution.
    3. summarise — report mean, p50, p95, p99, and stddev.

output:
    artifacts/benchmark_results.json — picked up by the README and the
    bias-report dashboard. proves the latency claim with a real run.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# repo root on sys.path so `src.*` imports work
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.config import ARTIFACTS_DIR, EXPORTS_DIR, IMAGE_SIZE


def benchmark_onnx_model(
    onnx_path: Path,
    num_runs: int = 100,
    num_warmup: int = 10,
    image_size: int = IMAGE_SIZE,
) -> dict:
    """
    measure single-image CPU latency for one ONNX model.

    args:
        onnx_path  — path to a fp32 or int8 ONNX file
        num_runs   — number of timed inferences (more = tighter percentiles)
        num_warmup — number of untimed inferences before measuring
        image_size — square input side length

    returns dict with model path, file size, and latency stats in ms.
    """
    try:
        import onnxruntime as ort
    except ImportError as e:
        raise RuntimeError(
            "onnxruntime is required for the benchmark. "
            "install with: pip install onnxruntime"
        ) from e

    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX model not found at {onnx_path}")

    # CPU-only execution — this is what production will run, no point
    # benchmarking on a GPU we will not have at serving time
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    # use a fixed seed so the input data is identical across fp32 and int8
    # benchmarks — keeps the comparison fair
    rng = np.random.default_rng(0)
    sample = rng.standard_normal((1, 3, image_size, image_size)).astype(np.float32)

    # warmup — discard these timings. first inference compiles internal
    # kernels; first few allocations are slower than steady state.
    for _ in range(num_warmup):
        sess.run(None, {"input": sample})

    # measurement loop
    timings_ms: list[float] = []
    for _ in range(num_runs):
        # perf_counter is monotonic and the highest-resolution clock python has
        start = time.perf_counter()
        sess.run(None, {"input": sample})
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        timings_ms.append(elapsed_ms)

    arr = np.array(timings_ms)

    # file size in MB — the second number on the README badge
    size_mb = onnx_path.stat().st_size / (1024 * 1024)

    return {
        "model_path": str(onnx_path),
        "model_size_mb": round(size_mb, 2),
        "num_runs": num_runs,
        "num_warmup": num_warmup,
        "latency_ms": {
            "mean": round(float(arr.mean()), 2),
            "stddev": round(float(arr.std()), 2),
            "p50": round(float(np.percentile(arr, 50)), 2),
            "p95": round(float(np.percentile(arr, 95)), 2),
            "p99": round(float(np.percentile(arr, 99)), 2),
            "min": round(float(arr.min()), 2),
            "max": round(float(arr.max()), 2),
        },
    }


def run_full_benchmark(
    fp32_path: Path,
    int8_path: Path,
    output_path: Path,
    num_runs: int = 100,
    image_size: int = IMAGE_SIZE,
) -> dict:
    """
    benchmark both fp32 and int8 models, write a single JSON report.

    the JSON report is the file the README links to as proof of the
    latency claim. format is stable so the dashboard can read it directly.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"benchmarking fp32: {fp32_path}")
    fp32_results = benchmark_onnx_model(fp32_path, num_runs=num_runs,
                                        image_size=image_size)
    print(f"  mean {fp32_results['latency_ms']['mean']} ms, "
          f"p99 {fp32_results['latency_ms']['p99']} ms")

    print(f"benchmarking int8: {int8_path}")
    int8_results = benchmark_onnx_model(int8_path, num_runs=num_runs,
                                        image_size=image_size)
    print(f"  mean {int8_results['latency_ms']['mean']} ms, "
          f"p99 {int8_results['latency_ms']['p99']} ms")

    # speedup = how many times faster int8 is than fp32 at the median
    fp32_p50 = fp32_results["latency_ms"]["p50"]
    int8_p50 = int8_results["latency_ms"]["p50"]
    speedup = round(fp32_p50 / int8_p50, 2) if int8_p50 > 0 else 0.0

    # size reduction = same field as the quantize step, recomputed for the report
    size_reduction_pct = round(
        (1 - int8_results["model_size_mb"] / fp32_results["model_size_mb"]) * 100,
        1,
    ) if fp32_results["model_size_mb"] > 0 else 0.0

    report = {
        "fp32": fp32_results,
        "int8": int8_results,
        "summary": {
            "speedup_vs_fp32": speedup,
            "size_reduction_pct": size_reduction_pct,
            "int8_p99_ms": int8_results["latency_ms"]["p99"],
            "latency_target_ms": 200,
            "meets_latency_target": int8_results["latency_ms"]["p99"] < 200,
        },
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\nreport written to {output_path}")
    print(f"int8 speedup: {speedup}x faster than fp32")
    print(f"int8 p99: {int8_results['latency_ms']['p99']} ms "
          f"(target: 200 ms, "
          f"{'PASS' if report['summary']['meets_latency_target'] else 'FAIL'})")

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fp32",
        type=str,
        default=str(EXPORTS_DIR / "age_model.onnx"),
    )
    parser.add_argument(
        "--int8",
        type=str,
        default=str(EXPORTS_DIR / "age_model_int8.onnx"),
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(ARTIFACTS_DIR / "benchmark_results.json"),
    )
    parser.add_argument("--num-runs", type=int, default=100)
    args = parser.parse_args()

    run_full_benchmark(
        fp32_path=Path(args.fp32),
        int8_path=Path(args.int8),
        output_path=Path(args.output),
        num_runs=args.num_runs,
    )


if __name__ == "__main__":
    main()
