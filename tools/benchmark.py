"""Mide la latencia de inferencia en la máquina actual.

Uso:
    python tools/benchmark.py --config config.local.yaml
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.detector import Detector


def main() -> None:
    parser = argparse.ArgumentParser(description="FamilyCentinel inference benchmark")
    parser.add_argument("--config", default="config.local.yaml")
    parser.add_argument("--frames", type=int, default=20, help="Número de frames a medir")
    args = parser.parse_args()

    cfg = load_config(args.config)
    print(f"\nCargando detector desde {cfg.model_path.name} …")
    detector = Detector(cfg.model_path, cfg.labels_path, cfg.detection.min_confidence)

    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    print("Calentando (3 frames)…")
    for _ in range(3):
        detector.detect(frame)

    print(f"Midiendo {args.frames} frames…")
    t0 = time.perf_counter()
    for _ in range(args.frames):
        detector.detect(frame)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    ms_per_frame = elapsed_ms / args.frames

    print(f"\n  Latencia media : {ms_per_frame:.1f} ms/frame")
    print(f"  Throughput     : {1000 / ms_per_frame:.0f} fps")
    print(f"  (Target del sistema: 10 fps — margen: {1000 / ms_per_frame / 10:.1f}×)\n")


if __name__ == "__main__":
    main()
