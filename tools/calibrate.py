"""Snapshot detector — captura un frame y anota las detecciones.

Útil para:
  - Verificar que el modelo detecta correctamente en la cámara configurada.
  - Ajustar `min_confidence` viendo los scores reales.
  - Identificar coordenadas para `exclusion_zones` en config.yaml.

Uso:
    python tools/calibrate.py [--config config.yaml] [--output frame.jpg]

Pasos:
    1. Asegúrate de tener las dependencias instaladas (pip install -r requirements.txt).
    2. Ejecuta el script apuntando a tu config.yaml.
    3. Examina la imagen generada y los valores impresos.
    4. Ajusta `detection.min_confidence` o añade `exclusion_zones` si hay
       detecciones falsas en zonas concretas de la imagen.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.camera import Camera
from src.config import load_config
from src.detector import Detector

log = logging.getLogger(__name__)

_DEFAULT_CONFIG = "config.yaml"

_COLORS = {
    "person": (0, 200, 0),
    "dog":    (255, 100, 0),
    "cat":    (0, 165, 255),
}
_COLOR_DEFAULT = (180, 180, 180)


def _annotate(frame: np.ndarray, detections) -> None:
    h, w = frame.shape[:2]
    for det in detections:
        ymin, xmin, ymax, xmax = det.bbox
        x1, y1 = int(xmin * w), int(ymin * h)
        x2, y2 = int(xmax * w), int(ymax * h)
        bbox_h = y2 - y1
        bbox_w = x2 - x1

        color = _COLORS.get(det.label, _COLOR_DEFAULT)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Centro normalizado (útil para exclusion_zones)
        cx = (xmin + xmax) / 2
        cy = (ymin + ymax) / 2

        lines = [
            f"{det.label} {det.score:.0%}",
            f"h={bbox_h}px  w={bbox_w}px",
            f"center=({cx:.2f}, {cy:.2f})",
        ]
        for idx, text in enumerate(lines):
            ty = y1 + 16 + idx * 17
            cv2.putText(
                frame, text, (x1 + 4, ty),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA,
            )

        print(
            f"  {det.label:8s} | score={det.score:.0%} | "
            f"h={bbox_h}px w={bbox_w}px | "
            f"center=({cx:.2f},{cy:.2f}) | "
            f"bbox=[{xmin:.2f},{ymin:.2f},{xmax:.2f},{ymax:.2f}]"
        )


def run(cfg, output_path: str) -> None:
    print("Connecting to camera…")
    camera = Camera(cfg.cameras[0])
    camera.open()

    print("Capturing frame…")
    frame = None
    for _ in range(5):  # descartar frames en buffer
        frame = camera.read()
    camera.release()

    if frame is None:
        print("ERROR: no se pudo capturar frame.")
        sys.exit(1)

    print(f"Frame: {frame.shape[1]}x{frame.shape[0]}px")
    print("Running inference…")

    detector = Detector(
        model_path=cfg.model_path,
        labels_path=cfg.labels_path,
        min_confidence=0.3,  # umbral bajo para ver todos los candidatos
    )

    detections = detector.detect(frame)
    relevant = [d for d in detections if d.label in ("person", "dog", "cat")]

    print(f"\n{'='*60}")
    print(f"Detected {len(relevant)} object(s) (min_confidence=0.3):")
    annotated = frame.copy()
    _annotate(annotated, relevant)
    print(f"{'='*60}")
    print(
        f"\nTip: para añadir zonas de exclusión usa el valor 'center' de cada"
        f" detección falsa en config.yaml bajo detection.exclusion_zones."
    )

    cv2.imwrite(output_path, annotated)
    print(f"Annotated frame saved to: {output_path}")


def main() -> None:
    logging.basicConfig(level=logging.WARNING, stream=sys.stdout)

    parser = argparse.ArgumentParser(description="FamilyCentinel snapshot detector")
    parser.add_argument("--config", default=_DEFAULT_CONFIG)
    parser.add_argument("--output", default="calibration_frame.jpg")
    args = parser.parse_args()

    cfg = load_config(args.config)
    run(cfg, args.output)


if __name__ == "__main__":
    main()
