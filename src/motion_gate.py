"""Filtro de movimiento de píxeles para eliminar falsos positivos estáticos.

Problema que resuelve:
    YOLOv8 detecta ropa en una silla / suéter en un sofá como "persona" porque
    la silueta se parece a un humano. Pero la ropa no se mueve.

Solución:
    Para cada detección, comparar los píxeles del bbox con el frame anterior.
    Si no hay cambio de píxeles → objeto estático → ignorar.
    Si hay cambio de píxeles → algo se movió → aceptar.

Comportamiento "one-way" para personas ya presentes:
    Una vez que el estabilizador ha marcado a una entidad como PRESENTE, el
    filtro se desactiva para esa entidad. Esto evita que una persona parada
    quieta durante unos segundos sea marcada como ausente.

    Ropa en silla: nunca tiene movimiento → nunca pasa el filtro → nunca
    se marca como presente → nunca molesta.
"""
from __future__ import annotations

import logging

import cv2
import numpy as np

log = logging.getLogger(__name__)

# Píxeles con diferencia > este valor se consideran "en movimiento".
_PIXEL_DIFF_THRESHOLD = 12
# Fracción mínima del bbox que debe estar en movimiento para aceptar la detección.
# Persona sentada quieta ≈ 0.5-1%  |  Ropa estática ≈ 0.05-0.2%
# Usar 0.3% como separador — el estabilizador (frames_to_activate=2) filtra el ruido.
_MIN_MOTION_FRACTION = 0.003  # 0.3%


class MotionGate:
    """Filtra detecciones sin movimiento de píxeles en su bounding box."""

    def __init__(
        self,
        pixel_threshold: int = _PIXEL_DIFF_THRESHOLD,
        motion_fraction: float = _MIN_MOTION_FRACTION,
    ) -> None:
        self._pixel_threshold = pixel_threshold
        self._motion_fraction = motion_fraction
        # Último frame por cámara (gris, para comparación eficiente)
        self._prev_gray: dict[str, np.ndarray] = {}

    def has_motion(
        self,
        camera_name: str,
        curr_gray: np.ndarray,
        bbox: tuple[float, float, float, float],
    ) -> bool:
        """True si hay movimiento de píxeles en el área del bbox.

        Devuelve True cuando no hay frame de referencia (primer frame tras
        arranque) para no bloquear detecciones legítimas al inicio.

        El frame se recibe ya en escala de grises porque el caller convierte
        BGR→GRAY una sola vez por frame y reutiliza el resultado para todas
        las detecciones — convertir N+1 veces costaba más que la propia
        comparación de píxeles.
        """
        prev_gray = self._prev_gray.get(camera_name)
        if prev_gray is None:
            return True

        h, w = curr_gray.shape[:2]
        ymin, xmin, ymax, xmax = bbox
        y1, y2 = max(0, int(ymin * h)), min(h, int(ymax * h))
        x1, x2 = max(0, int(xmin * w)), min(w, int(xmax * w))

        if (y2 - y1) < 4 or (x2 - x1) < 4:
            return True  # bbox demasiado pequeño para medir movimiento

        diff = cv2.absdiff(curr_gray[y1:y2, x1:x2], prev_gray[y1:y2, x1:x2])
        moving_px = int(np.sum(diff > self._pixel_threshold))
        total_px  = diff.size
        fraction  = moving_px / total_px

        log.debug(
            "[%s] motion in bbox: %.1f%% (%d/%d px)",
            camera_name, fraction * 100, moving_px, total_px,
        )
        return fraction >= self._motion_fraction

    def update(self, camera_name: str, frame_gray: np.ndarray) -> None:
        """Guardar el frame gris actual como referencia para el siguiente ciclo."""
        self._prev_gray[camera_name] = frame_gray
