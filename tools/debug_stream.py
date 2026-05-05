"""Servidor MJPEG de depuración con detecciones superpuestas en tiempo real.

Conecta a las cámaras configuradas, ejecuta la inferencia (con fallback a CPU
si el Coral no está disponible) y sirve el vídeo anotado vía HTTP/MJPEG en
el puerto 8080. Se puede abrir en cualquier navegador de la red local.

USO:
    # Desde el host (con dependencias instaladas):
    python tools/debug_stream.py --config config.yaml --port 8080

    # Dentro del contenedor (requiere exponer el puerto en docker-compose.yml):
    docker compose exec familycentinel python tools/debug_stream.py \
        --config /app/config/config.yaml --port 8080

    # Abrir en el navegador:
    http://<IP_HOST>:8080

CONTROLES (en el navegador):
    - La página muestra un stream por cámara detectada activa.
    - Las zonas de calibración (bandas) se muestran en gris claro.
    - Los bboxes se colorean según la entidad detectada:
        Verde → persona
        Azul  → perro
    - El estado del trigger de movimiento se muestra en la esquina superior.

NOTA: Esta herramienta usa CPU para la inferencia por defecto (no bloquea el
Coral si el servicio principal está corriendo). Pasar --use-tpu para forzar
el Edge TPU (sólo si el servicio principal está parado).
"""
from __future__ import annotations

import argparse
import io
import logging
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.camera import Camera
from src.config import AppConfig, load_config
from src.detector import Detector
from src.motion_gate import MotionGate
from src.onvif_trigger import OnvifTrigger

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ---------------------------------------------------------------------------
# Colores BGR por categoría
# ---------------------------------------------------------------------------
_COLORS = {
    "person":  (0, 200, 0),      # verde
    "dog":     (255, 100, 0),    # azul
    "unknown": (180, 180, 180),  # gris
}
_TEXT_BG = (20, 20, 20)


# ---------------------------------------------------------------------------
# Estado compartido entre threads
# ---------------------------------------------------------------------------
class StreamState:
    """Frame anotado más reciente por cámara, acceso thread-safe."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # camera_name → JPEG bytes del frame anotado más reciente
        self._frames: dict[str, bytes] = {}
        self._stats: dict[str, str] = {}

    def update(self, camera_name: str, frame: np.ndarray, stats: str) -> None:
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if ok:
            with self._lock:
                self._frames[camera_name] = buf.tobytes()
                self._stats[camera_name] = stats

    def get_frame(self, camera_name: str) -> Optional[bytes]:
        with self._lock:
            return self._frames.get(camera_name)

    def camera_names(self) -> list[str]:
        with self._lock:
            return list(self._frames.keys())


_state = StreamState()
_motion_gate = MotionGate()


# ---------------------------------------------------------------------------
# Servidor HTTP / MJPEG
# ---------------------------------------------------------------------------
_HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>FamilyCentinel Debug</title>
  <style>
    body {{ background: #111; color: #eee; font-family: monospace; margin: 0; padding: 16px; }}
    h1   {{ font-size: 1.1em; color: #4af; margin: 0 0 12px; }}
    .grid {{ display: flex; flex-wrap: wrap; gap: 12px; }}
    .cam  {{ background: #222; border: 1px solid #444; border-radius: 6px; padding: 8px; }}
    .cam h2 {{ font-size: 0.85em; color: #aaa; margin: 0 0 6px; }}
    img  {{ display: block; max-width: 100%; }}
    .legend {{ font-size: 0.75em; margin-top: 12px; color: #888; }}
    .legend span {{ margin-right: 12px; }}
    .person {{ color: #0c8; }} .dog {{ color: #68f; }} .quiet {{ color: #cc0; }}
  </style>
</head>
<body>
  <h1>FamilyCentinel — Debug Stream</h1>
  <div class="grid">
    {camera_tiles}
  </div>
  <div class="legend">
    <span class="person">■ Persona (activa)</span>
    <span class="dog">■ Perro (activo)</span>
    <span class="quiet">■ Quieto (modelo ve, producción filtra)</span>
  </div>
  <script>
    document.querySelectorAll('img[data-cam]').forEach(img => {{
      setInterval(() => {{
        img.src = '/stream/' + img.dataset.cam + '?' + Date.now();
      }}, 100);
    }});
  </script>
</body>
</html>
"""

_CAM_TILE = """
    <div class="cam">
      <h2>{name}</h2>
      <img data-cam="{name}" src="/stream/{name}" width="640">
    </div>
"""

_BOUNDARY = b"--frameboundary"


class DebugHandler(BaseHTTPRequestHandler):
    """Handler HTTP mínimo para MJPEG y página de estado."""

    def log_message(self, *args: object) -> None:
        pass

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?")[0]
        if path == "/" or path == "/index.html":
            self._serve_index()
        elif path.startswith("/stream/"):
            self._serve_jpeg(path[len("/stream/"):])
        else:
            self.send_response(404)
            self.end_headers()

    def _serve_index(self) -> None:
        names = _state.camera_names()
        tiles = "".join(_CAM_TILE.format(name=n) for n in names) if names else "<p>Esperando frames...</p>"
        html = _HTML_TEMPLATE.format(camera_tiles=tiles).encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(html)))
        self.end_headers()
        self.wfile.write(html)

    def _serve_jpeg(self, camera_name: str) -> None:
        frame_bytes = _state.get_frame(camera_name)
        if frame_bytes is None:
            self.send_response(404)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(frame_bytes)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(frame_bytes)


# ---------------------------------------------------------------------------
# Anotación de frames
# ---------------------------------------------------------------------------

def _draw_exclusion_zones(frame: np.ndarray, zones: list) -> None:
    if not zones:
        return
    h, w = frame.shape[:2]
    overlay = frame.copy()
    for z in zones:
        x1, y1 = int(z[0] * w), int(z[1] * h)
        x2, y2 = int(z[2] * w), int(z[3] * h)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 180), -1)
        cv2.rectangle(frame,   (x1, y1), (x2, y2), (0, 0, 220), 2)
        cv2.putText(frame, "ZONA EXCLUIDA", (x1+4, y1+16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 220), 1)
    cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)


def _annotate_detections(
    frame: np.ndarray,
    detections,
    cfg: AppConfig,
    cam_name: str = "",
) -> tuple[set[str], str]:
    h, w = frame.shape[:2]
    entities: set[str] = set()
    stats_lines: list[str] = []

    for det in detections:
        if cfg.detection.is_excluded(cam_name, det.bbox):
            continue

        if det.label == "person":
            label = "person"
            display = "person"
        elif det.label in ("dog", "cat"):
            label = "dog"
            display = det.label
        else:
            continue

        has_mov = _motion_gate.has_motion(cam_name, frame, det.bbox)

        ymin, xmin, ymax, xmax = det.bbox
        x1, y1 = int(xmin * w), int(ymin * h)
        x2, y2 = int(xmax * w), int(ymax * h)

        if has_mov:
            # Detección activa: color sólido — produciría inferencia en producción
            color = _COLORS.get(label, _COLORS["unknown"])
            thickness = 2
            entities.add(label)
            stats_lines.append(f"{label}({det.score:.0%})")
        else:
            # Detección sin movimiento: bbox punteado amarillo — modelo ve la entidad
            # pero el motion_gate la filtraría en producción (persona quieta)
            color = (0, 200, 200)   # amarillo
            thickness = 1

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        suffix = "" if has_mov else " ~quieto"
        text = f"{display} {det.score:.0%}{suffix}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), _TEXT_BG, -1)
        cv2.putText(
            frame, text, (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA,
        )

    return entities, " | ".join(stats_lines) or "—"


def _draw_overlay(
    frame: np.ndarray,
    camera_name: str,
    fps: float,
    stats: str,
    motion_active: bool,
) -> None:
    """Dibuja barra de estado e indicador de movimiento ONVIF."""
    ts = time.strftime("%H:%M:%S")
    lines = [
        f"{camera_name}  {ts}  {fps:.1f}fps",
        stats if stats != "—" else "sin detecciones",
    ]
    for i, line in enumerate(lines):
        y = 18 + i * 18
        cv2.putText(
            frame, line, (4, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA,
        )

    # Indicador ONVIF (esquina superior derecha)
    h, w = frame.shape[:2]
    label     = "ONVIF: MOVIMIENTO"   if motion_active else "ONVIF: sin movimiento"
    color     = (0, 60, 255)          if motion_active else (80, 80, 80)
    dot_color = (0, 80, 255)          if motion_active else (60, 60, 60)

    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    x = w - tw - 22
    y = 18
    overlay = frame.copy()
    cv2.rectangle(overlay, (x - 6, y - th - 4), (w - 4, y + 4), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    cv2.circle(frame, (x - 10, y - th // 2), 5, dot_color, -1, cv2.LINE_AA)
    cv2.putText(frame, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    # Borde rojo en todo el frame cuando el TPU estaría activo
    if motion_active:
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 60, 255), 3)


# ---------------------------------------------------------------------------
# Producer thread por cámara
# ---------------------------------------------------------------------------

def _camera_thread(
    cam: Camera,
    detector: Detector,
    detector_lock: threading.Lock,
    cfg: AppConfig,
    target_fps: int,
    onvif_trigger: OnvifTrigger,
) -> None:
    cam.open()
    frame_interval = 1.0 / max(target_fps, 1)
    last_time = time.monotonic()
    fps_counter = 0
    fps_display = 0.0
    fps_window_start = time.monotonic()

    try:
        while True:
            frame = cam.read()
            if frame is None:
                continue

            now = time.monotonic()
            elapsed = now - last_time
            if elapsed < frame_interval:
                time.sleep(frame_interval - elapsed)
            last_time = time.monotonic()

            fps_counter += 1
            if (time.monotonic() - fps_window_start) >= 2.0:
                fps_display = fps_counter / (time.monotonic() - fps_window_start)
                fps_counter = 0
                fps_window_start = time.monotonic()

            try:
                with detector_lock:
                    detections = detector.detect(frame)
                annotated = frame.copy()
                _draw_exclusion_zones(
                    annotated,
                    cfg.detection.exclusion_zones.get(cam.name, []),
                )
                entities, stats = _annotate_detections(annotated, detections, cfg, cam.name)
                motion_active = onvif_trigger.is_camera_active(cam.name)
                _draw_overlay(annotated, cam.name, fps_display, stats, motion_active)
                _state.update(cam.name, annotated, stats)
                _motion_gate.update(cam.name, frame)
            except Exception as exc:
                log.warning("[%s] Inference error: %s", cam.name, exc)

    finally:
        cam.release()


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="FamilyCentinel debug MJPEG stream")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument(
        "--use-tpu", action="store_true",
        help="Usar Edge TPU (sólo si el servicio principal está parado)",
    )
    parser.add_argument(
        "--fps", type=int, default=5,
        help="FPS objetivo del stream de debug (default: 5, no saturar el TPU)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    onvif_trigger = OnvifTrigger(cfg.onvif_trigger, cfg.cameras)
    onvif_trigger.start()

    log.info("Loading detector (use_tpu=%s)…", args.use_tpu)
    detector = Detector(
        model_path=cfg.model_path,
        labels_path=cfg.labels_path,
        min_confidence=cfg.detection.min_confidence,
        class_min_confidence=cfg.detection.class_min_confidence,
        use_tpu=args.use_tpu,
    )

    detector_lock = threading.Lock()

    shutdown = threading.Event()
    threads: list[threading.Thread] = []
    for cam_cfg in cfg.cameras:
        cam = Camera(cam_cfg, shutdown_event=shutdown)
        t = threading.Thread(
            target=_camera_thread,
            args=(cam, detector, detector_lock, cfg, args.fps, onvif_trigger),
            name=f"debug-{cam_cfg.name}",
            daemon=True,
        )
        threads.append(t)
        t.start()
        log.info("Camera '%s' debug thread started", cam_cfg.name)

    # Esperar a tener al menos un frame antes de abrir el servidor
    log.info("Waiting for first frames…")
    deadline = time.time() + 15
    while not _state.camera_names() and time.time() < deadline:
        time.sleep(0.2)

    server = HTTPServer(("0.0.0.0", args.port), DebugHandler)
    log.info("Debug stream available at: http://0.0.0.0:%d", args.port)
    log.info("Open in browser:           http://<HOST_IP>:%d", args.port)
    log.info("Press Ctrl+C to stop.")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("Stopping…")
    finally:
        shutdown.set()
        server.server_close()


if __name__ == "__main__":
    main()
