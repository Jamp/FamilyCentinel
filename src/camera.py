"""Video capture module — multi-camera, RTSP primary, USB fallback.

Cada instancia de `Camera` gestiona UN stream de vídeo. Para usar varias
cámaras, crear una instancia por cámara y pasarlas todas al `CameraPool`.

Diseño clave:
  - Producer thread independiente por cámara: captura frames continuamente.
  - Queue(maxsize=1) por cámara: siempre se procesa el frame más reciente,
    descartando frames acumulados mientras el TPU estaba ocupado.
  - `shutdown_event` compartido: SIGTERM interrumpe el sleep de reconexión
    inmediatamente, sin esperar los `reconnect_delay_s` segundos.
  - Enmascarado automático de credenciales en logs (password RTSP).

Backend RTSP: CAP_FFMPEG (OpenCV). Es más estable que GStreamer para cámaras
IP domésticas con H.264/MJPEG estándar. No requiere GStreamer instalado.
"""
from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Optional

import cv2
import numpy as np

from src.config import CameraConfig

log = logging.getLogger(__name__)

RTSP_BACKEND = cv2.CAP_FFMPEG

# Tipo de frame con metadatos de cámara para el procesador multi-cámara.
# Usamos una tupla simple para evitar overhead de NamedTuple en el hot path.
# Formato: (camera_name: str, frame: np.ndarray)
FrameItem = tuple[str, np.ndarray]


class Camera:
    """Gestiona un único stream de vídeo (RTSP o USB) con producer thread.

    Uso típico:
        camera = Camera(cfg, shutdown_event)
        camera.open()      # inicia el producer thread
        frame = camera.read()  # obtiene el frame más reciente
        camera.release()   # señaliza parada y espera al thread
    """

    def __init__(
        self,
        cfg: CameraConfig,
        shutdown_event: Optional[threading.Event] = None,
    ) -> None:
        self._cfg = cfg
        # Evento compartido con main.py: cuando se setea, el thread sale limpio.
        self._shutdown: threading.Event = shutdown_event or threading.Event()
        # Cola de un solo slot: descarta frames viejos automáticamente.
        self._frame_queue: queue.Queue[Optional[np.ndarray]] = queue.Queue(maxsize=1)
        self._thread: Optional[threading.Thread] = None

    @property
    def name(self) -> str:
        """Nombre identificador de esta cámara (de CameraConfig.name)."""
        return self._cfg.name

    # ------------------------------------------------------------------
    # Interfaz pública
    # ------------------------------------------------------------------

    def open(self) -> None:
        """Inicia el producer thread. Retorna inmediatamente (no bloqueante)."""
        self._thread = threading.Thread(
            target=self._capture_loop,
            name=f"cam-{self._cfg.name}",
            daemon=True,
        )
        self._thread.start()
        log.info("[%s] Producer thread started (%s)", self._cfg.name, self._source_description())

    def read(self, timeout_s: Optional[float] = None) -> Optional[np.ndarray]:
        """Devuelve el frame más reciente o None si no llega en `timeout_s`.

        El timeout por defecto es 2× el intervalo de frame para no bloquear
        indefinidamente si la cámara tiene problemas de conexión.
        """
        if timeout_s is None:
            timeout_s = (1.0 / max(self._cfg.target_fps, 1)) * 2
        try:
            return self._frame_queue.get(timeout=timeout_s)
        except queue.Empty:
            return None

    def release(self) -> None:
        """Señaliza parada y espera a que el producer thread termine."""
        self._shutdown.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=5.0)
            self._thread = None

    # ------------------------------------------------------------------
    # Producer thread
    # ------------------------------------------------------------------

    def _capture_loop(self) -> None:
        """Bucle de captura que corre en el producer thread.

        Gestiona la reconexión automática de forma transparente. Usa
        `self._shutdown.wait()` para que SIGTERM interrumpa el sleep de
        reconexión sin esperar el intervalo completo.
        """
        cap: Optional[cv2.VideoCapture] = None

        while not self._shutdown.is_set():
            if cap is None or not cap.isOpened():
                cap = self._connect_with_retry()
                if cap is None:
                    break  # shutdown señalizado durante el reintento
                continue

            ret, frame = cap.read()

            if not ret or frame is None:
                log.warning("[%s] Frame read failed — reconnecting", self._cfg.name)
                cap.release()
                cap = None
                continue

            # Reemplazar el frame acumulado con el más reciente.
            # Si la cola está vacía, get_nowait lanza Empty (ignorado).
            try:
                self._frame_queue.get_nowait()
            except queue.Empty:
                pass
            self._frame_queue.put(frame)

        if cap is not None:
            cap.release()
        log.info("[%s] Producer thread exited", self._cfg.name)

    def _connect_with_retry(self) -> Optional[cv2.VideoCapture]:
        """Intenta abrir el dispositivo, reintentando hasta tener éxito o shutdown."""
        while not self._shutdown.is_set():
            log.info("[%s] Connecting to %s…", self._cfg.name, self._source_description())
            cap = self._create_capture()
            if cap is not None and cap.isOpened():
                log.info(
                    "[%s] Connected: %dx%d @ %d fps target",
                    self._cfg.name,
                    self._cfg.width,
                    self._cfg.height,
                    self._cfg.target_fps,
                )
                return cap
            log.warning(
                "[%s] Connection failed — retrying in %.1fs",
                self._cfg.name,
                self._cfg.reconnect_delay_s,
            )
            # Event.wait() permite que SIGTERM interrumpa el sleep inmediatamente.
            self._shutdown.wait(timeout=self._cfg.reconnect_delay_s)

        return None

    # ------------------------------------------------------------------
    # Helpers internos
    # ------------------------------------------------------------------

    def _source_description(self) -> str:
        """Devuelve descripción de la fuente con credenciales enmascaradas."""
        if self._cfg.type == "rtsp":
            url = self._cfg.url
            if "@" in url:
                prefix = url.split("://")[0] + "://"
                rest = url.split("@", 1)[1]
                url = f"{prefix}***@{rest}"
            return f"RTSP {url}"
        return f"USB /dev/video{self._cfg.device_index}"

    def _create_capture(self) -> Optional[cv2.VideoCapture]:
        if self._cfg.type == "rtsp":
            cap = cv2.VideoCapture(self._cfg.url, RTSP_BACKEND)
        else:
            cap = cv2.VideoCapture(self._cfg.device_index)

        if not cap.isOpened():
            return None

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._cfg.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._cfg.height)
        # Buffer de 1 frame: siempre se lee el frame más reciente de la red,
        # no un frame acumulado mientras el TPU estaba ocupado.
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def __enter__(self) -> "Camera":
        self.open()
        return self

    def __exit__(self, *_: object) -> None:
        self.release()


class CameraPool:
    """Gestiona múltiples cámaras y provee frames de todas en una cola unificada.

    El procesador principal consume `(camera_name, frame)` de la cola y aplica
    la inferencia del TPU secuencialmente (el Coral sólo puede procesar una
    petición a la vez).

    El pool respeta la puerta de movimiento: sólo pone frames en la cola de
    salida si la cámara correspondiente tiene movimiento activo.

    Diseño de concurrencia:
      - Cada cámara tiene su propio producer thread (via Camera._capture_loop).
      - Un thread de multiplexado ("pool-mux") lee de todas las cámaras y
        pone los frames activos en `self.output_queue`.
      - El consumidor (main._loop) lee de `output_queue` sin necesidad de
        conocer qué cámara generó el frame.
    """

    def __init__(
        self,
        cameras: list[Camera],
        shutdown_event: threading.Event,
        is_camera_active_fn,  # callable(camera_name: str) -> bool
    ) -> None:
        self._cameras = cameras
        self._shutdown = shutdown_event
        self._is_active = is_camera_active_fn
        # Cola de salida: frames activos de cualquier cámara para el procesador.
        self.output_queue: queue.Queue[FrameItem] = queue.Queue(maxsize=len(cameras))
        self._mux_thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Abre todas las cámaras e inicia el thread de multiplexado."""
        for cam in self._cameras:
            cam.open()
        self._mux_thread = threading.Thread(
            target=self._mux_loop,
            name="pool-mux",
            daemon=True,
        )
        self._mux_thread.start()
        log.info("CameraPool started with %d camera(s)", len(self._cameras))

    def release(self) -> None:
        """Para todas las cámaras y espera al thread de multiplexado."""
        self._shutdown.set()
        for cam in self._cameras:
            cam.release()
        if self._mux_thread and self._mux_thread.is_alive():
            self._mux_thread.join(timeout=5.0)

    def _mux_loop(self) -> None:
        """Thread que round-robins sobre las cámaras activas y distribuye frames."""
        while not self._shutdown.is_set():
            any_active = False

            for cam in self._cameras:
                if not self._is_active(cam.name):
                    continue  # puerta de movimiento cerrada para esta cámara

                any_active = True
                # Timeout breve para no bloquear el round-robin en cámaras lentas.
                frame = cam.read(timeout_s=0.05)
                if frame is None:
                    continue

                try:
                    # Si la cola está llena, descartar el frame más viejo para
                    # no acumular latencia cuando el TPU procesa lento.
                    self.output_queue.get_nowait()
                except queue.Empty:
                    pass
                self.output_queue.put((cam.name, frame))

            if not any_active:
                # Ninguna cámara activa — dormir brevemente para no quemar CPU.
                self._shutdown.wait(timeout=0.2)

        log.info("CameraPool mux thread exited")
