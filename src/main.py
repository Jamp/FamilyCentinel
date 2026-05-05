"""FamilyCentinel — entrypoint y loop principal de procesamiento.

FLUJO GENERAL:
    1. Cargar configuración (config.yaml + variables de entorno).
    2. Conectar al broker MQTT y publicar MQTT Discovery para Home Assistant.
    3. Abrir todas las cámaras configuradas (producer threads RTSP independientes).
    4. Si onvif_trigger está activo: lanzar suscripciones PullPoint por cámara
       en un asyncio loop dedicado (daemon thread).
    5. Loop principal:
         - Si ninguna cámara tiene movimiento activo → esperar (TPU inactivo).
         - Si hay movimiento → leer frame de la cámara activa → inferencia TPU
           → debounce temporal → publicar cambios MQTT.

MULTI-CÁMARA:
    FamilyCentinel gestiona múltiples streams RTSP simultáneamente usando
    `CameraPool`. Cada cámara corre en su propio producer thread. El Coral
    Edge TPU procesa los frames secuencialmente (no en paralelo) ya que el
    hardware sólo admite una inferencia a la vez. El pool prioriza el frame
    más reciente de cualquier cámara activa.

SEGURIDAD:
    - `SecretMaskingFilter`: enmascara credenciales RTSP en TODOS los logs,
      incluidos los de librerías externas (FFmpeg, paho-mqtt).
    - `os.umask(0o027)`: archivos creados por el proceso (healthcheck) son
      640/750, no 644/755.
    - Las credenciales MQTT se pasan via variables de entorno, nunca se
      imprimen en logs (ver config.py `__repr__` enmascarados).

USO:
    python -m src.main [--config /app/config/config.yaml]
    CONFIG_PATH=/path/to/config.yaml python -m src.main
"""
from __future__ import annotations

import argparse
import logging
import os
import re
import signal
import sys
import threading
import time
from pathlib import Path

from src.camera import Camera, CameraPool
from src.config import AppConfig, load_config
from src.detector import Detector
from src.motion_gate import MotionGate
from src.mqtt_client import MqttClient
from src.onvif_trigger import OnvifTrigger
from src.stabilizer import Stabilizer

log = logging.getLogger(__name__)

_DEFAULT_CONFIG = "/app/config/config.yaml"
_HEALTHCHECK_FILE = "/tmp/healthcheck"
_HEARTBEAT_INTERVAL_S = 30
_IDLE_LOG_INTERVAL_S = 60.0  # frecuencia máxima del log "TPU inactivo"


# ---------------------------------------------------------------------------
# Filtro de enmascarado de secretos en logs
# ---------------------------------------------------------------------------
_RTSP_CRED_RE = re.compile(r"(rtsps?://)([^:@/\s]+):([^@/\s]+)@")


class SecretMaskingFilter(logging.Filter):
    """Reemplaza credenciales RTSP embebidas por '***' en TODOS los logs.

    Se instala en el logger raíz para cubrir también FFmpeg, OpenCV y paho,
    que pueden imprimir la URL RTSP completa en sus mensajes de error.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            msg = record.getMessage()
        except Exception:
            return True
        if "rtsp" in msg.lower() and "@" in msg:
            masked = _RTSP_CRED_RE.sub(r"\1\2:***@", msg)
            if masked != msg:
                record.msg = masked
                record.args = ()
        return True


def _setup_logging() -> None:
    """Configura logging a stdout con enmascarado automático de credenciales."""
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
    )
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(logging.INFO)
    # Filtro instalado tanto en el logger raíz como en cada handler por si
    # alguna librería añade sus propios handlers.
    mask = SecretMaskingFilter()
    root.addFilter(mask)
    for h in root.handlers:
        h.addFilter(mask)


def _write_healthcheck() -> None:
    """Actualiza el timestamp de /tmp/healthcheck para el Docker HEALTHCHECK."""
    Path(_HEALTHCHECK_FILE).touch()


# ---------------------------------------------------------------------------
# Clase principal
# ---------------------------------------------------------------------------

class FamilyCentinel:
    """Orquestador principal: cámaras → TPU → MQTT → Home Assistant."""

    def __init__(self, cfg: AppConfig) -> None:
        self._cfg = cfg

        # Evento compartido con Camera/CameraPool para apagado limpio.
        # Cuando se setea, todos los producer threads salen sin esperar sleeps.
        self._shutdown = threading.Event()

        # Crear una instancia Camera por cada entrada en `cameras:`.
        cameras = [
            Camera(cam_cfg, shutdown_event=self._shutdown)
            for cam_cfg in cfg.cameras
        ]

        # Puerta de movimiento ONVIF (PullPoint).
        self._onvif_trigger = OnvifTrigger(cfg.onvif_trigger, cfg.cameras)

        # Pool multi-cámara: distribuye frames activos a la cola de salida.
        self._pool = CameraPool(
            cameras=cameras,
            shutdown_event=self._shutdown,
            is_camera_active_fn=self._onvif_trigger.is_camera_active,
        )

        # Inferencia con Edge TPU — un único detector compartido por todas las cámaras.
        self._detector = Detector(
            model_path=cfg.model_path,
            labels_path=cfg.labels_path,
            min_confidence=cfg.detection.min_confidence,
            class_min_confidence=cfg.detection.class_min_confidence,
        )

        # Debounce temporal — estado agregado de todas las cámaras.
        self._stabilizer = Stabilizer(
            frames_to_activate=cfg.stabilization.frames_to_activate,
            frames_to_deactivate=cfg.stabilization.frames_to_deactivate,
            min_presence_seconds=cfg.stabilization.min_presence_seconds,
        )

        # Cliente MQTT: publish (Discovery + estados) + subscribe (motion topics).
        self._mqtt = MqttClient(cfg.mqtt, on_connect_cb=self._on_mqtt_connect)
        self._motion_gate = MotionGate()

        self._last_heartbeat: float = 0.0
        self._last_idle_log: float = 0.0
        self._running = False

    # ------------------------------------------------------------------
    # Ciclo de vida
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Conecta MQTT, abre cámaras, lanza ONVIF y entra en el loop principal."""
        self._running = True
        self._mqtt.connect()

        log.info("Waiting for MQTT connection…")
        if not self._mqtt.wait_connected(timeout=30.0):
            log.error("Could not connect to MQTT broker — exiting")
            sys.exit(1)

        self._onvif_trigger.start()
        self._pool.start()
        _write_healthcheck()
        log.info(
            "FamilyCentinel started — %d camera(s), onvif_trigger=%s",
            len(self._cfg.cameras),
            self._cfg.onvif_trigger.enabled,
        )
        self._loop()

    def stop(self, *_: object) -> None:
        """Señaliza apagado (llamado por SIGTERM/SIGINT).

        El shutdown_event desbloquea inmediatamente los `Event.wait()` dentro
        de los producer threads y del mux de CameraPool, garantizando que
        `docker stop` termina en < 5s sin necesitar SIGKILL.
        """
        log.info("Shutdown requested…")
        self._running = False
        self._shutdown.set()
        self._onvif_trigger.stop()

    # ------------------------------------------------------------------
    # Loop principal
    # ------------------------------------------------------------------

    def _loop(self) -> None:
        """Lee frames del pool y ejecuta el pipeline completo."""
        consecutive_tpu_errors = 0
        max_tpu_errors = 10

        while self._running:
            # ----------------------------------------------------------
            # Puerta de movimiento global: si ninguna cámara está activa,
            # el pool no produce frames y la cola de salida está vacía.
            # Dormimos para no quemar CPU y seguimos actualizando el heartbeat.
            # ----------------------------------------------------------
            if not self._onvif_trigger.is_active:
                self._log_idle_once()
                self._heartbeat()
                # Event.wait() permite que SIGTERM nos interrumpa aquí también.
                self._shutdown.wait(timeout=0.5)
                continue

            # Leer el frame más reciente de cualquier cámara activa.
            try:
                camera_name, frame = self._pool.output_queue.get(timeout=1.0)
            except Exception:
                # queue.Empty: ningún frame disponible — volver a comprobar.
                continue

            try:
                detected_entities = self._process_frame(camera_name, frame)
                state_changes = self._stabilizer.update(detected_entities)

                for entity, is_present in state_changes.items():
                    self._mqtt.publish_state(entity, is_present)

                consecutive_tpu_errors = 0

            except Exception:
                consecutive_tpu_errors += 1
                log.exception(
                    "TPU/inference error on camera '%s' (%d/%d)",
                    camera_name,
                    consecutive_tpu_errors,
                    max_tpu_errors,
                )
                if consecutive_tpu_errors >= max_tpu_errors:
                    log.error(
                        "Too many consecutive TPU errors — exiting. "
                        "Check that the Coral USB-C is connected and libedgetpu is loaded."
                    )
                    break

            self._heartbeat()

        self._shutdown_gracefully()

    # ------------------------------------------------------------------
    # Procesamiento de frame
    # ------------------------------------------------------------------

    def _process_frame(self, camera_name: str, frame) -> set[str]:
        detections = self._detector.detect(frame)
        entities: set[str] = set()
        target_classes = set(self._cfg.detection.classes)
        current_states = self._stabilizer.current_states()

        for det in detections:
            if det.label not in target_classes:
                continue
            if self._cfg.detection.is_excluded(camera_name, det.bbox):
                continue

            if det.label in ("dog", "cat"):
                entity = "dog"
            elif det.label == "person":
                entity = "person"
            else:
                continue

            # Filtro de movimiento: solo se aplica cuando la entidad está AUSENTE.
            # Si ya está marcada como presente (persona parada quieta), se omite
            # el filtro para no interrumpir presencias confirmadas.
            if not current_states.get(entity, False):
                if not self._motion_gate.has_motion(camera_name, frame, det.bbox):
                    log.debug("[%s] %s ignorado — sin movimiento en bbox", camera_name, entity)
                    continue

            entities.add(entity)

        self._motion_gate.update(camera_name, frame)

        # Si el TPU detectó algo, extender la ventana activa de esta cámara.
        # Así el sistema no entra en reposo mientras haya detecciones, aunque
        # ONVIF haya dejado de emitir eventos (entidad quieta pero presente).
        if entities:
            self._onvif_trigger.notify_detection(camera_name)

        return entities

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _heartbeat(self) -> None:
        """Actualiza el archivo que usa el Docker HEALTHCHECK."""
        now = time.monotonic()
        if now - self._last_heartbeat >= _HEARTBEAT_INTERVAL_S:
            _write_healthcheck()
            self._last_heartbeat = now

    def _log_idle_once(self) -> None:
        """Emite un log informativo máx. una vez por minuto mientras está inactivo."""
        now = time.monotonic()
        if now - self._last_idle_log >= _IDLE_LOG_INTERVAL_S:
            secs = self._onvif_trigger.seconds_since_last_global_event
            log.info(
                "ONVIF trigger active — TPU idle (%.0fs since last motion event)",
                secs,
            )
            self._last_idle_log = now

    def _on_mqtt_connect(self) -> None:
        """Llamado por MqttClient al (re)conectarse al broker.

        Re-publica Discovery y estados actuales para que HA no muestre
        "unknown" tras una reconexión del broker (clean_session=True).
        """
        self._mqtt.publish_discovery()

        for entity, present in self._stabilizer.current_states().items():
            self._mqtt.publish_state(entity, present)

    def _shutdown_gracefully(self) -> None:
        """Libera recursos en orden seguro."""
        log.info("Releasing resources…")
        self._pool.release()
        self._mqtt.disconnect()
        log.info("FamilyCentinel stopped.")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    _setup_logging()

    # Restricción de permisos: archivos creados por este proceso serán 640/750.
    os.umask(0o027)

    parser = argparse.ArgumentParser(description="FamilyCentinel presence detector")
    parser.add_argument(
        "--config",
        default=os.environ.get("CONFIG_PATH", _DEFAULT_CONFIG),
        help="Path to config.yaml (default: /app/config/config.yaml)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    app = FamilyCentinel(cfg)

    # Registrar handlers de señal antes de start() para que stop() sea
    # llamado incluso si SIGTERM llega durante la conexión MQTT inicial.
    signal.signal(signal.SIGTERM, app.stop)
    signal.signal(signal.SIGINT, app.stop)

    app.start()


if __name__ == "__main__":
    main()
