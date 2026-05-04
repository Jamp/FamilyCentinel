"""Motion trigger gate driven by Thingino camera MQTT events.

Thingino firmware (open-source, themactep/thingino-firmware) publica un
mensaje MQTT cuando su Motion Guard detecta movimiento. FamilyCentinel se
suscribe a esos topics y usa los eventos como puerta: el Edge TPU sólo
procesa frames de una cámara si ha tenido movimiento recientemente.

FORMATO DE TOPICS (configurable en Thingino web UI):
    thingino/<nombre_camara>         ← topic por defecto de Thingino

PAYLOAD (cualquier formato se acepta — sólo importa que haya mensaje):
    "motion"                         ← string simple
    {"camera_id": "abc", "ts": "…"}  ← JSON (versiones recientes de Thingino)

CONFIGURACIÓN EN THINGINO (una vez por cámara):
    1. Abre http://<IP_CAMARA>
    2. Tools → Motion Guard → activar
    3. Marcar "Send to MQTT"
    4. Rellenar host/puerto/usuario/contraseña del broker MQTT
    5. El topic lo configuras tú (recomendado: thingino/<nombre>)
    6. Pon ese topic en `cameras[n].motion_topic` en config.yaml

TRACKING POR CÁMARA:
    Cada cámara tiene su propio timestamp de último evento. Esto permite:
    - Activar sólo la cámara que vio el movimiento (más eficiente).
    - Mantener activas simultáneamente varias cámaras con movimiento propio.
    - El global `is_active` es True si CUALQUIER cámara tiene movimiento.
"""
from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict

from src.config import MotionTriggerConfig

log = logging.getLogger(__name__)

# Sentinel para el inicio: tiempo 0.0 significa "nunca visto movimiento".
_NEVER: float = 0.0


class MotionTrigger:
    """Puerta de movimiento con tracking por cámara.

    Thread-safe: `on_mqtt_message` es llamado desde el thread de red de paho
    y `is_camera_active` desde el thread principal del procesador.
    """

    def __init__(self, cfg: MotionTriggerConfig) -> None:
        self._cfg = cfg
        self._lock = threading.Lock()
        # Diccionario: topic_or_camera_name → monotonic timestamp del último evento.
        # Usamos defaultdict para no necesitar inicialización explícita.
        self._last_event: dict[str, float] = defaultdict(lambda: _NEVER)
        self._total_events: int = 0

        # Mapa inverso: topic MQTT → nombre de cámara.
        # Se construye cuando se registran las cámaras.
        self._topic_to_camera: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Registro de cámaras
    # ------------------------------------------------------------------

    def register_camera(self, camera_name: str, motion_topic: str) -> None:
        """Asocia un topic Thingino con el nombre de una cámara.

        Llamar antes de suscribirse al broker MQTT. El mapeo permite que
        `is_camera_active` consulte por nombre de cámara, no por topic.
        """
        if not motion_topic:
            return
        with self._lock:
            self._topic_to_camera[motion_topic] = camera_name
        log.debug("Motion trigger: '%s' mapped to camera '%s'", motion_topic, camera_name)

    def all_topics(self, global_topics: list[str]) -> list[str]:
        """Devuelve todos los topics a suscribirse: globales + por-cámara."""
        with self._lock:
            camera_topics = list(self._topic_to_camera.keys())
        return list(set(global_topics + camera_topics))

    # ------------------------------------------------------------------
    # Consultas de estado
    # ------------------------------------------------------------------

    @property
    def is_active(self) -> bool:
        """True si ALGUNA cámara (o global) tiene movimiento activo.

        Cuando motion_trigger está desactivado, siempre devuelve True
        (comportamiento transparente / pass-through).
        """
        if not self._cfg.enabled:
            return True
        now = time.monotonic()
        with self._lock:
            return any(
                now - ts < self._cfg.cooldown_seconds
                for ts in self._last_event.values()
            )

    def is_camera_active(self, camera_name: str) -> bool:
        """True si ESTA cámara específica tiene movimiento activo.

        Además, los eventos en global_topics activan TODAS las cámaras.
        Si motion_trigger está desactivado, siempre devuelve True.
        """
        if not self._cfg.enabled:
            return True
        now = time.monotonic()
        cooldown = self._cfg.cooldown_seconds
        with self._lock:
            # La cámara está activa si su propio evento es reciente…
            cam_active = (now - self._last_event[camera_name]) < cooldown
            # …o si hay un evento global reciente (activa todas las cámaras).
            global_active = any(
                k not in self._topic_to_camera.values()
                and (now - ts) < cooldown
                for k, ts in self._last_event.items()
            )
        return cam_active or global_active

    @property
    def seconds_since_last_global_event(self) -> float:
        """Segundos desde el evento más reciente en cualquier cámara."""
        with self._lock:
            if not self._last_event:
                return float("inf")
            return time.monotonic() - max(self._last_event.values())

    # ------------------------------------------------------------------
    # Callback MQTT
    # ------------------------------------------------------------------

    def on_mqtt_message(self, topic: str, payload: str) -> None:
        """Llamado por MqttClient al recibir un mensaje en un topic suscrito.

        Thingino HA integration publica "ON" al detectar movimiento y "OFF"
        al terminar. Solo "ON" activa el TPU — "OFF" se ignora para no
        reiniciar el cooldown cuando el movimiento cesa.
        """
        p = payload.strip().upper()
        # Ignorar: vacío, "OFF", "0", "FALSE" — señales de fin de movimiento.
        if not p or p in ("OFF", "0", "FALSE"):
            return

        now = time.monotonic()
        with self._lock:
            # Resolver topic → nombre de cámara (si está mapeado)
            key = self._topic_to_camera.get(topic, topic)
            self._last_event[key] = now
            self._total_events += 1
            count = self._total_events

        log.info(
            "Motion event #%d — topic='%s' camera='%s' — TPU active for %.0fs",
            count,
            topic,
            self._topic_to_camera.get(topic, "global"),
            self._cfg.cooldown_seconds,
        )
