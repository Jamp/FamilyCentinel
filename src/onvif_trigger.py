"""Puerta de movimiento basada en suscripciones ONVIF PullPoint.

Reemplaza al antiguo trigger MQTT de Thingino. ONVIF PullPoint es el mecanismo
estándar (WS-BaseNotification) que Thingino, Hikvision, Dahua, Reolink y
prácticamente cualquier cámara IP moderna implementan correctamente. El TPU
sólo procesa frames cuando el ONVIF de la cámara reporta movimiento.

ARQUITECTURA:
    - Un único asyncio loop corre en un daemon thread independiente.
    - Por cada cámara con `onvif.host` configurado se lanza una `asyncio.Task`.
    - Cada task abre una suscripción PullPoint y hace `PullMessages` (que
      bloquea en el broker ONVIF de la cámara hasta `Timeout` o `MessageLimit`).
    - Llamadas síncronas de `onvif-zeep` se envuelven en `asyncio.to_thread`.

THREADING:
    El thread principal (loop de procesamiento) sólo LEE el estado vía
    `is_active` / `is_camera_active`. El thread de asyncio ESCRIBE en
    `_last_event` desde la callback de PullMessages. Un único `threading.Lock`
    coordina ambos lados.

RECONEXIÓN:
    Backoff exponencial 2s → 60s. Se resetea a 2s tras una iteración con
    éxito (PullMessages devuelve sin excepción). `ONVIFCamera()` puede fallar
    con red caída — el wrapper de retry lo maneja sin reventar la task.

DESACTIVACIÓN:
    Cuando `cfg.enabled=False`, no se conecta a ONVIF en absoluto y los
    consultores devuelven siempre `True` (pass-through). Útil para testing
    o cuando se quiere procesar todos los frames sin gating.
"""
from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections import defaultdict
from typing import Optional

from lxml import etree
from onvif import ONVIFCamera

from src.config import CameraConfig, OnvifTriggerConfig

log = logging.getLogger(__name__)

# Namespace del schema ONVIF — usado para parsear el XML de Message con lxml.
# Thingino no incluye el Topic en las notificaciones, así que no filtramos por él;
# en cambio buscamos directamente los SimpleItem por nombre.
_ONVIF_NS = "http://www.onvif.org/ver10/schema"

# Nombres de SimpleItem que indican estado de movimiento. ONVIF estándar usa
# `IsMotion` (booleano). Algunos firmwares emiten `State` con el mismo semántico.
_MOTION_ITEM_NAMES = frozenset({"IsMotion", "State"})

# Backoff exponencial entre reconexiones por cámara.
_BACKOFF_INITIAL_S = 2.0
_BACKOFF_MAX_S = 60.0

# PullMessages: timeout por iteración. PT30S = 30 segundos en duración ISO 8601.
# La cámara mantiene la conexión abierta y emite mensajes en cuanto los tiene,
# o devuelve vacío al expirar el timeout (long-polling). MessageLimit limita
# cuántos eventos se entregan por respuesta.
_PULL_TIMEOUT = "PT30S"
_PULL_MESSAGE_LIMIT = 10

_NEVER: float = 0.0


class OnvifTrigger:
    """Puerta de movimiento ONVIF con tracking por cámara.

    Interface idéntica al antiguo `MotionTrigger`: el resto del sistema no
    necesita saber si los eventos vienen de MQTT o de ONVIF.
    """

    def __init__(
        self,
        cfg: OnvifTriggerConfig,
        cameras: list[CameraConfig],
    ) -> None:
        self._cfg = cfg
        self._cameras = cameras

        self._lock = threading.Lock()
        self._last_event: dict[str, float] = defaultdict(lambda: _NEVER)
        self._total_events: int = 0

        # asyncio loop ejecutado en thread propio. Se materializa en `start()`.
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    # ------------------------------------------------------------------
    # Consultas de estado (thread-safe — main thread reader)
    # ------------------------------------------------------------------

    @property
    def is_active(self) -> bool:
        """True si ALGUNA cámara tiene movimiento dentro del cooldown."""
        if not self._cfg.enabled:
            return True
        now = time.monotonic()
        with self._lock:
            return any(
                now - ts < self._cfg.cooldown_seconds
                for ts in self._last_event.values()
            )

    def is_camera_active(self, camera_name: str) -> bool:
        """True si esta cámara concreta tiene movimiento dentro del cooldown."""
        if not self._cfg.enabled:
            return True
        now = time.monotonic()
        with self._lock:
            return (now - self._last_event[camera_name]) < self._cfg.cooldown_seconds

    @property
    def seconds_since_last_global_event(self) -> float:
        """Segundos desde el evento más reciente en cualquier cámara."""
        with self._lock:
            if not self._last_event:
                return float("inf")
            return time.monotonic() - max(self._last_event.values())

    # ------------------------------------------------------------------
    # Ciclo de vida del thread asyncio
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Lanza el loop asyncio en un daemon thread y suscribe a cada cámara."""
        if not self._cfg.enabled:
            log.info("ONVIF trigger disabled — pass-through mode (TPU always active)")
            return

        cameras_with_onvif = [c for c in self._cameras if c.onvif.host]
        if not cameras_with_onvif:
            log.warning(
                "ONVIF trigger enabled but no camera has `onvif.host` configured. "
                "Add an `onvif:` block to each camera or set `onvif_trigger.enabled=false`."
            )
            return

        self._thread = threading.Thread(
            target=self._run_loop,
            name="onvif-trigger",
            daemon=True,
        )
        self._thread.start()
        log.info(
            "ONVIF trigger started — %d camera(s) subscribed",
            len(cameras_with_onvif),
        )

    def stop(self) -> None:
        """Señaliza parada al loop asyncio y espera al thread brevemente."""
        if self._loop is None or self._thread is None:
            return
        self._stop_event.set()
        # Programar la cancelación de tareas en el loop desde fuera de su thread.
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5.0)
        if self._thread.is_alive():
            log.warning("ONVIF trigger thread did not exit cleanly within 5s")

    # ------------------------------------------------------------------
    # Internals — asyncio
    # ------------------------------------------------------------------

    def _run_loop(self) -> None:
        """Punto de entrada del thread: crea loop y arranca tasks por cámara."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        try:
            tasks = [
                self._loop.create_task(
                    self._camera_task(cam),
                    name=f"onvif-{cam.name}",
                )
                for cam in self._cameras
                if cam.onvif.host
            ]
            try:
                self._loop.run_forever()
            finally:
                for task in tasks:
                    task.cancel()
                # Drenar cancelaciones para que no queden warnings de tareas pendientes.
                self._loop.run_until_complete(
                    asyncio.gather(*tasks, return_exceptions=True)
                )
        finally:
            self._loop.close()
            self._loop = None

    async def _camera_task(self, cam: CameraConfig) -> None:
        """Loop de reconexión + PullMessages para una sola cámara."""
        backoff = _BACKOFF_INITIAL_S
        host, port = cam.onvif.host, cam.onvif.port
        user, password = cam.onvif.username, cam.onvif.password

        while not self._stop_event.is_set():
            try:
                await self._subscribe_and_pull(cam.name, host, port, user, password)
                # Si _subscribe_and_pull retorna sin excepción es por stop.
                return
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.warning(
                    "ONVIF[%s] connection error (%s) — retrying in %.1fs",
                    cam.name,
                    exc,
                    backoff,
                )
                try:
                    await asyncio.sleep(backoff)
                except asyncio.CancelledError:
                    raise
                backoff = min(backoff * 2, _BACKOFF_MAX_S)

    async def _subscribe_and_pull(
        self,
        camera_name: str,
        host: str,
        port: int,
        user: str,
        password: str,
    ) -> None:
        """Crea suscripción PullPoint y procesa mensajes hasta que falle o pare."""
        # ONVIFCamera y todas sus operaciones son síncronas (zeep + WSDL files);
        # las desplazamos a un threadpool para no bloquear el loop.
        cam = await asyncio.to_thread(ONVIFCamera, host, port, user, password)
        await asyncio.to_thread(cam.update_xaddrs)

        # CreatePullPointSubscription registra a FamilyCentinel como subscriber.
        # Sin esta llamada, PullMessages devuelve siempre vacío porque la cámara
        # no tiene ningún subscriber al que encolar eventos.
        events = cam.create_events_service()
        await asyncio.to_thread(
            events.CreatePullPointSubscription,
            {"InitialTerminationTime": "PT1H"},
        )
        service = cam.create_pullpoint_service()
        log.info("ONVIF[%s] PullPoint subscription created on %s:%d", camera_name, host, port)

        while not self._stop_event.is_set():
            messages = await asyncio.to_thread(
                service.PullMessages,
                {"Timeout": _PULL_TIMEOUT, "MessageLimit": _PULL_MESSAGE_LIMIT},
            )
            notifications = getattr(messages, "NotificationMessage", None) or []

            if not notifications:
                log.debug("ONVIF[%s] empty PullMessages response", camera_name)
                continue

            for msg in notifications:
                self._handle_notification(camera_name, msg)

    def _handle_notification(self, camera_name: str, msg: object) -> None:
        """Extrae IsMotion del NotificationMessage y actualiza el estado.

        Thingino no incluye el Topic en las notificaciones, así que en lugar de
        filtrar por topic usamos lxml para buscar directamente los SimpleItem
        en el elemento Data del mensaje.
        """
        elem = getattr(getattr(msg, "Message", None), "_value_1", None)
        if elem is None:
            return

        # Buscar todos los SimpleItem bajo tt:Data
        data_el = elem.find(f"{{{_ONVIF_NS}}}Data")
        if data_el is None:
            return

        for item in data_el.findall(f"{{{_ONVIF_NS}}}SimpleItem"):
            name = item.get("Name")
            if name not in _MOTION_ITEM_NAMES:
                continue

            value = item.get("Value", "")
            # ONVIF transporta booleanos como strings "true"/"false".
            is_motion = value.lower() == "true"

            if not is_motion:
                # Sólo nos interesa el flanco de subida: "stop motion" no
                # debe reiniciar el cooldown.
                continue

            now = time.monotonic()
            with self._lock:
                self._last_event[camera_name] = now
                self._total_events += 1
                count = self._total_events

            log.info(
                "ONVIF motion #%d — camera='%s' — TPU active for %.0fs",
                count,
                camera_name,
                self._cfg.cooldown_seconds,
            )
