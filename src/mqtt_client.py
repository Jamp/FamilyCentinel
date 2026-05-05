"""MQTT client with Home Assistant Discovery support.

Uses paho-mqtt 2.x API (CallbackAPIVersion.VERSION2).

SEGURIDAD:
  - Soporta TLS (parámetro `mqtt.tls.enabled` en config.yaml). Cuando está
    activado, paho-mqtt valida el certificado del broker contra `ca_certs`.
  - Las credenciales se pasan a paho vía `username_pw_set` y NUNCA se
    registran en logs (paho no las imprime y este módulo tampoco).
  - El client_id incluye el hostname del contenedor; en redes hostiles esto
    podría ser informativo pero en una LAN doméstica es aceptable y ayuda
    al troubleshooting.

MQTT Discovery:
  Each sensor is published at:
    homeassistant/binary_sensor/familycentinel/<sensor>/config
  State is published at:
    familycentinel/<sensor>/state  (payload: ON / OFF)
  LWT (Last Will & Testament) at:
    familycentinel/status  (online / offline)
"""
from __future__ import annotations

import json
import logging
import socket
import ssl
import threading
import time
from typing import Callable, Optional

import paho.mqtt.client as mqtt

from src.config import MqttConfig
from src.entities import ENTITIES

log = logging.getLogger(__name__)

# Alias kept for readability — SENSORS and ENTITIES are identical.
SENSORS = ENTITIES
_STATUS_TOPIC = "familycentinel/status"


class MqttClient:
    def __init__(self, cfg: MqttConfig, on_connect_cb: Optional[Callable] = None) -> None:
        self._cfg = cfg
        self._on_connect_cb = on_connect_cb
        self._connected = threading.Event()

        self._client = mqtt.Client(
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
            client_id=f"{cfg.device_id}_{socket.gethostname()}",
            clean_session=True,
        )

        if cfg.username:
            self._client.username_pw_set(cfg.username, cfg.password)

        # Configurar TLS si esta habilitado en la configuracion.
        if cfg.tls.enabled:
            self._configure_tls()

        # Last Will & Testament - HA uses this to mark the device unavailable
        self._client.will_set(
            topic=_STATUS_TOPIC,
            payload="offline",
            qos=1,
            retain=True,
        )

        self._client.on_connect = self._on_connect
        self._client.on_disconnect = self._on_disconnect

    # ------------------------------------------------------------------
    # TLS
    # ------------------------------------------------------------------

    def _configure_tls(self) -> None:
        """Activa TLS con validacion estricta por defecto."""
        cfg_tls = self._cfg.tls
        # TLSv1.2+ obligatorio. paho expone ssl.PROTOCOL_TLS_CLIENT que
        # rechaza versiones anteriores y valida certificado por defecto.
        self._client.tls_set(
            ca_certs=cfg_tls.ca_certs,
            tls_version=ssl.PROTOCOL_TLS_CLIENT,
        )
        if cfg_tls.insecure:
            log.warning(
                "MQTT TLS configured with insecure=true: certificate hostname "
                "validation is DISABLED. Use only for local self-signed brokers."
            )
            self._client.tls_insecure_set(True)
        log.info(
            "MQTT TLS enabled (ca_certs=%s, insecure=%s)",
            cfg_tls.ca_certs or "<system default>",
            cfg_tls.insecure,
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """Connect and start the network loop in a background thread."""
        log.info("Connecting to MQTT broker %s:%d (tls=%s)...",
                 self._cfg.host, self._cfg.port, self._cfg.tls.enabled)
        self._client.connect_async(
            host=self._cfg.host,
            port=self._cfg.port,
            keepalive=self._cfg.keepalive,
        )
        self._client.loop_start()

    def wait_connected(self, timeout: float = 30.0) -> bool:
        return self._connected.wait(timeout=timeout)

    def publish_discovery(self) -> None:
        """Publish HA MQTT Discovery config payloads for all three sensors."""
        device_block = {
            "identifiers": [self._cfg.device_id],
            "name": self._cfg.device_name,
            "model": "FamilyCentinel v1",
            "manufacturer": "FamilyCentinel",
        }

        sensor_meta = {
            "person": {"name": "Person Present", "icon": "mdi:account"},
            "dog":    {"name": "Dog Present",    "icon": "mdi:dog"},
        }

        for sensor in SENSORS:
            meta = sensor_meta[sensor]
            state_topic = self._state_topic(sensor)
            config_topic = (
                f"{self._cfg.discovery_prefix}/binary_sensor/"
                f"{self._cfg.device_id}_{sensor}/config"
            )
            payload = {
                "name": meta["name"],
                "unique_id": f"{self._cfg.device_id}_{sensor}",
                "device_class": "occupancy",
                "state_topic": state_topic,
                "payload_on": "ON",
                "payload_off": "OFF",
                "availability_topic": _STATUS_TOPIC,
                "payload_available": "online",
                "payload_not_available": "offline",
                "icon": meta["icon"],
                "device": device_block,
            }
            self._publish(config_topic, json.dumps(payload), retain=True)
            log.info("Discovery published for sensor '%s'", sensor)

    def publish_state(self, sensor: str, present: bool) -> None:
        """Publish ON/OFF state for a single sensor."""
        if sensor not in SENSORS:
            raise ValueError(f"Unknown sensor: {sensor!r}")
        topic = self._state_topic(sensor)
        payload = "ON" if present else "OFF"
        # QoS 1 ensures HA receives the state even if the network blips
        # right when the change occurs (retain handles late subscribers).
        self._publish(topic, payload, retain=True, qos=1)
        log.debug("State published: %s -> %s", topic, payload)

    def publish_online(self) -> None:
        self._publish(_STATUS_TOPIC, "online", retain=True, qos=1)

    def disconnect(self) -> None:
        self._publish(_STATUS_TOPIC, "offline", retain=True, qos=1)
        time.sleep(0.5)
        self._client.loop_stop()
        self._client.disconnect()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _state_topic(self, sensor: str) -> str:
        return f"{self._cfg.device_id}/{sensor}/state"

    def _publish(
        self,
        topic: str,
        payload: str,
        retain: bool = False,
        qos: int = 0,
    ) -> None:
        result = self._client.publish(topic, payload, qos=qos, retain=retain)
        if result.rc != mqtt.MQTT_ERR_SUCCESS:
            log.warning("MQTT publish failed (rc=%d) for topic %s", result.rc, topic)

    # ------------------------------------------------------------------
    # Callbacks (paho-mqtt 2.x signatures)
    # ------------------------------------------------------------------

    def _on_connect(
        self,
        client: mqtt.Client,
        userdata: object,
        connect_flags: mqtt.ConnectFlags,
        reason_code: mqtt.ReasonCode,
        properties: object,
    ) -> None:
        if reason_code == 0:
            log.info("MQTT connected to %s:%d", self._cfg.host, self._cfg.port)
            self._connected.set()
            self.publish_online()
            if self._on_connect_cb:
                self._on_connect_cb()
        else:
            log.error("MQTT connection refused - reason code: %s", reason_code)

    def _on_disconnect(
        self,
        client: mqtt.Client,
        userdata: object,
        disconnect_flags: mqtt.DisconnectFlags,
        reason_code: mqtt.ReasonCode,
        properties: object,
    ) -> None:
        self._connected.clear()
        if reason_code != 0:
            log.warning(
                "MQTT disconnected unexpectedly (rc=%s) - paho will retry",
                reason_code,
            )
