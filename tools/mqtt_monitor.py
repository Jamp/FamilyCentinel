"""Monitor de consola para estados MQTT de FamilyCentinel en tiempo real.

Muestra en la terminal los cambios de estado de todos los sensores de
FamilyCentinel y los eventos de movimiento Thingino. Útil para verificar
que el sistema funciona antes de configurar automatizaciones en HA.

USO:
    python tools/mqtt_monitor.py --config config.yaml

    # O especificando el broker directamente:
    python tools/mqtt_monitor.py --host 192.168.1.10 --port 1883

SALIDA DE EJEMPLO:
    ┌─────────────────────────────────────────────────────┐
    │  FamilyCentinel MQTT Monitor                         │
    │  Broker: 192.168.1.10:1883   Ctrl+C para salir      │
    └─────────────────────────────────────────────────────┘
    [14:32:01] STATUS     familycentinel/status → online
    [14:32:01] SENSOR     familycentinel/adult/state → OFF
    [14:32:01] SENSOR     familycentinel/child/state → OFF
    [14:32:01] SENSOR     familycentinel/dog/state   → OFF
    [14:32:15] MOTION     thingino/salon → {"camera_id":"abc","ts":"1234"}
    [14:32:18] SENSOR ✓   familycentinel/adult/state → ON   (cambió)
    [14:33:01] SENSOR     familycentinel/adult/state → OFF
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import paho.mqtt.client as mqtt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config

logging.basicConfig(level=logging.WARNING)

# ANSI colors (desactivar en Windows sin soporte)
try:
    import os
    _ANSI = os.name != "nt" or os.environ.get("FORCE_COLOR")
except Exception:
    _ANSI = False

_C = {
    "reset":  "\033[0m"  if _ANSI else "",
    "green":  "\033[92m" if _ANSI else "",
    "red":    "\033[91m" if _ANSI else "",
    "yellow": "\033[93m" if _ANSI else "",
    "cyan":   "\033[96m" if _ANSI else "",
    "gray":   "\033[90m" if _ANSI else "",
    "bold":   "\033[1m"  if _ANSI else "",
}

# Estado previo para detectar cambios
_prev_states: dict[str, str] = {}


def _ts() -> str:
    return time.strftime("%H:%M:%S")


def _print_row(tag: str, topic: str, payload: str, changed: bool = False) -> None:
    color = _C["reset"]
    if tag == "SENSOR":
        color = _C["green"] if payload == "ON" else _C["red"]
    elif tag == "MOTION":
        color = _C["yellow"]
    elif tag == "STATUS":
        color = _C["cyan"]

    mark = f"{_C['bold']}✓{_C['reset']} " if changed else "  "
    tag_str = f"{color}{tag:<8}{_C['reset']}"
    payload_str = f"{color}{payload}{_C['reset']}"
    changed_str = f"  {_C['gray']}(cambió){_C['reset']}" if changed else ""

    print(f"[{_ts()}] {tag_str} {mark}{topic:<45} → {payload_str}{changed_str}")


def build_client(
    host: str,
    port: int,
    username: str | None,
    password: str | None,
    device_id: str,
    motion_topics: list[str],
) -> mqtt.Client:

    client = mqtt.Client(
        callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
        client_id=f"fc-monitor-{int(time.time())}",
        clean_session=True,
    )

    if username:
        client.username_pw_set(username, password)

    def on_connect(cl, userdata, flags, rc, props):
        if rc == 0:
            # Suscribirse a todos los topics relevantes de FamilyCentinel
            topics = [
                (f"{device_id}/+/state", 1),    # estados de sensores
                (f"{device_id}/status", 1),       # LWT online/offline
            ]
            for t in motion_topics:
                topics.append((t, 1))
            cl.subscribe(topics)
            print(f"\n{_C['cyan']}Conectado a {host}:{port}{_C['reset']}")
            print(f"{_C['gray']}Escuchando: {[t for t, _ in topics]}{_C['reset']}\n")
        else:
            print(f"Error de conexión: {rc}")

    def on_message(cl, userdata, msg):
        topic = msg.topic
        payload = msg.payload.decode("utf-8", errors="replace").strip()

        changed = _prev_states.get(topic) != payload
        _prev_states[topic] = payload

        if "/state" in topic and device_id in topic:
            sensor = topic.split("/")[1] if "/" in topic else topic
            _print_row("SENSOR", topic, payload, changed)
        elif "status" in topic and device_id in topic:
            _print_row("STATUS", topic, payload)
        elif any(t == topic for t in motion_topics):
            _print_row("MOTION", topic, payload[:60])
        else:
            _print_row("OTHER", topic, payload[:60])

    def on_disconnect(cl, userdata, flags, rc, props):
        if rc != 0:
            print(f"\n{_C['yellow']}Desconectado (rc={rc}) — reconectando…{_C['reset']}")

    client.on_connect = on_connect
    client.on_message = on_message
    client.on_disconnect = on_disconnect

    return client


def main() -> None:
    parser = argparse.ArgumentParser(description="FamilyCentinel MQTT monitor")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--host",   default=None)
    parser.add_argument("--port",   type=int, default=None)
    parser.add_argument("--username", default=None)
    parser.add_argument("--password", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    host     = args.host     or cfg.mqtt.host
    port     = args.port     or cfg.mqtt.port
    username = args.username or cfg.mqtt.username
    password = args.password or cfg.mqtt.password
    device_id = cfg.mqtt.device_id

    motion_topics = [c.motion_topic for c in cfg.cameras if c.motion_topic]
    if cfg.motion_trigger.global_topics:
        motion_topics += cfg.motion_trigger.global_topics

    print(f"""
{_C['bold']}┌──────────────────────────────────────────────────────┐
│  FamilyCentinel MQTT Monitor                          │
│  Broker: {host}:{port:<5}   Ctrl+C para salir      │
└──────────────────────────────────────────────────────┘{_C['reset']}
{_C['gray']}Leyenda:  SENSOR (verde=ON, rojo=OFF)  MOTION (amarillo)  ✓=cambio{_C['reset']}
""")

    client = build_client(host, port, username, password, device_id, motion_topics)

    try:
        client.connect(host, port, keepalive=30)
        client.loop_forever()
    except KeyboardInterrupt:
        print(f"\n{_C['gray']}Saliendo.{_C['reset']}")
    except ConnectionRefusedError:
        print(f"\n{_C['red']}Error: no se puede conectar a {host}:{port}. "
              f"¿Está el broker MQTT en marcha?{_C['reset']}")
        sys.exit(1)
    finally:
        client.disconnect()


if __name__ == "__main__":
    main()
