"""Monitor de consola para estados MQTT de FamilyCentinel en tiempo real.

Muestra en la terminal los cambios de estado de todos los sensores de
FamilyCentinel. Útil para verificar que el sistema funciona antes de
configurar automatizaciones en HA.

Nota: el monitor de eventos de movimiento Thingino MQTT se eliminó al
migrar el trigger a ONVIF PullPoint (los eventos ya no pasan por MQTT).

USO:
    python tools/mqtt_monitor.py --config config.yaml

    # O especificando el broker directamente:
    python tools/mqtt_monitor.py --host 192.168.1.10 --port 1883
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
            topics = [
                (f"{device_id}/+/state", 1),    # estados de sensores
                (f"{device_id}/status", 1),     # LWT online/offline
            ]
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
            _print_row("SENSOR", topic, payload, changed)
        elif "status" in topic and device_id in topic:
            _print_row("STATUS", topic, payload)
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

    print(f"""
{_C['bold']}┌──────────────────────────────────────────────────────┐
│  FamilyCentinel MQTT Monitor                          │
│  Broker: {host}:{port:<5}   Ctrl+C para salir      │
└──────────────────────────────────────────────────────┘{_C['reset']}
{_C['gray']}Leyenda:  SENSOR (verde=ON, rojo=OFF)  ✓=cambio{_C['reset']}
""")

    client = build_client(host, port, username, password, device_id)

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
