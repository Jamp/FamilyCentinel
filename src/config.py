"""Configuration loader and validator for FamilyCentinel.

SEGURIDAD:
  - `MqttConfig` y `CameraConfig` redefinen `__repr__` para enmascarar
    credenciales (password MQTT y password embebido en URL RTSP). Esto evita
    que un `log.exception(...)` o un `print(cfg)` filtren secretos a stdout
    o a la rotación de logs de Docker.
  - Se usa `yaml.safe_load` (nunca `yaml.load`) para evitar deserialización
    arbitraria de objetos Python (CVE-2020-14343 y similares).
  - Las credenciales MQTT se pueden inyectar via variables de entorno
    (MQTT_HOST, MQTT_PORT, MQTT_USERNAME, MQTT_PASSWORD) desde un .env con
    permisos 600, sin necesidad de escribirlas en config.yaml.

COMPATIBILIDAD:
  - Se soporta tanto `camera:` (cámara única, legacy) como `cameras:` (lista).
    Si se usa `camera:`, se convierte automáticamente a lista de un elemento.
"""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers de enmascarado — sólo para representación en logs, no afectan al
# valor real de los campos.
# ---------------------------------------------------------------------------
_RTSP_CRED_RE = re.compile(r"^(?P<scheme>rtsps?://)(?P<user>[^:@/]+):(?P<pass>[^@/]+)@")


def mask_rtsp_url(url: str) -> str:
    """Devuelve la URL RTSP con la contraseña sustituida por '***'."""
    if not url:
        return url
    return _RTSP_CRED_RE.sub(r"\g<scheme>\g<user>:***@", url)


def _mask_secret(value: Optional[str]) -> str:
    if not value:
        return "<empty>"
    return "***"


# ---------------------------------------------------------------------------
# Dataclasses de configuración
# ---------------------------------------------------------------------------

@dataclass
class CameraConfig:
    """Configuración de una cámara individual.

    Multi-cámara: añadir una entrada por cámara en la sección `cameras:` del
    YAML. Cada cámara puede tener su propio topic Thingino en `motion_topic`.

    Compatibilidad: si se usa la sección `camera:` (singular, versión antigua),
    se convierte automáticamente a una lista con un elemento.
    """
    # Nombre identificador de la cámara (aparece en logs y en el topic MQTT).
    # Debe ser único si hay varias cámaras.
    name: str = "default"

    # Tipo de fuente: "rtsp" (cámara IP, recomendado) o "usb" (fallback local).
    type: str = "rtsp"

    # URL RTSP completa incluyendo credenciales.
    # SEGURIDAD: usar usuario de sólo lectura con contraseña aleatoria.
    url: str = "rtsp://192.168.1.100:554/stream"

    # Índice de dispositivo USB (sólo si type: "usb").
    device_index: int = 0

    # FPS objetivo de captura. 10 fps es suficiente para detección de presencia.
    target_fps: int = 10

    # Resolución solicitada al stream RTSP (puede ser ignorada por la cámara).
    width: int = 640
    height: int = 480

    # Segundos entre reintentos de conexión tras un fallo.
    reconnect_delay_s: float = 5.0

    # Topic MQTT de Thingino para el trigger de movimiento de ESTA cámara.
    # Dejar vacío ("") si no se usa trigger de movimiento para esta cámara.
    # Formato Thingino: thingino/<nombre_camara>
    motion_topic: str = ""

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"CameraConfig(name={self.name!r}, type={self.type!r}, "
            f"url={mask_rtsp_url(self.url)!r}, target_fps={self.target_fps}, "
            f"motion_topic={self.motion_topic!r})"
        )


@dataclass
class DetectionConfig:
    """Parámetros del modelo de detección de objetos."""
    # Confianza mínima global [0.0–1.0]. Se aplica a todas las clases salvo
    # que se especifique un valor concreto en `class_min_confidence`.
    min_confidence: float = 0.5
    # Clases del modelo COCO que se procesan. "person" incluye adultos y niños.
    classes: list[str] = field(default_factory=lambda: ["person", "dog"])
    # Umbrales de confianza por clase. Sobreescriben `min_confidence`.
    # Útil para perros pequeños: el modelo los detecta con confianza baja
    # aunque los vea correctamente.
    class_min_confidence: dict[str, float] = field(default_factory=dict)
    # Zonas de exclusión por cámara: detecciones cuyo CENTRO caiga dentro
    # de alguna zona son ignoradas. Útil para objetos estáticos (ropa en sofá,
    # pósters, etc.) que el modelo confunde con personas.
    # Formato: {camera_name: [[xmin, ymin, xmax, ymax], ...]} normalizado [0-1].
    exclusion_zones: dict[str, list[list[float]]] = field(default_factory=dict)

    def confidence_for(self, label: str) -> float:
        return self.class_min_confidence.get(label, self.min_confidence)

    def is_excluded(self, camera_name: str, bbox: tuple) -> bool:
        """True si el centro del bbox cae en una zona de exclusión de esa cámara."""
        zones = self.exclusion_zones.get(camera_name, [])
        if not zones:
            return False
        ymin, xmin, ymax, xmax = bbox
        cx = (xmin + xmax) / 2
        cy = (ymin + ymax) / 2
        return any(z[0] <= cx <= z[2] and z[1] <= cy <= z[3] for z in zones)


@dataclass
class StabilizationConfig:
    """Debounce temporal para evitar parpadeos en Home Assistant.

    Una entidad pasa a "presente" tras `frames_to_activate` detecciones
    consecutivas, y a "ausente" cuando se cumplen DOS condiciones a la vez:
      1. `frames_to_deactivate` frames consecutivos sin detección.
      2. Han pasado al menos `min_presence_seconds` desde la última detección.

    La condición 2 protege contra falsos ausentes en escenarios donde la
    persona está presente pero el modelo no la detecta temporalmente:
      - Durmiendo bajo una cobija
      - Agachada fuera del campo de visión
      - Sentada muy quieta (confianza baja intermitente)

    Con 10 fps y valores por defecto:
      - frames_to_activate=3        → presencia confirmada en ~0.3s
      - frames_to_deactivate=30     → empieza a considerar ausencia tras 3s
      - min_presence_seconds=600    → ausencia confirmada 10 min después de
                                       la última detección positiva
    """
    frames_to_activate: int = 3
    frames_to_deactivate: int = 30
    # Segundos mínimos desde la última detección para marcar ausente.
    # 0 = desactivado (comportamiento original sin protección).
    # 600 = 10 minutos (recomendado para hogares con personas que duermen).
    min_presence_seconds: float = 600.0


@dataclass
class MotionTriggerConfig:
    """Puerta de movimiento basada en eventos MQTT de Thingino.

    Cuando está activo, el Edge TPU sólo procesa frames mientras alguna de las
    cámaras haya detectado movimiento recientemente (dentro de `cooldown_seconds`).
    Esto reduce drásticamente el consumo de CPU/TPU en periodos sin actividad.

    Cada cámara puede tener su propio topic configurado en `CameraConfig.motion_topic`.
    Esta sección también soporta topics globales (que no están asociados a una cámara
    concreta) en `global_topics`.

    CONFIGURACIÓN EN THINGINO (una vez por cámara):
      1. Abre la web UI: http://<IP_CAMARA>
      2. Ve a: Tools → Motion Guard
      3. Activa "Motion Guard" y marca "Send to MQTT"
      4. Configura el mismo broker MQTT de FamilyCentinel
      5. Anota el topic (por defecto: thingino/<nombre_camara>)
      6. Pon ese topic en `cameras[n].motion_topic` del config.yaml
    """
    # False = procesar todos los frames siempre (comportamiento original).
    enabled: bool = False

    # Topics globales opcionales (cualquier evento en ellos activa TODAS las cámaras).
    # Útil si tienes un sensor PIR externo o quieres un topic maestro de activación.
    global_topics: list[str] = field(default_factory=list)

    # Segundos que el TPU permanece activo tras el último evento de movimiento.
    # Aumentar si hay personas que se mueven lentamente y se pierden detecciones.
    cooldown_seconds: float = 30.0


@dataclass
class MqttTlsConfig:
    """Configuración TLS para la conexión MQTT.

    Activar cuando el broker MQTT es accesible desde otras VLANs o redes
    no completamente confiables (p.ej. broker en DMZ o acceso remoto).
    Para redes domésticas con broker en localhost o 192.168.x.x la configuración
    por defecto (disabled) es aceptable con las demás capas de seguridad.
    """
    enabled: bool = False
    # Ruta dentro del contenedor al CA bundle (montar como volumen ro).
    # Si es None y enabled=True, usa el bundle del sistema operativo.
    ca_certs: Optional[str] = None
    # True desactiva la validación de hostname/cert. SÓLO para entornos de
    # pruebas con certificados auto-firmados. Nunca en producción real.
    insecure: bool = False


@dataclass
class MqttConfig:
    """Configuración del broker MQTT (Home Assistant / Mosquitto).

    Las credenciales pueden pasarse por variables de entorno para no
    escribirlas en el archivo de configuración:
      MQTT_HOST, MQTT_PORT, MQTT_USERNAME, MQTT_PASSWORD
    """
    host: str = "localhost"
    port: int = 1883
    username: Optional[str] = None
    password: Optional[str] = None
    discovery_prefix: str = "homeassistant"
    device_name: str = "FamilyCentinel"
    device_id: str = "familycentinel"
    keepalive: int = 60
    tls: MqttTlsConfig = field(default_factory=MqttTlsConfig)

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"MqttConfig(host={self.host!r}, port={self.port}, "
            f"username={self.username!r}, password={_mask_secret(self.password)}, "
            f"discovery_prefix={self.discovery_prefix!r}, "
            f"device_name={self.device_name!r}, device_id={self.device_id!r}, "
            f"keepalive={self.keepalive}, tls={self.tls!r})"
        )


@dataclass
class AppConfig:
    """Configuración raíz de FamilyCentinel."""
    # Lista de cámaras. Soporta una o más cámaras RTSP/USB.
    cameras: list[CameraConfig] = field(default_factory=lambda: [CameraConfig()])
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    stabilization: StabilizationConfig = field(default_factory=StabilizationConfig)
    motion_trigger: MotionTriggerConfig = field(default_factory=MotionTriggerConfig)
    mqtt: MqttConfig = field(default_factory=MqttConfig)
    model_path: Path = Path("/app/models/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite")
    labels_path: Path = Path("/app/models/coco_labels.txt")


# ---------------------------------------------------------------------------
# Loader principal
# ---------------------------------------------------------------------------

def _merge_env(cfg: AppConfig) -> None:
    """Sobreescribe configuración MQTT con variables de entorno cuando existen.

    Las variables de entorno tienen mayor precedencia que config.yaml para
    facilitar la inyección de secretos desde un archivo .env (chmod 600)
    sin necesidad de modificar el YAML.
    """
    mapping = {
        "MQTT_HOST":             ("mqtt", "host"),
        "MQTT_PORT":             ("mqtt", "port"),
        "MQTT_USERNAME":         ("mqtt", "username"),
        "MQTT_PASSWORD":         ("mqtt", "password"),
        "MQTT_DISCOVERY_PREFIX": ("mqtt", "discovery_prefix"),
    }
    for env_var, (section, key) in mapping.items():
        value = os.environ.get(env_var)
        if value is None or value == "":
            continue
        section_obj = getattr(cfg, section)
        existing = getattr(section_obj, key)
        if existing is not None:
            try:
                value = type(existing)(value)  # type: ignore[arg-type]
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid value for env var {env_var}: cannot cast "
                    f"'{value}' to {type(existing).__name__}"
                ) from exc
        setattr(section_obj, key, value)


def _parse_cameras(raw: dict) -> list[CameraConfig]:
    """Parsea `cameras:` (lista) o `camera:` (singular, legacy) del YAML."""
    # Formato nuevo: cameras es una lista
    cameras_raw = raw.get("cameras")
    if cameras_raw is not None:
        if not isinstance(cameras_raw, list):
            raise ValueError("`cameras` must be a list of camera configs")
        result = []
        for i, cam in enumerate(cameras_raw):
            if not isinstance(cam, dict):
                raise ValueError(f"`cameras[{i}]` must be a mapping")
            # Asignar nombre por defecto si no se especifica
            if "name" not in cam:
                cam["name"] = f"cam{i}"
            result.append(CameraConfig(**cam))
        return result

    # Formato legacy: camera es un dict singular — convertir a lista
    camera_raw = raw.get("camera", {}) or {}
    if camera_raw:
        if "name" not in camera_raw:
            camera_raw["name"] = "default"
        return [CameraConfig(**camera_raw)]

    return [CameraConfig()]


def load_config(config_path) -> AppConfig:
    """Carga config.yaml, aplica sobreescrituras de env y devuelve AppConfig validado.

    Orden de precedencia (mayor a menor):
      1. Variables de entorno (MQTT_HOST, etc.)
      2. config.yaml
      3. Valores por defecto de los dataclasses
    """
    config_path = Path(config_path)
    if not config_path.exists():
        log.warning("Config file %s not found — using defaults", config_path)
        cfg = AppConfig()
        _merge_env(cfg)
        return cfg

    with config_path.open() as fh:
        raw = yaml.safe_load(fh) or {}

    if not isinstance(raw, dict):
        raise ValueError(
            f"Config root must be a mapping; got {type(raw).__name__}"
        )

    # Secciones opcionales — usar dict vacío como fallback seguro
    det_raw    = raw.get("detection", {}) or {}
    stab_raw   = raw.get("stabilization", {}) or {}
    motion_raw = raw.get("motion_trigger", {}) or {}
    mqtt_raw   = raw.get("mqtt", {}) or {}
    tls_raw    = mqtt_raw.pop("tls", {}) if isinstance(mqtt_raw, dict) else {}
    tls_raw    = tls_raw or {}

    mqtt_obj = (
        MqttConfig(**mqtt_raw, tls=MqttTlsConfig(**tls_raw))
        if mqtt_raw
        else MqttConfig(tls=MqttTlsConfig(**tls_raw))
    )

    cfg = AppConfig(
        cameras=_parse_cameras(raw),
        detection=DetectionConfig(**det_raw) if det_raw else DetectionConfig(),
        stabilization=StabilizationConfig(**stab_raw) if stab_raw else StabilizationConfig(),
        motion_trigger=MotionTriggerConfig(**motion_raw) if motion_raw else MotionTriggerConfig(),
        mqtt=mqtt_obj,
        model_path=Path(raw.get("model_path", AppConfig.model_path)),
        labels_path=Path(raw.get("labels_path", AppConfig.labels_path)),
    )

    _merge_env(cfg)

    if not cfg.cameras:
        raise ValueError("At least one camera must be configured under `cameras:`")

    camera_names = [c.name for c in cfg.cameras]
    if len(camera_names) != len(set(camera_names)):
        raise ValueError(
            f"Camera names must be unique; found duplicates in: {camera_names}"
        )

    # Aviso de seguridad: credenciales sin TLS fuera de localhost
    if (
        cfg.mqtt.username
        and not cfg.mqtt.tls.enabled
        and cfg.mqtt.host not in ("localhost", "127.0.0.1", "::1")
    ):
        log.warning(
            "MQTT credentials are sent in clear-text to %s:%d (TLS disabled). "
            "Enable mqtt.tls.enabled or restrict the broker to localhost.",
            cfg.mqtt.host,
            cfg.mqtt.port,
        )

    log.info(
        "Config loaded: %d camera(s), motion_trigger=%s",
        len(cfg.cameras),
        cfg.motion_trigger.enabled,
    )
    return cfg
