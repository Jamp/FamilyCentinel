# FamilyCentinel

Detector de presencia doméstico **100% local** que procesa el stream RTSP de una cámara IP usando un **Coral Edge TPU USB-C** y publica el estado de presencia a **Home Assistant vía MQTT Discovery**.

No requiere ninguna conexión a la nube. Todo el procesamiento ocurre en casa.

---

## ¿Qué detecta?

| Sensor HA | Condición |
|-----------|-----------|
| `binary_sensor.familycentinel_person` | Persona presente en cámara |
| `binary_sensor.familycentinel_dog` | Perro presente (clase `dog` del modelo COCO) |

---

## Hardware necesario

| Componente | Notas |
|---|---|
| **Coral USB-C Accelerator** (MA2485) | [coral.ai/products/accelerator](https://coral.ai/products/accelerator/) — necesita USB 3.0 para máxima velocidad |
| Host Linux (arm64 o amd64) | Raspberry Pi 4/5, NUC, servidor x86 |
| Cámara IP con stream RTSP | Cualquier cámara doméstica con RTSP H.264/MJPEG |
| Broker MQTT | Normalmente el add-on Mosquitto de Home Assistant |

> **Nota sobre el Coral USB-C (MA2485):** el dispositivo re-enumera en el bus USB durante la primera inferencia — por eso se mapea `/dev/bus/usb` completo y no un path específico.

---

## Instalación paso a paso

### 1. Prerrequisitos del host

```bash
# Instalar libedgetpu en el HOST (además de dentro del contenedor)
echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" \
  | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
sudo apt-get update
sudo apt-get install libedgetpu1-std

# Añadir tu usuario al grupo plugdev (acceso USB sin root)
sudo usermod -aG plugdev $USER
# Cerrar sesión y volver a entrar para aplicar
```

### 2. Clonar y configurar

```bash
git clone https://github.com/Jamp/FamilyCentinel.git
cd familycentinel

# Copiar y editar la configuración
cp config.example.yaml config.yaml
nano config.yaml   # Ajustar URL RTSP, credenciales MQTT, etc.
```

### 3. Levantar con Docker Compose

```bash
docker compose up -d

# Verificar logs
docker compose logs -f
```

Los modelos se descargan automáticamente durante el primer `docker compose build`.

---

## Herramienta de diagnóstico

`tools/calibrate.py` captura un frame de la primera cámara configurada, ejecuta la inferencia y guarda la imagen anotada con los bboxes, scores y coordenadas de centro de cada detección.

Útil para ajustar `detection.min_confidence` y definir `exclusion_zones`.

```bash
# Desde el host (con dependencias instaladas)
python tools/calibrate.py --config config.yaml --output snapshot.jpg

# Dentro del contenedor
docker compose exec familycentinel python tools/calibrate.py \
  --config /app/config/config.yaml --output /tmp/snapshot.jpg
```

---

## Cómo aparece en Home Assistant

FamilyCentinel usa **MQTT Discovery**. Cuando el servicio arranca, publica automáticamente los mensajes de configuración y Home Assistant crea los tres `binary_sensor` sin ninguna configuración manual.

Los sensores aparecen agrupados bajo el dispositivo **"FamilyCentinel"** en:
`Configuración → Dispositivos e integraciones → MQTT`

### Topics MQTT relevantes

| Topic | Descripción |
|---|---|
| `homeassistant/binary_sensor/familycentinel_<sensor>/config` | Discovery config (retain) |
| `familycentinel/<sensor>/state` | Estado `ON` / `OFF` (retain) |
| `familycentinel/status` | `online` / `offline` (LWT) |

---

## Versiones fijadas y decisiones de diseño

| Componente | Versión | Razón |
|---|---|---|
| Python | 3.11 | Estable LTS con soporte activo hasta 2027 |
| `ai-edge-litert` | ≥1.0.1 | Sucesor oficial de `tflite-runtime`; soporta Python 3.11+ |
| `paho-mqtt` | ≥2.1.0 | Única versión con soporte MQTT 5.0 y API callback v2 |
| `opencv-python-headless` | ≥4.9.0 | Sin deps de GUI; mantiene soporte RTSP vía FFmpeg |
| `libedgetpu1-std` | repo coral | Clock reducido (menor temperatura); `max` disponible si se necesita latencia mínima |
| Modelo | SSD MobileNet v2 COCO | Mejor equilibrio velocidad/precisión para detección COCO en Edge TPU |

**Decisión: `ai-edge-litert` en lugar de `pycoral`**
El repositorio oficial `google-coral/edgetpu` fue archivado en abril de 2026. `pycoral` sólo tiene wheels para Python 3.6–3.9. Se usa el delegate de `libedgetpu` directamente desde `ai-edge-litert`, que es la vía recomendada por Google para Python 3.10+.

**Decisión: Buffer de cámara = 1 frame**
OpenCV por defecto mantiene un buffer de 4-8 frames en RTSP. Forzar buffer=1 garantiza que siempre se procesa el frame más reciente, evitando latencia acumulada cuando el TPU procesa más lento que el stream.

---

## Troubleshooting

### Coral USB-C no detectado

```
Could not load Edge TPU delegate — falling back to CPU
```

1. Verificar que el dispositivo está conectado: `lsusb | grep 18d1` (debe aparecer `18d1:9302` o `18d1:9303`)
2. Verificar permisos: `ls -la /dev/bus/usb/*/*` — el usuario del contenedor necesita acceso
3. Comprobar que `libedgetpu1-std` está instalado en el HOST: `dpkg -l libedgetpu1-std`
4. En Raspberry Pi 5 puede ser necesario usar `libedgetpu1-max` — cambiar en el Dockerfile
5. Reiniciar el contenedor tras conectar el Coral por primera vez

### Stream RTSP que se cae

El módulo `camera.py` reconecta automáticamente tras `reconnect_delay_s` segundos. Para depurar:

```bash
# Probar el stream manualmente
ffplay rtsp://usuario:pass@192.168.1.100:554/stream

# O con OpenCV
python -c "
import cv2
cap = cv2.VideoCapture('rtsp://usuario:pass@192.168.1.100:554/stream', cv2.CAP_FFMPEG)
print('Opened:', cap.isOpened())
ret, frame = cap.read()
print('Read:', ret, frame.shape if ret else 'N/A')
"
```

### Falsos positivos

- Aumentar `detection.min_confidence` a 0.6
- Añadir `detection.exclusion_zones` para ignorar zonas con objetos estáticos
- Ejecutar `tools/calibrate.py` para ver los scores reales en cada zona del frame

### Los sensores no aparecen en Home Assistant

1. Verificar que el broker MQTT está corriendo: `mosquitto_ping -h 192.168.1.10`
2. Suscribirse manualmente para ver los mensajes:
   ```bash
   mosquitto_sub -h 192.168.1.10 -t "homeassistant/#" -v
   mosquitto_sub -h 192.168.1.10 -t "familycentinel/#" -v
   ```
3. Verificar que HA tiene habilitada la integración MQTT con Discovery activado

---

## Limitaciones conocidas

- **Múltiples personas en el mismo frame:** se detectan todas correctamente siempre que no se superpongan demasiado.
- **Iluminación nocturna:** si la cámara tiene IR, el modelo funciona con imágenes en escala de grises convertidas por OpenCV; reducir `min_confidence` si hay más falsos negativos.
- **El perro detrás de una persona:** oclusión parcial puede reducir la confianza por debajo del umbral.
- **Resolución baja del stream:** el modelo acepta 300×300px; streams de 640×480 o superior son más que suficientes.
