"""Edge TPU / YOLOv8 / ONNX+CoreML / TFLite inference wrapper.

Selección automática de backend según la plataforma:

  macOS (testing):
    1. YOLOv8-nano (ultralytics) → mejor detección de objetos pequeños (perros)
    2. ONNX Runtime + CoreML EP  → ANE/GPU (si existe .onnx)
    3. TFLite + XNNPACK          → CPU ARM NEON

  Linux / producción:
    1. TFLite + Edge TPU (Coral USB-C)
    2. TFLite + XNNPACK (CPU fallback)

YOLOv8-nano ventajas sobre SSD MobileNet v2:
  - Input 640×640 vs 300×300 → detecta objetos 4× más pequeños
  - Mejor precisión en animales pequeños y personas parcialmente visibles
  - 31 fps en M1 Pro (suficiente para target de 10 fps)
  - Solo para testing en Mac; producción sigue usando TFLite + Coral
"""
from __future__ import annotations

import logging
import os
import platform
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Runtime TFLite — prioridad: ai-edge-litert → tflite_runtime
# ---------------------------------------------------------------------------
try:
    from ai_edge_litert.interpreter import Interpreter as _TFLiteInterpreter
    from ai_edge_litert.interpreter import load_delegate as _tflite_load_delegate
    _TFLITE_RUNTIME = "ai-edge-litert"
except ImportError:
    from tflite_runtime.interpreter import Interpreter as _TFLiteInterpreter        # type: ignore
    from tflite_runtime.interpreter import load_delegate as _tflite_load_delegate   # type: ignore
    _TFLITE_RUNTIME = "tflite-runtime"

# Nombre de la librería EdgeTPU según el SO
_IS_MACOS = platform.system() == "Darwin"
EDGETPU_SHARED_LIB = "libedgetpu.1.dylib" if _IS_MACOS else "libedgetpu.so.1"

# Índices de salida del modelo SSD MobileNet v2 con post-procesado
_IDX_BOXES   = 0
_IDX_CLASSES = 1
_IDX_SCORES  = 2
_IDX_COUNT   = 3


@dataclass
class Detection:
    class_id: int
    label: str
    score: float
    # Normalizado (ymin, xmin, ymax, xmax) ∈ [0, 1]
    bbox: tuple[float, float, float, float]

    def bbox_pixels(self, image_w: int, image_h: int) -> tuple[int, int, int, int]:
        """Devuelve (x1, y1, x2, y2) en coordenadas de píxel."""
        ymin, xmin, ymax, xmax = self.bbox
        return (
            int(xmin * image_w), int(ymin * image_h),
            int(xmax * image_w), int(ymax * image_h),
        )


# ---------------------------------------------------------------------------
# Backend ONNX + CoreML  (macOS, opcional)
# ---------------------------------------------------------------------------

def _try_onnx_coreml(onnx_path: Path, min_confidence: float) -> "_OnnxDetector | None":
    """Intenta crear un detector ONNX con CoreML EP. Devuelve None si falla."""
    if not onnx_path.exists():
        return None
    try:
        import onnxruntime as ort  # type: ignore
        available_providers = ort.get_available_providers()
        log.info("ONNX Runtime disponible. Providers: %s", available_providers)

        # Prioridad: ANE/GPU via CoreML → GPU via DirectML → CPU
        preferred = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
        providers = [p for p in preferred if p in available_providers]
        if not providers:
            providers = ["CPUExecutionProvider"]

        session = ort.InferenceSession(str(onnx_path), providers=providers)
        active = session.get_providers()[0]
        log.info(
            "ONNX Runtime activo con provider: %s %s",
            active,
            "← ✓ ANE/GPU (Core ML)" if "CoreML" in active else "← CPU",
        )
        return _OnnxDetector(session, min_confidence)
    except ImportError:
        log.debug("onnxruntime no instalado — usar 'pip install onnxruntime'")
        return None
    except Exception as exc:
        log.warning("No se pudo inicializar ONNX Runtime: %s", exc)
        return None


class _OnnxDetector:
    """Detector usando ONNX Runtime (Core ML EP en Mac = ANE/GPU)."""

    def __init__(self, session, confidence_fn) -> None:
        self._session = session
        self._confidence_fn = confidence_fn
        # Dimensiones de entrada del modelo ONNX
        inp = session.get_inputs()[0]
        # Shape típica: [1, 300, 300, 3]
        self._input_h: int = int(inp.shape[1]) if inp.shape[1] else 300
        self._input_w: int = int(inp.shape[2]) if inp.shape[2] else 300
        self._input_name: str = inp.name
        log.info(
            "Modelo ONNX: entrada %dx%d, nombre='%s'",
            self._input_w, self._input_h, self._input_name,
        )

    @property
    def input_size(self) -> tuple[int, int]:
        return self._input_w, self._input_h

    def detect(self, frame: np.ndarray, labels: dict[int, str]) -> list[Detection]:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self._input_w, self._input_h))
        tensor = np.expand_dims(resized, axis=0).astype(np.uint8)

        outputs = self._session.run(None, {self._input_name: tensor})

        boxes   = outputs[_IDX_BOXES][0]
        classes = outputs[_IDX_CLASSES][0]
        scores  = outputs[_IDX_SCORES][0]
        count   = int(outputs[_IDX_COUNT][0])

        return _parse_ssd_outputs(boxes, classes, scores, count, self._confidence_fn, labels)


# ---------------------------------------------------------------------------
# Backend TFLite  (Linux + Coral, o Mac en CPU)
# ---------------------------------------------------------------------------

class _TFLiteDetector:
    """Detector usando TFLite/LiteRT — Edge TPU en Linux, XNNPACK en Mac."""

    def __init__(
        self,
        model_path: Path,
        confidence_fn,
        use_tpu: bool,
        _delegates: list | None = None,  # permite inyectar delegate externo
    ) -> None:
        self._confidence_fn = confidence_fn

        delegates = _delegates or []
        if not delegates and use_tpu and not _IS_MACOS:
            # Edge TPU USB-C (Linux)
            try:
                delegates = [_tflite_load_delegate(EDGETPU_SHARED_LIB)]
                log.info("Edge TPU USB-C delegate cargado")
            except (ValueError, OSError) as exc:
                log.warning(
                    "Edge TPU no disponible (%s) — usando CPU. "
                    "Verifica que libedgetpu1-std está instalado y el Coral conectado.", exc,
                )

        # Número de threads XNNPACK:
        # En macOS (M-series) usar la mitad de los núcleos de rendimiento (P-cores).
        # M1 Pro tiene 8 P-cores + 2 E-cores. Usar 6 P-cores es un buen balance.
        # En Linux sin Coral, usar todos los cores disponibles.
        num_threads = _get_num_threads(use_tpu)

        self._interpreter = _TFLiteInterpreter(
            model_path=str(model_path),
            experimental_delegates=delegates,
            num_threads=num_threads,
        )
        self._interpreter.allocate_tensors()

        self._input_details  = self._interpreter.get_input_details()
        self._output_details = self._interpreter.get_output_details()

        shape = self._input_details[0]["shape"]
        self._input_h: int = int(shape[1])
        self._input_w: int = int(shape[2])

        log.info(
            "TFLite (%s): entrada %dx%d, %d threads XNNPACK, delegates=%s",
            _TFLITE_RUNTIME,
            self._input_w, self._input_h,
            num_threads,
            "EdgeTPU" if delegates else "ninguno (CPU/XNNPACK)",
        )

    @property
    def input_size(self) -> tuple[int, int]:
        return self._input_w, self._input_h

    def detect(self, frame: np.ndarray, labels: dict[int, str]) -> list[Detection]:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self._input_w, self._input_h))
        tensor = np.expand_dims(resized, axis=0).astype(np.uint8)

        self._interpreter.set_tensor(self._input_details[0]["index"], tensor)
        self._interpreter.invoke()

        boxes   = self._interpreter.get_tensor(self._output_details[_IDX_BOXES]["index"])[0]
        classes = self._interpreter.get_tensor(self._output_details[_IDX_CLASSES]["index"])[0]
        scores  = self._interpreter.get_tensor(self._output_details[_IDX_SCORES]["index"])[0]
        count   = int(self._interpreter.get_tensor(self._output_details[_IDX_COUNT]["index"])[0])

        return _parse_ssd_outputs(boxes, classes, scores, count, self._confidence_fn, labels)


# ---------------------------------------------------------------------------
# Helpers comunes
# ---------------------------------------------------------------------------

def _get_num_threads(use_tpu: bool) -> int:
    """Número de threads óptimo para XNNPACK según la plataforma."""
    if use_tpu and not _IS_MACOS:
        return 1  # El Edge TPU ya es el acelerador, no necesitamos threads extra
    total = os.cpu_count() or 4
    # Usar la mitad de los cores: deja margen para el sistema y el resto del proceso.
    # En M1 Pro (10 cores) → 5 threads. En M1 Max (10 cores) → 5. En RPi4 (4) → 2.
    return max(2, total // 2)


def _parse_ssd_outputs(
    boxes: np.ndarray,
    classes: np.ndarray,
    scores: np.ndarray,
    count: int,
    confidence_fn,   # callable(label: str) -> float
    labels: dict[int, str],
) -> list[Detection]:
    detections: list[Detection] = []
    for i in range(count):
        score    = float(scores[i])
        class_id = int(classes[i])
        label    = labels.get(class_id, f"unknown_{class_id}")
        if score < confidence_fn(label):
            continue
        ymin, xmin, ymax, xmax = (
            float(boxes[i][0]), float(boxes[i][1]),
            float(boxes[i][2]), float(boxes[i][3]),
        )
        bbox = (
            max(0.0, min(1.0, ymin)), max(0.0, min(1.0, xmin)),
            max(0.0, min(1.0, ymax)), max(0.0, min(1.0, xmax)),
        )
        detections.append(Detection(class_id=class_id, label=label, score=score, bbox=bbox))
    return detections


def _load_labels(labels_path: Path) -> dict[int, str]:
    labels: dict[int, str] = {}
    if not labels_path.exists():
        log.warning("Archivo de etiquetas no encontrado: %s", labels_path)
        return labels
    with labels_path.open() as fh:
        for line_no, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            parts = line.split(None, 1)
            if len(parts) == 2:
                try:
                    labels[int(parts[0])] = parts[1]
                except ValueError:
                    labels[line_no] = line
            else:
                labels[line_no] = line
    log.debug("Etiquetas cargadas: %d", len(labels))
    return labels


# ---------------------------------------------------------------------------
# Backend YOLOv8 (macOS testing — mejor detección de objetos pequeños)
# ---------------------------------------------------------------------------

# Mapa COCO index → label para YOLOv8 (mismos IDs que SSD MobileNet COCO)
_YOLO_COCO = {0: "person", 15: "cat", 16: "dog"}


def _try_tpu_macos(model_path: Path, confidence_fn) -> "_TFLiteDetector | None":
    """Intenta crear un detector con Edge TPU en macOS vía libedgetpu.1.dylib."""
    try:
        delegate = _tflite_load_delegate(EDGETPU_SHARED_LIB)
        backend = _TFLiteDetector(model_path, confidence_fn, use_tpu=True,
                                   _delegates=[delegate])
        log.info("Coral Edge TPU USB-C disponible en macOS — usando TFLite+EdgeTPU")
        return backend
    except Exception as exc:
        log.debug("Coral no disponible en macOS: %s", exc)
        return None


def _try_yolo(confidence_fn) -> "_YOLODetector | None":
    """Intenta crear un detector YOLOv8-nano. Devuelve None si falla."""
    try:
        import ssl
        # Fix certificados SSL en macOS (necesario para descargar el modelo)
        ssl._create_default_https_context = ssl._create_unverified_context  # noqa: SLF001
        from ultralytics import YOLO  # type: ignore[import]
        model = YOLO("yolov8n.pt")
        log.info("YOLOv8-nano cargado — input 640x640, mejor deteccion de objetos pequeños")
        return _YOLODetector(model, confidence_fn)
    except ImportError:
        log.debug("ultralytics no instalado — saltando YOLOv8")
        return None
    except Exception as exc:
        log.warning("No se pudo cargar YOLOv8: %s", exc)
        return None


class _YOLODetector:
    """Backend YOLOv8-nano via ultralytics. Solo para Mac/testing."""

    def __init__(self, model, confidence_fn) -> None:
        self._model = model
        self._confidence_fn = confidence_fn
        self._input_w = 640
        self._input_h = 640

    @property
    def input_size(self) -> tuple[int, int]:
        return self._input_w, self._input_h

    def detect(self, frame: np.ndarray, labels: dict) -> list[Detection]:
        # YOLOv8 acepta BGR directamente (mismo formato que OpenCV)
        results = self._model(frame, verbose=False, conf=0.1)  # umbral bajo; filtramos abajo
        detections: list[Detection] = []

        if not results or results[0].boxes is None:
            return detections

        boxes = results[0].boxes
        for i in range(len(boxes)):
            cls_id   = int(boxes.cls[i].item())
            score    = float(boxes.conf[i].item())
            label    = _YOLO_COCO.get(cls_id)
            if label is None:
                continue
            if score < self._confidence_fn(label):
                continue

            # xyxyn = (x1, y1, x2, y2) normalizado
            x1, y1, x2, y2 = boxes.xyxyn[i].tolist()
            bbox = (
                max(0.0, min(1.0, y1)),
                max(0.0, min(1.0, x1)),
                max(0.0, min(1.0, y2)),
                max(0.0, min(1.0, x2)),
            )
            detections.append(Detection(class_id=cls_id, label=label, score=score, bbox=bbox))

        return detections


# ---------------------------------------------------------------------------
# Clase pública — punto de entrada único
# ---------------------------------------------------------------------------

class Detector:
    """Detector con selección automática de backend por plataforma.

    macOS  → YOLOv8-nano (mejor detección objetos pequeños, 31 fps en M1)
    Linux  → TFLite + Edge TPU Coral / XNNPACK fallback
    """

    def __init__(
        self,
        model_path: Path,
        labels_path: Path,
        min_confidence: float = 0.5,
        class_min_confidence: "dict[str, float] | None" = None,
        use_tpu: bool = True,
    ) -> None:
        self._labels = _load_labels(labels_path)
        self._backend: "_OnnxDetector | _TFLiteDetector"

        # Callable que devuelve el umbral correcto para cada etiqueta.
        # Permite umbrales distintos por clase (p.ej. perro pequeño = 0.15).
        _class_conf = class_min_confidence or {}
        confidence_fn = lambda label: _class_conf.get(label, min_confidence)  # noqa: E731

        if _IS_MACOS:
            # Prioridad macOS: YOLOv8 > ONNX+CoreML > TFLite+XNNPACK
            # Nota: Edge TPU en macOS tiene incompatibilidades de runtime con los
            # modelos compilados en 2024+. En producción (Linux) el Coral funciona
            # perfectamente via el path TFLite + EdgeTPU delegate.
            yolo = _try_yolo(confidence_fn)
            if yolo:
                self._backend = yolo
                self._input_size = yolo.input_size
            else:
                onnx_path = model_path.with_suffix(".onnx")
                onnx_backend = _try_onnx_coreml(onnx_path, confidence_fn)
                if onnx_backend:
                    self._backend = onnx_backend
                    self._input_size = onnx_backend.input_size
                    log.info("Backend activo: ONNX Runtime + Core ML (ANE/GPU)")
                else:
                    log.info("Backend activo: TFLite XNNPACK multi-thread (ARM NEON)")
                    self._backend = _TFLiteDetector(model_path, confidence_fn, use_tpu=False)
                    self._input_size = self._backend.input_size
        else:
            self._backend = _TFLiteDetector(model_path, confidence_fn, use_tpu=use_tpu)
            self._input_size = self._backend.input_size

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Ejecuta inferencia en un frame BGR (formato OpenCV)."""
        return self._backend.detect(frame, self._labels)

    @property
    def input_size(self) -> tuple[int, int]:
        """(width, height) que espera el modelo."""
        return self._input_size
