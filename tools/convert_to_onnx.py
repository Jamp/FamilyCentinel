"""Convierte el modelo TFLite Edge TPU a formato ONNX para usar el ANE del Mac.

El flujo es:
    .tflite  →  (tf2onnx)  →  .onnx  →  onnxruntime + CoreML EP  →  ANE/GPU

Tras la conversión, `detector.py` detecta automáticamente el archivo .onnx
y usa ONNX Runtime con Core ML Execution Provider (ANE en M-series).

REQUISITOS:
    pip install tf2onnx onnxruntime
    (ya incluidos en requirements-dev.txt)

USO:
    python tools/convert_to_onnx.py                         # usa rutas por defecto
    python tools/convert_to_onnx.py --model models/custom.tflite

    O via Makefile:
    make onnx
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

_DEFAULT_TFLITE = Path("models/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite")


def convert(tflite_path: Path, onnx_path: Path) -> None:
    if not tflite_path.exists():
        log.error("Modelo TFLite no encontrado: %s", tflite_path)
        log.error("Ejecuta primero: make models  (o bash tools/download_models.sh)")
        sys.exit(1)

    if onnx_path.exists():
        log.info("El archivo ONNX ya existe: %s", onnx_path)
        log.info("Borrarlo para reconvertir.")
        return

    # Verificar que tf2onnx está disponible
    try:
        import tf2onnx  # noqa: F401
    except ImportError:
        log.error("tf2onnx no está instalado.")
        log.error("Instalar: pip install tf2onnx")
        sys.exit(1)

    log.info("Convirtiendo %s → %s …", tflite_path.name, onnx_path.name)
    log.info("(Puede tardar 30-60 segundos la primera vez)")

    # tf2onnx puede convertir TFLite directamente:
    #   python -m tf2onnx.convert --tflite <input> --output <output> --opset 13
    cmd = [
        sys.executable, "-m", "tf2onnx.convert",
        "--tflite", str(tflite_path),
        "--output", str(onnx_path),
        "--opset", "13",
    ]
    log.info("Ejecutando: %s", " ".join(cmd))

    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        log.error("La conversión falló (código %d).", result.returncode)
        sys.exit(1)

    log.info("")
    log.info("✓ Conversión exitosa: %s", onnx_path)
    log.info("")
    log.info("El detector usará ONNX + Core ML automáticamente.")
    log.info("Prueba con: make debug   o   make run")
    _verify_onnx(onnx_path)


def _verify_onnx(onnx_path: Path) -> None:
    try:
        import onnxruntime as ort  # type: ignore
        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        inp = session.get_inputs()[0]
        outs = session.get_outputs()
        log.info("Verificación ONNX:")
        log.info("  Entrada:  %s %s %s", inp.name, inp.shape, inp.type)
        for o in outs:
            log.info("  Salida:   %s %s %s", o.name, o.shape, o.type)

        providers = ort.get_available_providers()
        has_coreml = "CoreMLExecutionProvider" in providers
        log.info("")
        log.info("Core ML EP disponible: %s", "✓ SÍ (usará ANE/GPU)" if has_coreml else "✗ NO")
        if not has_coreml:
            log.warning(
                "Core ML EP no disponible. "
                "Instala onnxruntime>=1.18 en Apple Silicon para tenerlo."
            )
    except ImportError:
        log.info("(onnxruntime no instalado — instalar para verificar: pip install onnxruntime)")
    except Exception as exc:
        log.warning("Verificación falló: %s", exc)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convierte TFLite a ONNX para ANE en Mac")
    parser.add_argument(
        "--model",
        type=Path,
        default=_DEFAULT_TFLITE,
        help="Ruta al modelo .tflite (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Ruta de salida .onnx (default: mismo directorio que el .tflite)",
    )
    args = parser.parse_args()

    tflite_path: Path = args.model
    onnx_path: Path = args.output or tflite_path.with_suffix(".onnx")

    convert(tflite_path, onnx_path)


if __name__ == "__main__":
    main()
