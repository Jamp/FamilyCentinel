#!/usr/bin/env bash
# Descarga los modelos necesarios.
# - ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite  -> produccion (Coral)
# - ssd_mobilenet_v2_coco_quant_postprocess.tflite          -> Mac/CPU testing
set -euo pipefail

DIR="$(cd "$(dirname "$0")/.." && pwd)/models"
mkdir -p "$DIR"

BASE="https://raw.githubusercontent.com/google-coral/test_data/master"

download() {
    local file="$DIR/$1"
    if [ -f "$file" ]; then
        echo "  ya existe: $1"
    else
        echo "  descargando: $1"
        curl -fsSL -o "$file" "$BASE/$1"
    fi
}

download "ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite"
download "ssd_mobilenet_v2_coco_quant_postprocess.tflite"
download "coco_labels.txt"
echo "Modelos listos en: $DIR"
