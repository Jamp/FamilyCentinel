PYTHON   = python3.11
VENV     = .venv
PIP      = $(VENV)/bin/pip
PYTHON_V = $(VENV)/bin/python
CONFIG   = config.local.yaml

.PHONY: help setup models onnx debug monitor calibrate benchmark run clean

help:
	@echo ""
	@echo "  FamilyCentinel - comandos disponibles"
	@echo ""
	@echo "  make setup      Crear .venv e instalar dependencias"
	@echo "  make models     Descargar modelos Edge TPU a ./models/"
	@echo "  make onnx       Convertir TFLite a ONNX para ANE/GPU en Mac"
	@echo "  make debug      Stream MJPEG con detecciones -> http://localhost:8080"
	@echo "  make monitor    Monitor MQTT en consola (Ctrl+C para salir)"
	@echo "  make calibrate  Capturar frame con bboxes para calibrar umbrales"
	@echo "  make benchmark  Medir latencia de inferencia en esta maquina"
	@echo "  make run        Arrancar el servicio principal"
	@echo "  make clean      Borrar .venv y caches"
	@echo ""
	@echo "  Config: $(CONFIG)"
	@echo "  Inicio rapido Mac: make setup && make models && make onnx && make debug"
	@echo ""

$(VENV)/bin/activate:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip --quiet
	$(PIP) install -r requirements-dev.txt

setup: $(VENV)/bin/activate

models/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite:
	bash tools/download_models.sh

models: models/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite

onnx: setup models
	$(PYTHON_V) tools/convert_to_onnx.py

debug: setup models
	$(PYTHON_V) tools/debug_stream.py --config $(CONFIG) --port 8080 --fps 5

monitor: setup
	$(PYTHON_V) tools/mqtt_monitor.py --config $(CONFIG)

calibrate: setup models
	$(PYTHON_V) tools/calibrate.py --config $(CONFIG) --output calibration_frame.jpg

benchmark: setup models
	$(PYTHON_V) tools/benchmark.py --config $(CONFIG)

run: setup models
	CONFIG_PATH=$(CONFIG) $(PYTHON_V) -m src.main

clean:
	rm -rf $(VENV) calibration_frame.jpg
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
