# syntax=docker/dockerfile:1.7

# =============================================================================
# FamilyCentinel - Local presence detector
# Multi-arch image (linux/amd64, linux/arm64) for x86_64 hosts and Raspberry Pi 5
# Uses Coral Edge TPU USB-C (MA2485) for TFLite inference
# =============================================================================

ARG BUILDPLATFORM
ARG TARGETPLATFORM

# -----------------------------------------------------------------------------
# Stage 1: builder — fetch model + labels and build wheels
# -----------------------------------------------------------------------------
FROM --platform=${TARGETPLATFORM} python:3.11-slim-bookworm AS builder

ARG TARGETPLATFORM
ARG BUILDPLATFORM

# Build-time tools (not propagated to the final image)
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# Pre-fetch the Edge TPU model and the COCO label map. Doing this in the
# builder stage keeps the layer cacheable and avoids shipping curl in the
# final image.
WORKDIR /build/models
RUN curl -fsSL -o ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite \
        https://raw.githubusercontent.com/google-coral/test_data/master/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite \
 && curl -fsSL -o coco_labels.txt \
        https://raw.githubusercontent.com/google-coral/test_data/master/coco_labels.txt

# Build wheels for all Python deps so the final stage is a clean install.
WORKDIR /build
COPY requirements.txt /build/requirements.txt
RUN pip wheel --no-cache-dir --wheel-dir /build/wheels -r requirements.txt


# -----------------------------------------------------------------------------
# Stage 2: runtime — minimal image with libedgetpu and the application
# -----------------------------------------------------------------------------
FROM --platform=${TARGETPLATFORM} python:3.11-slim-bookworm AS runtime

ARG TARGETPLATFORM

LABEL org.opencontainers.image.title="FamilyCentinel" \
      org.opencontainers.image.description="Local presence detector with Coral Edge TPU + RTSP + MQTT" \
      org.opencontainers.image.source="https://github.com/jampgold/FamilyCentinel" \
      org.opencontainers.image.licenses="MIT"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# -----------------------------------------------------------------------------
# System dependencies
#  - libglib2.0-0 / libsm6 / libxext6 / libxrender-dev / libgl1: OpenCV runtime
#  - ffmpeg: H.264/H.265 RTSP decoding fallback
#  - udev: USB device hotplug detection (Coral USB-C)
#  - curl + gnupg: needed only to register the Coral apt repo (purged below)
# -----------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        gnupg \
        udev \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libgl1 \
        ffmpeg \
    && \
    # -------------------------------------------------------------------------
    # Register the official Google Coral apt repository.
    #
    # NOTE: For Raspberry Pi 5 or some ARM distros where the standard runtime
    # is unstable, swap `libedgetpu1-std` for `libedgetpu1-max`. The "max"
    # variant runs the TPU at a higher clock (warmer chip, slightly more
    # current draw on USB) and is sometimes required for reliable enumeration
    # on Pi 5 with the MA2485 dongle.
    # -------------------------------------------------------------------------
    curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg \
        | gpg --dearmor -o /usr/share/keyrings/coral-edgetpu-archive-keyring.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/coral-edgetpu-archive-keyring.gpg] https://packages.cloud.google.com/apt coral-edgetpu-stable main" \
        > /etc/apt/sources.list.d/coral-edgetpu.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends libedgetpu1-std && \
    # Trim the image: drop apt lists and tooling only needed for repo setup
    apt-get purge -y --auto-remove gnupg && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# -----------------------------------------------------------------------------
# Python dependencies (installed from pre-built wheels)
# -----------------------------------------------------------------------------
COPY --from=builder /build/wheels /tmp/wheels
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --no-index --find-links=/tmp/wheels -r /tmp/requirements.txt \
 && rm -rf /tmp/wheels /tmp/requirements.txt

# -----------------------------------------------------------------------------
# Application layout
# -----------------------------------------------------------------------------
WORKDIR /app

# Models fetched in the builder stage
COPY --from=builder /build/models /app/models

# Application source
COPY src/ /app/src/

# Config directory (mount config.yaml here at runtime)
RUN mkdir -p /app/config

# -----------------------------------------------------------------------------
# Non-root user with USB access
#
# `plugdev` is the conventional group for hot-pluggable devices on Debian.
# Adding `appuser` to it lets the container talk to the Coral USB-C device
# exposed via /dev/bus/usb without needing --privileged.
# -----------------------------------------------------------------------------
RUN groupadd --system --gid 46 plugdev 2>/dev/null || true && \
    useradd --system --create-home --uid 1000 --gid plugdev --shell /usr/sbin/nologin appuser && \
    chown -R appuser:plugdev /app

USER appuser

# -----------------------------------------------------------------------------
# Healthcheck
#
# The application is expected to `touch /tmp/healthcheck` on every successful
# inference loop iteration. If the file is missing or older than 120 seconds,
# the container is reported as unhealthy.
# -----------------------------------------------------------------------------
HEALTHCHECK --interval=60s --timeout=10s --start-period=30s --retries=3 \
    CMD test -f /tmp/healthcheck && \
        [ $(( $(date +%s) - $(stat -c %Y /tmp/healthcheck) )) -lt 120 ] || exit 1

ENTRYPOINT ["python", "-m", "src.main"]
