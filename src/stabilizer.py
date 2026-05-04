"""Temporal debounce for presence state transitions.

Avoids chattering in Home Assistant by requiring N consecutive positive
detections to flip to "present" and M consecutive missed frames to flip
to "absent".

Protección anti-falso-ausente (caso: persona durmiendo bajo una cobija):
    Una vez marcada como PRESENTE, una entidad no puede marcarse AUSENTE
    hasta que hayan pasado al menos `min_presence_seconds` desde la última
    detección. Esto evita que una persona quieta o cubierta sea considerada
    ausente solo porque el modelo no la detecta temporalmente.

    Valor recomendado: 600 segundos (10 minutos). Una persona dormida se
    mueve al menos una vez cada 10 minutos, lo que reinicia el contador.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from src.entities import ENTITIES

log = logging.getLogger(__name__)

_NEVER: float = 0.0


@dataclass
class _EntityState:
    present: bool = False
    consecutive_hits: int = 0
    consecutive_misses: int = 0
    last_detected_at: float = field(default_factory=lambda: _NEVER)


class Stabilizer:
    def __init__(
        self,
        frames_to_activate: int,
        frames_to_deactivate: int,
        min_presence_seconds: float = 0.0,
    ) -> None:
        self._activate = frames_to_activate
        self._deactivate = frames_to_deactivate
        self._min_presence_s = min_presence_seconds
        self._states: dict[str, _EntityState] = {e: _EntityState() for e in ENTITIES}

    def update(self, detections: set[str]) -> dict[str, bool]:
        """
        Accept the set of entity labels detected in the current frame and
        return a mapping of entity → new_state only for entities whose
        stable state *changed*.
        """
        changes: dict[str, bool] = {}
        now = time.monotonic()

        for entity, state in self._states.items():
            detected = entity in detections

            if detected:
                state.consecutive_hits += 1
                state.consecutive_misses = 0
                state.last_detected_at = now
            else:
                state.consecutive_misses += 1
                state.consecutive_hits = 0

            if not state.present and state.consecutive_hits >= self._activate:
                state.present = True
                state.last_detected_at = now
                log.info("Entity '%s' → PRESENT", entity)
                changes[entity] = True

            elif state.present and state.consecutive_misses >= self._deactivate:
                # Solo marcar ausente si ya pasó el tiempo mínimo de presencia.
                elapsed = now - state.last_detected_at
                if self._min_presence_s > 0 and elapsed < self._min_presence_s:
                    log.debug(
                        "Entity '%s': %d misses but min_presence not elapsed "
                        "(%.0fs / %.0fs) — staying PRESENT",
                        entity, state.consecutive_misses, elapsed, self._min_presence_s,
                    )
                else:
                    state.present = False
                    log.info(
                        "Entity '%s' → ABSENT (%.0fs since last detection)",
                        entity, elapsed,
                    )
                    changes[entity] = False

        return changes

    def current_states(self) -> dict[str, bool]:
        return {e: s.present for e, s in self._states.items()}
