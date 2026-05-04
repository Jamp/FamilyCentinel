"""Canonical entity names used across stabilizer, MQTT and classifier.

Single source of truth — avoids silent drift when the set is updated.
"""
ENTITIES: tuple[str, ...] = ("person", "dog")
