"""Dependency-free zone analytics for tracked object observations.

Only tracked detections generate identity events. Untracked detections still
contribute to instantaneous occupancy, but never to unique visitor counts.
"""

from dataclasses import dataclass, field
import math


@dataclass(frozen=True)
class Zone:
    name: str
    x1: float
    y1: float
    x2: float
    y2: float

    @classmethod
    def parse(cls, spec: str) -> "Zone":
        """Parse NAME:X1,Y1,X2,Y2 in normalized image coordinates."""
        try:
            name, coordinates = spec.split(":", 1)
            values = [float(part.strip()) for part in coordinates.split(",")]
        except (ValueError, AttributeError) as exc:
            raise ValueError(f"Invalid zone {spec!r}. Expected name:x1,y1,x2,y2") from exc
        if not name.strip() or len(values) != 4 or not all(math.isfinite(v) for v in values):
            raise ValueError("Zone needs a name and four finite coordinates")
        x1, y1, x2, y2 = values
        if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
            raise ValueError("Zone coordinates must be in [0, 1], with x1 < x2 and y1 < y2")
        return cls(name.strip(), *values)

    def contains(self, box: tuple[float, float, float, float], size: tuple[int, int]) -> bool:
        """Use bottom-centre (ground contact) for a more stable zone crossing."""
        width, height = size
        if width <= 0 or height <= 0:
            raise ValueError("Frame dimensions must be positive")
        x1, _, x2, y2 = box
        x, y = ((x1 + x2) / 2 / width, y2 / height)
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2


@dataclass(frozen=True)
class Observation:
    class_name: str
    confidence: float
    box: tuple[float, float, float, float]
    track_id: int | None = None


@dataclass
class _Presence:
    first_seen: float
    last_seen: float
    zones: dict[str, float] = field(default_factory=dict)
    dwell_sent: set[str] = field(default_factory=set)
    last_box: tuple[float, float, float, float] = (0, 0, 0, 0)
    confidence: float = 0.0


class ZoneMonitor:
    """Compute occupancy and deduplicated entry, exit and dwell events.

    The caller provides monotonically increasing media-relative timestamps in
    seconds. Lost tracks exit after a grace period rather than immediately.
    """

    def __init__(self, zones: list[Zone], dwell_seconds: float = 0, lost_timeout: float = 1):
        if not zones or len({z.name for z in zones}) != len(zones):
            raise ValueError("At least one zone with a unique name is required")
        if dwell_seconds < 0 or not math.isfinite(dwell_seconds):
            raise ValueError("dwell_seconds must be non-negative and finite")
        if lost_timeout < 0 or not math.isfinite(lost_timeout):
            raise ValueError("lost_timeout must be non-negative and finite")
        self.zones = zones
        self.dwell_seconds = dwell_seconds
        self.lost_timeout = lost_timeout
        self._tracks: dict[tuple[str, int], _Presence] = {}
        self._last_time = -1.0
        self.totals = {"enter": 0, "exit": 0, "dwell": 0}
        self.untracked_observations = 0

    def _event(self, kind: str, zone: str, key: tuple[str, int], state: _Presence,
               seconds: float, frame: int, reason: str | None = None) -> dict:
        self.totals[kind] += 1
        event = {"type": kind, "zone": zone, "class_name": key[0], "track_id": key[1],
                 "time_s": round(seconds, 3), "frame": frame,
                 "box": list(state.last_box), "confidence": round(state.confidence, 4)}
        if reason:
            event["reason"] = reason
        if kind in ("exit", "dwell"):
            event["duration_s"] = round(max(0, seconds - state.zones[zone]), 3)
        return event

    def update(self, observations: list[Observation], seconds: float, frame: int,
               size: tuple[int, int]) -> tuple[list[dict], dict[tuple[str, str], int]]:
        if not math.isfinite(seconds) or seconds < self._last_time:
            raise ValueError("Frame timestamps must be finite and nondecreasing")
        self._last_time = seconds
        events: list[dict] = []
        occupancy: dict[tuple[str, str], int] = {}
        seen: set[tuple[str, int]] = set()

        # Avoid double-counting duplicated tracker output in the same frame.
        unique: dict[tuple[str, int], Observation] = {}
        anonymous: list[Observation] = []
        for obj in observations:
            if obj.track_id is None:
                anonymous.append(obj)
            else:
                key = (obj.class_name, obj.track_id)
                if key not in unique or unique[key].confidence < obj.confidence:
                    unique[key] = obj
        self.untracked_observations += len(anonymous)

        for obj in [*unique.values(), *anonymous]:
            inside = [z.name for z in self.zones if z.contains(obj.box, size)]
            for zone in inside:
                count_key = (zone, obj.class_name)
                occupancy[count_key] = occupancy.get(count_key, 0) + 1
            if obj.track_id is None:
                continue
            key = (obj.class_name, obj.track_id)
            seen.add(key)
            state = self._tracks.get(key)
            if state is not None and seconds - state.last_seen > self.lost_timeout:
                for zone in list(state.zones):
                    events.append(self._event("exit", zone, key, state, state.last_seen, frame, "lost"))
                del self._tracks[key]
                state = None
            if state is None:
                state = _Presence(first_seen=seconds, last_seen=seconds)
                self._tracks[key] = state
            state.last_seen = seconds
            state.last_box = obj.box
            state.confidence = obj.confidence
            for zone in list(state.zones):
                if zone not in inside:
                    events.append(self._event("exit", zone, key, state, seconds, frame))
                    del state.zones[zone]
                    state.dwell_sent.discard(zone)
            for zone in inside:
                if zone not in state.zones:
                    state.zones[zone] = seconds
                    events.append(self._event("enter", zone, key, state, seconds, frame))
                if (self.dwell_seconds > 0 and zone not in state.dwell_sent
                        and seconds - state.zones[zone] >= self.dwell_seconds):
                    events.append(self._event("dwell", zone, key, state, seconds, frame))
                    state.dwell_sent.add(zone)

        for key, state in list(self._tracks.items()):
            if key not in seen and seconds - state.last_seen > self.lost_timeout:
                for zone in list(state.zones):
                    events.append(self._event("exit", zone, key, state, state.last_seen, frame, "lost"))
                del self._tracks[key]
        return events, occupancy

    def finish(self, seconds: float, frame: int) -> list[dict]:
        """Close remaining visits when processing stops; no phantom late dwell events."""
        events = []
        for key, state in list(self._tracks.items()):
            for zone in state.zones:
                events.append(self._event("exit", zone, key, state, min(seconds, state.last_seen),
                                          frame, "end_of_stream"))
        self._tracks.clear()
        return events
