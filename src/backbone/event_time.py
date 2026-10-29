"""Resolve BAS event times to causal samples on a declared source frame grid."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation, ROUND_FLOOR, ROUND_HALF_UP
from typing import Any


@dataclass(frozen=True)
class EventTimeResolution:
    status: str
    reason: str | None
    position_ms: str | None
    native_frame_rate_hz: int | None
    native_zero_based_frame: int | None
    nearest_native_frame: int | None
    stable_event_frame: int | None
    stable_frame_delta_from_nearest: int | None
    causal_query_timestamp_ms: str | None
    timing_error_ms: str | None
    logical_model_grid_hz: int | None
    physical_contact_certified: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def resolve_event_time(
    position_ms: object,
    stable_event_frame: object,
    *,
    clock: object,
    gsr_info: dict,
    source_frame_count: int,
) -> EventTimeResolution:
    """Map an event timestamp to the last source frame not later than it.

    BAS ``frame`` is retained as an independent consistency check, not as
    authority for selecting a potentially future observation. The model
    grid is 25 Hz only when confirmed by the source GSR metadata.
    """
    frame_rate = gsr_info.get("frame_rate")
    try:
        fps_decimal = Decimal(str(frame_rate))
        fps = int(fps_decimal)
        if fps_decimal != fps or fps <= 0:
            raise ValueError
    except (InvalidOperation, TypeError, ValueError, OverflowError):
        return EventTimeResolution(
            "unsupported", "invalid_source_frame_rate", None, None, None,
            None, None, None, None, None, None,
        )
    if fps != 25:
        return EventTimeResolution(
            "unsupported", "unsupported_native_frame_rate", None, fps, None,
            None, None, None, None, None, fps,
        )
    if str(clock) != "ms from the first frame of the half's released video":
        return EventTimeResolution(
            "unsupported", "unsupported_event_clock_declaration", None, fps, None,
            None, None, None, None, None, fps,
        )
    try:
        clip_start = Decimal(str(gsr_info["clip_start"]))
        if clip_start != 0:
            return EventTimeResolution(
                "unsupported", "nonzero_source_clip_origin_not_resolved", None, fps, None,
                None, None, None, None, None, fps,
            )
        position = Decimal(str(position_ms))
        stable_frame = int(stable_event_frame)
        if not position.is_finite() or position < 0:
            raise ValueError
    except (KeyError, InvalidOperation, TypeError, ValueError, OverflowError):
        return EventTimeResolution(
            "unsupported", "invalid_event_timestamp_or_stable_frame", None, fps, None,
            None, None, None, None, None, fps,
        )
    frame_position = position * Decimal(fps) / Decimal(1000)
    floor_frame = int(frame_position.to_integral_value(rounding=ROUND_FLOOR))
    nearest_frame = int(frame_position.to_integral_value(rounding=ROUND_HALF_UP))
    stable_delta = stable_frame - nearest_frame
    if floor_frame >= source_frame_count:
        return EventTimeResolution(
            "unsupported", "event_after_source_frame_range", str(position), fps, floor_frame,
            nearest_frame, stable_frame, stable_delta, None, None, fps,
        )
    # The event identity's frame follows the dataset's rounded annotation
    # convention, with the observed 0/1 index-origin ambiguity. The model
    # query remains strictly floor-rounded and can never use a future frame.
    if stable_delta not in (0, 1):
        return EventTimeResolution(
            "unsupported", "stable_frame_inconsistent_with_rounded_source_time",
            str(position), fps, floor_frame, nearest_frame, stable_frame, stable_delta,
            None, None, fps,
        )
    causal_time = Decimal(floor_frame * 1000) / Decimal(fps)
    error = position - causal_time
    if error < 0 or error >= Decimal(1000) / Decimal(fps):
        raise AssertionError("Causal event-frame rounding must have error in [0, one frame)")
    return EventTimeResolution(
        "supported", None, str(position), fps, floor_frame, nearest_frame,
        stable_frame, stable_delta, str(causal_time), str(error), fps,
    )
