"""Pace gameplay images without waiting for the viewer or queuing snapshots."""

import os
import threading
import time


def bounded_setting(name, default, minimum, maximum):
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        return default
    return max(minimum, min(maximum, value))


class FrameDelivery:
    def __init__(self, emit, *, require_ack=True, clock=time.monotonic):
        self.emit = emit
        self.clock = clock
        self.require_ack = require_ack
        self.interval = 1 / bounded_setting("STREAM_FPS", 30, 1, 60)
        self.max_in_flight = bounded_setting("STREAM_MAX_IN_FLIGHT", 3, 1, 6)
        self.next_frame_at = 0.0
        self._pending = set()
        self._sequence = 0
        self._lock = threading.Lock()

    def send_if_due(self, render):
        now = self.clock()
        with self._lock:
            if now < self.next_frame_at:
                return False
            if self.require_ack and len(self._pending) >= self.max_in_flight:
                return False
            # Keep the cadence under small timing jitter; never catch up in bursts.
            if now - self.next_frame_at >= self.interval:
                self.next_frame_at = now + self.interval
            else:
                self.next_frame_at += self.interval
            self._sequence += 1
            token = self._sequence
            if self.require_ack:
                self._pending.add(token)

        def acknowledge(*_args):
            with self._lock:
                self._pending.discard(token)

        try:
            # Capture the current environment only when a frame can be sent.
            # No session/input lock is held during rendering, encoding or emit.
            frame = render()
            if self.require_ack:
                self.emit(frame, callback=acknowledge)
            else:
                self.emit(frame)
        except Exception:
            acknowledge()
            raise
        return True
