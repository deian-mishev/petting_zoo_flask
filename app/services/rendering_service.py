from app.config.session_state import SessionState
from app.services.frame_delivery import bounded_setting
import cv2

JPEG_QUALITY = bounded_setting("STREAM_JPEG_QUALITY", 75, 1, 100)
_JPEG_OPTIONS = (cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY)

def render_frame(session: SessionState) -> bytes:
    frame = session.env.render()
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    success, buffer = cv2.imencode('.jpg', frame, _JPEG_OPTIONS)
    if not success:
        raise RuntimeError("Could not encode gameplay frame")
    return buffer.tobytes()
