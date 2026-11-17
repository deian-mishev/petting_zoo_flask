from app.config.session_state import SessionState
from app.services.frame_delivery import bounded_setting
import cv2

JPEG_QUALITY = bounded_setting("STREAM_JPEG_QUALITY", 75, 1, 100)

def render_frame(session: SessionState) -> bytes:
    frame = session.env.render()
    success, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not success:
        raise RuntimeError("Could not encode gameplay frame")
    return buffer.tobytes()
