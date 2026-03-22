from app.config.session_state import SessionState
import cv2

def render_frame(session: SessionState) -> bytes:
    frame = session.env.render()
    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
    return buffer.tobytes()