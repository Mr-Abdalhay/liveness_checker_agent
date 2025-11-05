import cv2
import numpy as np
import logging
from flask import Flask, request
from flask_socketio import SocketIO, emit
from flask_cors import CORS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("LivenessServer")

app = Flask(__name__)
app.config["SECRET_KEY"] = "secret!"
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", max_http_buffer_size=1_000_000)

# Face detection using Haar Cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Facial landmark detector setup
landmark_detector = cv2.face.createFacemarkLBF()
landmark_detector.loadModel("/root/Liveness_service/lbfmodel.yaml")


# Liveness challenge parameters
CHALLENGE_SEQUENCE = ["CENTER", "BLINK", "CENTER"]
EAR_THRESHOLD = 0.25  # Increased threshold - easier to trigger
EAR_CONSEC_FRAMES = 1  # Reduced to 1 frame - faster detection
BLINK_REQUIRED = 6  # Number of blinks required to complete the challenge

user_sessions = {}

def get_eye_aspect_ratio(eye_pts):
    v1 = np.linalg.norm(eye_pts[1] - eye_pts[5])
    v2 = np.linalg.norm(eye_pts[2] - eye_pts[4])
    h = np.linalg.norm(eye_pts[0] - eye_pts[3])
    return (v1 + v2) / (2.0 * h)



def advance_challenge(sid):
    session = user_sessions[sid]
    session["challenge_index"] += 1
    if session["challenge_index"] >= len(CHALLENGE_SEQUENCE):
        token = "dummy_liveness_token"
        emit("result", {"status": "SUCCESS", "token": token}, room=sid)
        user_sessions.pop(sid, None)
        logger.info(f"Session {sid} PASSED.")
    else:
        next_ch = CHALLENGE_SEQUENCE[session["challenge_index"]]
        session.update(blink_counter=0, blink_detected=False, blink_total=0)
        emit("challenge", {"challenge": next_ch}, room=sid)
        logger.info(f"Session {sid} challenge advanced to {next_ch}")

@socketio.on("connect")
def handle_connect():
    sid = request.sid
    user_sessions[sid] = {"challenge_index": 0, "blink_counter": 0, "blink_detected": False, "blink_total": 0}
    emit("challenge", {"challenge": CHALLENGE_SEQUENCE[0]}, room=sid)
    logger.info(f"Client connected: {sid}")

@socketio.on("disconnect")
def handle_disconnect():
    sid = request.sid
    user_sessions.pop(sid, None)
    logger.info(f"Client disconnected: {sid}")

@socketio.on("frame")
def handle_frame(data):
    sid = request.sid
    session = user_sessions.get(sid)
    if not session:
        return

    # decode JPEG bytes
    arr = np.frombuffer(data, np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        emit("feedback", {"status": "FAIL", "message": "Invalid frame"} , room=sid)
        return

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    if len(faces) != 1:
        emit("feedback", {"status": "FAIL", "message": "Keep one face in frame."}, room=sid)
        return

    # Check if face is centered
    face = faces[0]
    frame_center_x = frame.shape[1] // 2
    frame_center_y = frame.shape[0] // 2
    face_center_x = face[0] + face[2] // 2  # x + width/2
    face_center_y = face[1] + face[3] // 2  # y + height/2
    
    # Calculate distance from center
    distance_from_center = ((face_center_x - frame_center_x) ** 2 + (face_center_y - frame_center_y) ** 2) ** 0.5
    max_allowed_distance = min(frame.shape[0], frame.shape[1]) * 0.15  # 15% of smaller dimension
    
    if distance_from_center > max_allowed_distance:
        emit("feedback", {"status": "FAIL", "message": "Please center your face within the circle."}, room=sid)
        return

    # Check face size (distance from camera)
    face_width = face[2]
    face_height = face[3]
    min_face_size = min(frame.shape[0], frame.shape[1]) * 0.2  # 20% of smaller dimension
    max_face_size = min(frame.shape[0], frame.shape[1]) * 0.6  # 60% of smaller dimension
    
    if face_width < min_face_size or face_height < min_face_size:
        emit("feedback", {"status": "FAIL", "message": "Move closer to the camera."}, room=sid)
        return
    
    if face_width > max_face_size or face_height > max_face_size:
        emit("feedback", {"status": "FAIL", "message": "Move further from the camera."}, room=sid)
        return

    ok, landmarks = landmark_detector.fit(gray, np.array([faces[0]]))
    if not ok or landmarks is None or len(landmarks[0]) == 0:
        emit("feedback", {"status": "FAIL", "message": "Facial landmarks not detected."}, room=sid)
        return

    lm = landmarks[0][0]
    current = CHALLENGE_SEQUENCE[session["challenge_index"]]

    if current == "BLINK":
        left = lm[36:42]
        right = lm[42:48]
        ear = (get_eye_aspect_ratio(left) + get_eye_aspect_ratio(right)) / 2.0
        
        # Provide real-time feedback about blink progress
        if ear < EAR_THRESHOLD:
            session["blink_counter"] += 1
            # Send feedback when eyes are closing
            if session["blink_counter"] == 1:
                emit("feedback", {"status": "INFO", "message": f"Blink {session['blink_total'] + 1}/{BLINK_REQUIRED}: Eyes closing... Keep them closed briefly."}, room=sid)
        else:
            # Eyes are open - check if we had a valid blink
            if session["blink_counter"] >= EAR_CONSEC_FRAMES:
                session["blink_total"] += 1
                session["blink_detected"] = True
                emit("feedback", {"status": "SUCCESS", "message": f"Blink {session['blink_total']}/{BLINK_REQUIRED} detected!"}, room=sid)
                
                # Check if we've reached the required number of blinks
                if session["blink_total"] >= BLINK_REQUIRED:
                    emit("feedback", {"status": "SUCCESS", "message": "All blinks completed! Moving to next challenge."}, room=sid)
                    advance_challenge(sid)
                else:
                    # Reset for next blink
                    session["blink_detected"] = False
            session["blink_counter"] = 0

    else:  # CENTER or others
        advance_challenge(sid)

if __name__ == "__main__":
    logger.info("Starting liveness server on port 5000…")
    socketio.run(app, host="0.0.0.0", port=5000)
