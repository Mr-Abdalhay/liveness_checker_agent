import cv2 
import numpy as np 
import logging 
import os 
import random 
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

# Liveness challenge parameters - SIMPLIFIED FOR BETTER UX 
# Random sequence for better security 
def get_random_challenge_sequence(): 
    """Generate random challenge sequence for each session""" 
    challenges = ["TURN_LEFT", "TURN_RIGHT"] 
    random.shuffle(challenges) 
    # Always start with CENTER and end with CENTER 
    return ["CENTER"] + challenges + ["CENTER"] 

HEAD_TURN_THRESHOLD = 0.05  # Much easier - just slight head movement needed 

user_sessions = {} 

def advance_challenge(sid): 
    session = user_sessions[sid] 
    session["challenge_index"] += 1 
    if session["challenge_index"] >= len(session["challenge_sequence"]): 
        token = "dummy_liveness_token" 
        emit("result", {"status": "SUCCESS", "token": token}, room=sid) 
        user_sessions.pop(sid, None) 
        logger.info(f"Session {sid} PASSED.") 
    else: 
        next_ch = session["challenge_sequence"][session["challenge_index"]] 
        emit("challenge", {"challenge": next_ch}, room=sid) 
        logger.info(f"Session {sid} challenge advanced to {next_ch}") 

@socketio.on("connect") 
def handle_connect(): 
    sid = request.sid 
    # Generate random challenge sequence for this session 
    challenge_sequence = get_random_challenge_sequence() 
    user_sessions[sid] = { 
        "challenge_index": 0, 
        "challenge_sequence": challenge_sequence
    } 
    emit("challenge", {"challenge": challenge_sequence[0]}, room=sid) 
    logger.info(f"Client connected: {sid} with sequence: {challenge_sequence}") 

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

    # Calculate distance from center - OVAL SHAPE (not circle) 
    # Allow more vertical tolerance than horizontal for natural face shape 
    horizontal_tolerance = frame.shape[1] * 0.20  # 20% horizontal tolerance 
    vertical_tolerance = frame.shape[0] * 0.25  # 25% vertical tolerance (oval shape) 
    horizontal_offset = abs(face_center_x - frame_center_x) 
    vertical_offset = abs(face_center_y - frame_center_y) 

    # Check if face is within oval bounds 
    if horizontal_offset > horizontal_tolerance or vertical_offset > vertical_tolerance: 
        emit("feedback", {"status": "FAIL", "message": "Please center your face within the oval guide."}, room=sid) 
        return 

    # Check face size (distance from camera) - MORE LENIENT 
    face_width = face[2] 
    face_height = face[3] 
    min_face_size = min(frame.shape[0], frame.shape[1]) * 0.15  # Reduced from 20% to 15% 
    max_face_size = min(frame.shape[0], frame.shape[1]) * 0.70  # Increased from 60% to 70% 
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
    current = session["challenge_sequence"][session["challenge_index"]] 

    if current == "TURN_LEFT": 
        # Check if face is turned left (user's perspective) 
        nose_tip = lm[30]  # Nose tip point 
        left_eye_center = lm[39]  # Left eye center 
        right_eye_center = lm[42]  # Right eye center 

        # Calculate face center from eye centers (more accurate than face box) 
        eyes_center_x = (left_eye_center[0] + right_eye_center[0]) / 2 
        face_width = abs(right_eye_center[0] - left_eye_center[0]) 

        # Calculate nose offset from eyes center 
        nose_offset = (nose_tip[0] - eyes_center_x) / face_width 

        # When user turns LEFT, nose moves RIGHT relative to eyes 
        if nose_offset > HEAD_TURN_THRESHOLD: 
            emit("feedback", {"status": "SUCCESS", "message": "Good! Face turned left."}, room=sid) 
            advance_challenge(sid) 
        else: 
            emit("feedback", {"status": "INFO", "message": "Turn your head slightly to the left."}, room=sid) 

    elif current == "TURN_RIGHT": 
        # Check if face is turned right (user's perspective) 
        nose_tip = lm[30] 
        left_eye_center = lm[39] 
        right_eye_center = lm[42] 

        # Calculate face center from eye centers 
        eyes_center_x = (left_eye_center[0] + right_eye_center[0]) / 2 
        face_width = abs(right_eye_center[0] - left_eye_center[0]) 

        # Calculate nose offset from eyes center 
        nose_offset = (nose_tip[0] - eyes_center_x) / face_width 

        # When user turns RIGHT, nose moves LEFT relative to eyes 
        if nose_offset < -HEAD_TURN_THRESHOLD: 
            emit("feedback", {"status": "SUCCESS", "message": "Good! Face turned right."}, room=sid) 
            advance_challenge(sid) 
        else: 
            emit("feedback", {"status": "INFO", "message": "Turn your head slightly to the right."}, room=sid) 
    else: 
        # CENTER or others 
        advance_challenge(sid) 

if __name__ == "__main__": 
    # Check if SSL certificates exist 
    cert_path = "/root/Liveness_service/certs/server.crt" 
    key_path = "/root/Liveness_service/certs/server.key" 

    if os.path.exists(cert_path) and os.path.exists(key_path): 
        logger.info("Starting SECURE liveness server on port 5000 with SSL...") 
        # For Flask-SocketIO with eventlet, pass certificate files directly 
        socketio.run(app, host="0.0.0.0", port=5000, keyfile=key_path, certfile=cert_path) 
    else: 
        logger.warning("SSL certificates not found! Run generate_ssl_cert.sh first.") 
        logger.info("Starting INSECURE liveness server on port 5000 without SSL...") 
        socketio.run(app, host="0.0.0.0", port=5000) 
