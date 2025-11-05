"""
Simple Liveness Detection Server
ONLY does liveness verification and saves verified images
No passport integration - just pure liveness detection

Run: python liveness_server.py
Server runs on port 5002
"""

from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import base64
import logging
import os
import cv2
import numpy as np
import time
import random
import threading
from datetime import datetime, timedelta

# Import your existing liveness detection code
import sys
sys.path.append('..')

import mediapipe as mp
from Liveness_AI import (
    face_mesh, eye_aspect_ratio, calculate_head_pose, check_head_movement,
    detect_mouth_opening, detect_smile, detect_eyebrow_raising, detect_tongue_visibility,
    detect_single_eye_closing, LEFT_EYE, RIGHT_EYE, BLINK_THRESHOLD,
    EnhancedLivenessDetector
)

# Override the detector to use shorter timeouts and more lenient detection
class FastLivenessDetector:
    def __init__(self):
        self.reset_state()
        
    def reset_state(self):
        """Reset all detector state"""
        self.verification_steps = []
        self.current_step = 0
        self.verification_complete = False
        self.verification_failed = False
        self.failure_message = ""
        self.failure_start_time = 0
        self.verification_start_time = 0
        self.total_blinks = 0
        
        # Step-specific state
        self.blink_count = 0
        self.last_blink_time = 0
        self.step_start_time = 0
        self.action_detected = False
        self.action_start_time = 0
        self.consecutive_correct_frames = 0
        
    def start_verification(self):
        """Initialize new verification sequence with faster/easier steps"""
        self.reset_state()
        self.verification_start_time = time.time()
        
        # Create simpler, faster verification steps
        self.verification_steps = [
            {
                "type": "blink", 
                "target": 2,  # Reduced from 3
                "timeout": 25   # Reduced from 10
            },
            {
                "type": "head_movement", 
                "target": "left",
                "timeout": 40   # Reduced from 8
            },
            {
                "type": "action", 
                "target": {"detector": "smile", "instruction": "Please smile", "duration": 1.0},
                "timeout": 40   # Reduced from 10
            }
        ]
        
        self.step_start_time = time.time()
        logger.info(f"Started fast verification with {len(self.verification_steps)} steps")

    def get_instruction_text(self):
        """Get current instruction for display"""
        if self.verification_failed:
            return self.failure_message
        
        if self.current_step >= len(self.verification_steps):
            return "VERIFICATION COMPLETE!"
        
        step = self.verification_steps[self.current_step]
        
        if step["type"] == "blink":
            remaining = step["target"] - self.blink_count
            return f"Blink {remaining} more time{'s' if remaining != 1 else ''}"
        elif step["type"] == "head_movement":
            direction = step["target"].upper()
            return f"Turn your head {direction}"
        elif step["type"] == "action":
            return step["target"]["instruction"]
        
        return "Processing..."

    def process_frame(self, landmarks, image_width, image_height):
        """Process a single frame and update verification state"""
        if self.verification_failed:
            # Auto-restart after failure timeout
            if time.time() - self.failure_start_time > 2.0:  # Reduced timeout
                self.start_verification()
            return False

        if self.current_step >= len(self.verification_steps):
            if not self.verification_complete:
                self.verification_complete = True
                logger.info("Fast verification completed successfully!")
            return True

        # Calculate metrics
        left_ear = eye_aspect_ratio(landmarks, LEFT_EYE)
        right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE)
        avg_ear = (left_ear + right_ear) / 2.0
        yaw, pitch = calculate_head_pose(landmarks, image_width, image_height)
        
        # Check for timeout
        current_time = time.time()
        step_duration = current_time - self.step_start_time
        step = self.verification_steps[self.current_step]
        
        if step_duration > step.get("timeout", 8):
            logger.warning(f"Step {self.current_step + 1} timed out after {step_duration:.1f}s")
            self.verification_failed = True
            self.failure_message = f"Step timeout - try {step.get('target', 'action')} faster"
            self.failure_start_time = current_time
            return False

        # Process current step
        if step["type"] == "blink":
            return self._process_blink_step(avg_ear, step)
        elif step["type"] == "head_movement":
            return self._process_head_movement_step(yaw, step)
        elif step["type"] == "action":
            return self._process_action_step(landmarks, left_ear, right_ear, step)
        
        return False

    def _process_blink_step(self, ear, step):
        """Process blink detection step with more lenient thresholds"""
        # More lenient blink threshold
        LENIENT_BLINK_THRESHOLD = 0.25  # Increased from 0.2
        
        if ear < LENIENT_BLINK_THRESHOLD:
            current_time = time.time()
            if current_time - self.last_blink_time > 0.2:  # Reduced from 0.3
                self.blink_count += 1
                self.total_blinks += 1
                self.last_blink_time = current_time
                logger.info(f"Blink detected! Count: {self.blink_count}/{step['target']} (EAR: {ear:.3f})")
        
        if self.blink_count >= step["target"]:
            self._advance_step()
            return True
        return False

    def _process_head_movement_step(self, yaw, step):
        """Process head movement with more lenient thresholds"""
        target_direction = step["target"]
        
        # More lenient thresholds
        LEFT_THRESHOLD = -0.2    # Reduced from -0.3
        RIGHT_THRESHOLD = 0.2    # Reduced from 0.3
        
        movement_detected = False
        if target_direction == "left" and yaw < LEFT_THRESHOLD:
            movement_detected = True
        elif target_direction == "right" and yaw > RIGHT_THRESHOLD:
            movement_detected = True
        elif target_direction == "center" and abs(yaw) < 0.1:
            movement_detected = True
        
        if movement_detected:
            self.consecutive_correct_frames += 1
            
            # Require fewer consecutive frames
            if self.consecutive_correct_frames >= 5:  # Reduced from 8
                logger.info(f"Head movement completed: {target_direction} (yaw: {yaw:.3f})")
                self._advance_step()
                return True
        else:
            self.consecutive_correct_frames = 0
        
        return False

    def _process_action_step(self, landmarks, ear_left, ear_right, step):
        """Process facial action with more lenient detection"""
        action = step["target"]
        
        action_detected = False
        if action["detector"] == "smile":
            action_detected = detect_smile(landmarks)
        elif action["detector"] == "mouth_open":
            action_detected = detect_mouth_opening(landmarks)
        elif action["detector"] == "eyebrow_raise":
            action_detected = detect_eyebrow_raising(landmarks)
        
        current_time = time.time()
        required_duration = action.get("duration", 0.8)  # Reduced from 1.0
        
        if action_detected:
            if not self.action_detected:
                self.action_detected = True
                self.action_start_time = current_time
                logger.info(f"Action started: {action['detector']}")
            elif current_time - self.action_start_time >= required_duration:
                logger.info(f"Action completed: {action['detector']}")
                self._advance_step()
                return True
        else:
            self.action_detected = False
            self.action_start_time = 0
        
        return False

    def _advance_step(self):
        """Move to next verification step"""
        self.current_step += 1
        self.step_start_time = time.time()
        
        # Reset step-specific state
        self.blink_count = 0
        self.action_detected = False
        self.action_start_time = 0
        self.consecutive_correct_frames = 0
        
        logger.info(f"Advanced to step {self.current_step + 1}/{len(self.verification_steps)}")

    def get_verification_results(self):
        """Get final verification results"""
        verification_time = time.time() - self.verification_start_time
        return {
            'success': self.verification_complete,
            'steps_completed': len(self.verification_steps) if self.verification_complete else self.current_step,
            'total_blinks': self.total_blinks,
            'verification_time': verification_time,
            'timestamp': datetime.now().isoformat()
        }

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Simple logging with DEBUG level for troubleshooting
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
SESSION_EXPIRY_MINUTES = 10
MAX_SESSIONS = 100

# Storage
active_sessions = {}

# Ensure output directory exists
os.makedirs('verified_images', exist_ok=True)

class LivenessSession:
    def __init__(self, session_id):
        self.session_id = session_id
        self.created_at = datetime.now()
        self.detector = FastLivenessDetector()  # Use the faster detector
        self.detector.start_verification()
        self.completed = False
        self.verified_image_path = None
        self.last_frame = None
        
    def is_expired(self):
        return datetime.now() - self.created_at > timedelta(minutes=SESSION_EXPIRY_MINUTES)
    
    def save_verified_image(self, frame):
        """Save the verified image when liveness is complete"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
            filename = f"verified_images/liveness_verified_{self.session_id}_{timestamp}.jpg"
            
            # Add verification overlay
            overlay_frame = frame.copy()
            cv2.putText(overlay_frame, "LIVENESS VERIFIED", (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.putText(overlay_frame, f"Session: {self.session_id}", (50, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(overlay_frame, f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 
                       (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Save image
            cv2.imwrite(filename, overlay_frame)
            self.verified_image_path = filename
            
            logger.info(f"Verified image saved: {filename}")
            return filename
            
        except Exception as e:
            logger.error(f"Failed to save verified image: {e}")
            return None

def generate_session_id():
    return f"liveness_{int(time.time())}_{random.randint(1000, 9999)}"

def cleanup_expired_sessions():
    """Clean up expired sessions"""
    while True:
        try:
            expired_sessions = [
                sid for sid, session in active_sessions.items() 
                if session.is_expired()
            ]
            
            for sid in expired_sessions:
                del active_sessions[sid]
            
            if expired_sessions:
                logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")
                
        except Exception as e:
            logger.error(f"Cleanup error: {e}")
        
        time.sleep(300)  # Run every 5 minutes

# Start cleanup thread
cleanup_thread = threading.Thread(target=cleanup_expired_sessions, daemon=True)
cleanup_thread.start()

@app.route('/')
def home():
    return render_template_string("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Liveness Detection Server</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 50px auto; padding: 20px; }
            .status { background: #e8f5e8; padding: 20px; border-radius: 5px; margin: 20px 0; }
            .endpoint { background: #f9f9f9; padding: 15px; margin: 10px 0; border-radius: 5px; }
            code { background: #eee; padding: 2px 5px; border-radius: 3px; }
        </style>
    </head>
    <body>
        <h1>👁️ Liveness Detection Server</h1>
        <div class="status">
            <h3>✅ Server Status: ACTIVE</h3>
            <p>Pure liveness detection - no passport integration</p>
            <p>Active Sessions: {{ active_sessions }}</p>
        </div>
        
        <h2>📡 API Endpoints:</h2>
        
        <div class="endpoint">
            <h4>Start Session</h4>
            <p><code>POST /start</code></p>
            <p>Start new liveness verification session</p>
        </div>
        
        <div class="endpoint">
            <h4>Process Frame</h4>
            <p><code>POST /process</code></p>
            <p>Process camera frame for liveness detection</p>
        </div>
        
        <div class="endpoint">
            <h4>Complete Verification</h4>
            <p><code>POST /complete</code></p>
            <p>Complete verification and get final result</p>
        </div>
        
        <div class="endpoint">
            <h4>Session Status</h4>
            <p><code>GET /status/&lt;session_id&gt;</code></p>
            <p>Get current session status</p>
        </div>
        
        <div class="endpoint">
            <h4>Health Check</h4>
            <p><code>GET /health</code></p>
            <p>Server health status</p>
        </div>
        
        <h2>💾 Features:</h2>
        <ul>
            <li>Real-time liveness detection</li>
            <li>Automatic verified image saving</li>
            <li>Session management</li>
            <li>Progress tracking</li>
            <li>No authentication required</li>
        </ul>
        
        <div style="margin-top: 30px; color: #666; text-align: center;">
            Liveness Only | Images Saved | Ready to Use
        </div>
    </body>
    </html>
    """, active_sessions=len(active_sessions))

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'service': 'liveness-detection-only',
        'version': '1.0',
        'active_sessions': len(active_sessions),
        'max_sessions': MAX_SESSIONS,
        'features': ['liveness_detection', 'image_saving', 'session_management']
    })

@app.route('/start', methods=['POST'])
def start_session():
    """Start a new liveness verification session"""
    try:
        if len(active_sessions) >= MAX_SESSIONS:
            return jsonify({'error': 'Service at capacity'}), 503
        
        session_id = generate_session_id()
        session = LivenessSession(session_id)
        active_sessions[session_id] = session
        
        logger.info(f"Started session: {session_id}")
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'instruction': session.detector.get_instruction_text(),
            'current_step': 1,
            'total_steps': len(session.detector.verification_steps),
            'expires_in_minutes': SESSION_EXPIRY_MINUTES
        })
        
    except Exception as e:
        logger.error(f"Start session error: {e}")
        return jsonify({'error': 'Failed to start session'}), 500

@app.route('/process', methods=['POST'])
def process_frame():
    """Process a camera frame for liveness detection"""
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        image_data = data.get('image')
        
        if not session_id or session_id not in active_sessions:
            return jsonify({'error': 'Invalid session ID'}), 400
        
        session = active_sessions[session_id]
        
        if session.is_expired():
            del active_sessions[session_id]
            return jsonify({'error': 'Session expired'}), 400
        
        if session.completed:
            return jsonify({
                'success': True,
                'verification_complete': True,
                'verification_failed': False,
                'message': 'Verification already completed',
                'verified_image_path': session.verified_image_path
            })
        
        if not image_data:
            return jsonify({'error': 'No image provided'}), 400
        
        # Decode image
        try:
            if image_data.startswith('data:image'):
                image_data = image_data.split(',')[1]
            image_bytes = base64.b64decode(image_data)
            nparr = np.frombuffer(image_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is None:
                logger.error("Failed to decode image - cv2.imdecode returned None")
                return jsonify({'error': 'Invalid image data - decode failed'}), 400
                
            # Store last frame for saving when complete
            session.last_frame = frame
            
            logger.info(f"Successfully decoded frame: {frame.shape}")
            
        except Exception as e:
            logger.error(f"Image decode error: {str(e)}")
            return jsonify({'error': f'Failed to decode image: {str(e)}'}), 400
        
        # Process with MediaPipe - fix image dimensions issue
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Ensure frame has proper dimensions for MediaPipe
        height, width = frame.shape[:2]
        if height == 0 or width == 0:
            return jsonify({'error': 'Invalid image dimensions'}), 400
        
        # Process with MediaPipe
        results = face_mesh.process(rgb_frame)
        
        if not results.multi_face_landmarks:
            logger.debug("No face landmarks detected in frame")
            return jsonify({
                'success': False,
                'face_detected': False,
                'instruction': session.detector.get_instruction_text(),
                'step_completed': False,
                'verification_complete': False,
                'verification_failed': False,
                'current_step': session.detector.current_step + 1,
                'total_steps': len(session.detector.verification_steps),
                'error': 'No face detected'
            })
        
        # Process frame with detector
        try:
            landmarks = results.multi_face_landmarks[0].landmark
            height, width = frame.shape[:2]
            
            logger.debug(f"Processing frame with detector: {width}x{height}, landmarks: {len(landmarks)}")
            logger.debug(f"Current step: {session.detector.current_step + 1}/{len(session.detector.verification_steps)}")
            logger.debug(f"Step instruction: {session.detector.get_instruction_text()}")
            
            step_completed = session.detector.process_frame(landmarks, width, height)
            
            logger.debug(f"Detector result: step_completed={step_completed}, verification_complete={session.detector.verification_complete}")
            
        except Exception as e:
            logger.error(f"Detector processing error: {str(e)}")
            return jsonify({
                'success': False,
                'face_detected': True,
                'instruction': session.detector.get_instruction_text(),
                'step_completed': False,
                'verification_complete': False,
                'verification_failed': False,
                'current_step': session.detector.current_step + 1,
                'total_steps': len(session.detector.verification_steps),
                'error': f'Detection processing failed: {str(e)}'
            })
        
        # Check if verification completed
        verification_complete = session.detector.verification_complete
        verification_failed = session.detector.verification_failed
        
        if verification_complete and not session.completed:
            session.completed = True
            # Save verified image
            image_path = session.save_verified_image(frame)
            logger.info(f"Liveness verification completed for session: {session_id}, image saved: {image_path}")
        
        # Prepare response
        response_data = {
            'success': True,
            'face_detected': True,
            'instruction': session.detector.get_instruction_text(),
            'step_completed': step_completed,
            'verification_complete': verification_complete,
            'verification_failed': verification_failed,
            'current_step': session.detector.current_step + 1,
            'total_steps': len(session.detector.verification_steps),
            'verified_image_path': session.verified_image_path if session.completed else None
        }
        
        logger.debug(f"Sending response: {response_data}")
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Process frame error: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return jsonify({
            'success': False,
            'face_detected': False,
            'instruction': 'Processing error occurred',
            'step_completed': False,
            'verification_complete': False,
            'verification_failed': True,
            'current_step': 0,
            'total_steps': 0,
            'error': f'Frame processing failed: {str(e)}'
        }), 500

@app.route('/complete', methods=['POST'])
def complete_verification():
    """Complete verification and get final results"""
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        
        if not session_id or session_id not in active_sessions:
            return jsonify({'error': 'Invalid session ID'}), 400
        
        session = active_sessions[session_id]
        
        if session.is_expired():
            del active_sessions[session_id]
            return jsonify({'error': 'Session expired'}), 400
        
        if not session.completed:
            return jsonify({'error': 'Verification not completed yet'}), 400
        
        # Get verification results
        results = session.detector.get_verification_results()
        
        # Encode verified image if available
        verified_image_base64 = None
        if session.verified_image_path and os.path.exists(session.verified_image_path):
            try:
                with open(session.verified_image_path, 'rb') as f:
                    image_bytes = f.read()
                    verified_image_base64 = base64.b64encode(image_bytes).decode('utf-8')
            except Exception as e:
                logger.error(f"Failed to encode verified image: {e}")
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'verified_image': verified_image_base64,
            'verified_image_path': session.verified_image_path,
            'verification_time': results['verification_time'],
            'steps_completed': results['steps_completed'],
            'total_blinks': results['total_blinks'],
            'timestamp': results['timestamp'],
            'results': results
        })
        
    except Exception as e:
        logger.error(f"Complete verification error: {e}")
        return jsonify({'error': 'Failed to complete verification'}), 500

@app.route('/status/<session_id>')
def get_session_status(session_id):
    """Get session status"""
    try:
        if session_id not in active_sessions:
            return jsonify({'error': 'Session not found'}), 404
        
        session = active_sessions[session_id]
        
        if session.is_expired():
            del active_sessions[session_id]
            return jsonify({'error': 'Session expired'}), 400
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'created_at': session.created_at.isoformat(),
            'completed': session.completed,
            'current_step': session.detector.current_step + 1,
            'total_steps': len(session.detector.verification_steps),
            'instruction': session.detector.get_instruction_text(),
            'verification_complete': session.detector.verification_complete,
            'verification_failed': session.detector.verification_failed,
            'verified_image_path': session.verified_image_path
        })
        
    except Exception as e:
        logger.error(f"Get status error: {e}")
        return jsonify({'error': 'Failed to get status'}), 500

@app.route('/stats')
def get_stats():
    """Get server statistics"""
    return jsonify({
        'active_sessions': len(active_sessions),
        'max_sessions': MAX_SESSIONS,
        'session_expiry_minutes': SESSION_EXPIRY_MINUTES,
        'service_type': 'liveness_only',
        'features': {
            'liveness_detection': True,
            'image_saving': True,
            'passport_integration': False,
            'authentication': False
        }
    })

if __name__ == '__main__':
    print("👁️ Liveness Detection Server")
    print("=" * 40)
    print("Version: 1.0 Liveness-Only")
    print("Port: 5002")
    print("Features: Liveness Detection + Image Saving")
    print("Authentication: DISABLED")
    print("=" * 40)
    print("\nEndpoints:")
    print("  - POST /start           : Start verification session")
    print("  - POST /process         : Process camera frame")
    print("  - POST /complete        : Complete verification")
    print("  - GET  /status/<id>     : Get session status")
    print("  - GET  /health          : Health check")
    print("  - GET  /stats           : Server statistics")
    print("\n💾 Verified images saved to: ./verified_images/")
    print("\n🚀 Quick Test:")
    print("  curl -X POST http://localhost:5002/start")
    print("  curl http://localhost:5002/health")
    print("\nPress Ctrl+C to stop")
    
    app.run(
        host='0.0.0.0',
        port=5002,
        debug=False,
        threaded=True
    )
