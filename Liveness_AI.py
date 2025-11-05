import cv2
import mediapipe as mp
import numpy as np
import time
import random
import math
import os
from datetime import datetime
import logging

# Set up logging for debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1, 
    refine_landmarks=True, 
    min_detection_confidence=0.5,  # Increased for better accuracy
    min_tracking_confidence=0.5
)

# Enhanced Eye Aspect Ratio calculation
def eye_aspect_ratio(landmarks, eye_indices):
    try:
        p1 = np.array([landmarks[eye_indices[0]].x, landmarks[eye_indices[0]].y])
        p2 = np.array([landmarks[eye_indices[1]].x, landmarks[eye_indices[1]].y])
        p3 = np.array([landmarks[eye_indices[2]].x, landmarks[eye_indices[2]].y])
        p4 = np.array([landmarks[eye_indices[3]].x, landmarks[eye_indices[3]].y])
        p5 = np.array([landmarks[eye_indices[4]].x, landmarks[eye_indices[4]].y])
        p6 = np.array([landmarks[eye_indices[5]].x, landmarks[eye_indices[5]].y])

        vertical1 = np.linalg.norm(p2 - p6)
        vertical2 = np.linalg.norm(p3 - p5)
        horizontal = np.linalg.norm(p1 - p4)

        if horizontal == 0:
            return 0
        
        ear = (vertical1 + vertical2) / (2.0 * horizontal)
        return ear
    except:
        return 0.25  # Default value if calculation fails

# FIXED: Enhanced head pose calculation with correct coordinate mapping
def calculate_head_pose(landmarks, image_width, image_height):
    """
    Calculate head pose with corrected coordinate system
    Returns (yaw, pitch) where:
    - yaw: negative = left turn, positive = right turn (from user's perspective)
    - pitch: negative = looking up, positive = looking down
    """
    try:
        # Key facial landmarks
        nose_tip = landmarks[1]
        nose_bridge = landmarks[168]
        left_eye_corner = landmarks[33]   # User's left eye (right side in mirrored image)
        right_eye_corner = landmarks[263] # User's right eye (left side in mirrored image)
        left_mouth_corner = landmarks[61]
        right_mouth_corner = landmarks[291]
        chin = landmarks[18]
        forehead = landmarks[10]
        
        # Convert normalized coordinates to pixel coordinates for better precision
        nose_tip_px = (int(nose_tip.x * image_width), int(nose_tip.y * image_height))
        left_eye_px = (int(left_eye_corner.x * image_width), int(left_eye_corner.y * image_height))
        right_eye_px = (int(right_eye_corner.x * image_width), int(right_eye_corner.y * image_height))
        
        # Calculate face center between eyes
        eye_center_x = (left_eye_corner.x + right_eye_corner.x) / 2
        eye_center_y = (left_eye_corner.y + right_eye_corner.y) / 2
        
        # Calculate face width and height for normalization
        face_width = abs(right_eye_corner.x - left_eye_corner.x)
        face_height = abs(forehead.y - chin.y)
        
        if face_width == 0 or face_height == 0:
            return 0, 0
        
        # Calculate yaw (horizontal head rotation)
        # When user turns right, nose moves right in the image
        nose_offset_x = (nose_tip.x - eye_center_x) / face_width
        
        # For mirrored camera feed, the coordinate system is already correct
        # When user turns their head right, nose moves right in the mirrored view
        yaw = nose_offset_x * 2.0  # Multiply by 2 for better sensitivity
        
        # Calculate pitch (vertical head rotation)
        nose_offset_y = (nose_tip.y - eye_center_y) / face_height
        pitch = nose_offset_y * 2.0
        
        return yaw, pitch
        
    except Exception as e:
        logger.warning(f"Head pose calculation failed: {e}")
        return 0, 0

# FIXED: Corrected head movement checking with proper thresholds
def check_head_movement(yaw, target_direction):
    """
    Check if head is in the target direction
    yaw: positive = right turn, negative = left turn (from user's perspective)
    """
    # Conservative thresholds for reliable detection
    LEFT_THRESHOLD = -0.15    # User turns left
    RIGHT_THRESHOLD = 0.1    # User turns right  
    CENTER_THRESHOLD = 0.05  # Facing center
    
    logger.debug(f"Head movement check - Yaw: {yaw:.3f}, Target: {target_direction}")
    
    if target_direction == "left" and yaw < LEFT_THRESHOLD:
        return True
    elif target_direction == "right" and yaw > RIGHT_THRESHOLD:
        return True
    elif target_direction == "center" and abs(yaw) < CENTER_THRESHOLD:
        return True
    return False

# FIXED: Enhanced wrong movement detection
def detect_wrong_movement(yaw, target_direction):
    """
    Detect if user is moving in wrong direction
    """
    WRONG_THRESHOLD = 0.25
    
    if target_direction == "left" and yaw > WRONG_THRESHOLD:
        logger.debug(f"Wrong movement: Asked LEFT, got RIGHT (yaw: {yaw:.3f})")
        return True
    elif target_direction == "right" and yaw < -WRONG_THRESHOLD:
        logger.debug(f"Wrong movement: Asked RIGHT, got LEFT (yaw: {yaw:.3f})")
        return True
    elif target_direction == "center" and abs(yaw) > 0.4:
        logger.debug(f"Wrong movement: Asked CENTER, head turned (yaw: {yaw:.3f})")
        return True
    return False

# Enhanced mouth opening detection
def detect_mouth_opening(landmarks):
    try:
        # More accurate mouth landmarks
        upper_lip_top = landmarks[13]
        lower_lip_bottom = landmarks[14]
        left_corner = landmarks[61]
        right_corner = landmarks[291]
        
        # Calculate mouth opening
        mouth_height = abs(upper_lip_top.y - lower_lip_bottom.y)
        mouth_width = abs(right_corner.x - left_corner.x)
        
        # Adaptive threshold based on mouth width
        if mouth_width == 0:
            return False
            
        mouth_ratio = mouth_height / mouth_width
        return mouth_ratio > 0.05 and mouth_height > 0.015
    except:
        return False

# Enhanced tongue detection
def detect_tongue_visibility(landmarks):
    try:
        # First check if mouth is open enough
        if not detect_mouth_opening(landmarks):
            return False
        
        # Check for significant mouth opening that suggests tongue
        upper_lip = landmarks[13]
        lower_lip = landmarks[14]
        mouth_height = abs(upper_lip.y - lower_lip.y)
        
        # Tongue usually requires more mouth opening
        return mouth_height > 0.025
    except:
        return False

# Enhanced eyebrow detection
def detect_eyebrow_raising(landmarks):
    try:
        # Better eyebrow points
        left_eyebrow = landmarks[70]
        right_eyebrow = landmarks[300]
        left_eye_top = landmarks[159]
        right_eye_top = landmarks[386]
        
        # Calculate distances
        left_distance = left_eye_top.y - left_eyebrow.y
        right_distance = right_eye_top.y - right_eyebrow.y
        
        # Threshold for eyebrow raising
        THRESHOLD = 0.018
        return left_distance > THRESHOLD or right_distance > THRESHOLD
    except:
        return False

# Enhanced smile detection
def detect_smile(landmarks):
    try:
        left_corner = landmarks[61]
        right_corner = landmarks[291]
        upper_lip = landmarks[13]
        lower_lip = landmarks[14]
        
        # Calculate mouth width and height
        mouth_width = abs(right_corner.x - left_corner.x)
        mouth_height = abs(upper_lip.y - lower_lip.y)
        
        if mouth_height == 0:
            return False
        
        # Check mouth aspect ratio and corner elevation
        mouth_ratio = mouth_width / mouth_height
        
        # Check if corners are elevated (smile characteristic)
        mouth_center_y = (upper_lip.y + lower_lip.y) / 2
        left_elevation = mouth_center_y - left_corner.y
        right_elevation = mouth_center_y - right_corner.y
        
        # Smile if corners are elevated and mouth is relatively wide
        elevated = left_elevation > 0.008 and right_elevation > 0.008
        wide_enough = mouth_ratio > 2.0
        
        return elevated and wide_enough
    except:
        return False

# Enhanced single eye closing
def detect_single_eye_closing(landmarks, ear_left, ear_right):
    try:
        CLOSED_THRESHOLD = 0.15
        OPEN_THRESHOLD = 0.25
        
        left_closed = ear_left < CLOSED_THRESHOLD
        right_closed = ear_right < CLOSED_THRESHOLD
        left_open = ear_left > OPEN_THRESHOLD
        right_open = ear_right > OPEN_THRESHOLD
        
        # One eye clearly closed, other clearly open
        return (left_closed and right_open) or (right_closed and left_open)
    except:
        return False

# Enhanced image saving with metadata
def save_verification_image(image, verification_data):
    try:
        if not os.path.exists("verification_images"):
            os.makedirs("verification_images")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Include milliseconds
        filename = f"verification_images/verification_{timestamp}.jpg"
        
        # Create info overlay
        info_image = image.copy()
        
        # Add verification stamp
        cv2.putText(info_image, "LIVENESS VERIFIED", (50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        cv2.putText(info_image, f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 
                   (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(info_image, f"Steps: {verification_data['steps_completed']}", 
                   (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(info_image, f"Blinks: {verification_data['total_blinks']}", 
                   (50, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(info_image, f"Time: {verification_data['verification_time']:.1f}s", 
                   (50, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add security watermark
        cv2.putText(info_image, "ANTI-SPOOFING VERIFIED", (50, 220), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Save with high quality
        cv2.imwrite(filename, info_image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        logger.info(f"Verification image saved: {filename}")
        return filename
    except Exception as e:
        logger.error(f"Failed to save verification image: {e}")
        return None

# Eye landmark indices
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# Enhanced thresholds
BLINK_THRESHOLD = 0.2
CONSECUTIVE_FRAMES_THRESHOLD = 3

# Enhanced verification questions
VERIFICATION_ACTIONS = [
    {"text": "Show me a smile", "detector": "smile", "duration": 1.5},
    {"text": "Close both eyes", "detector": "close_eyes", "duration": 1.0},
    {"text": "Open your mouth", "detector": "mouth_open", "duration": 1.5},
    {"text": "Raise your eyebrows", "detector": "eyebrow_raise", "duration": 1.5},
    {"text": "Show your tongue", "detector": "tongue", "duration": 2.0},
    {"text": "Wink (close one eye)", "detector": "single_eye", "duration": 1.5}
]

class EnhancedLivenessDetector:
    def __init__(self):
        self.reset_state()
        
        # Enhanced tracking
        self.head_pose_buffer = []
        self.buffer_size = 10
        self.stability_threshold = 0.1
        self.min_stable_frames = 5
        
        # Anti-spoofing measures
        self.face_size_history = []
        self.movement_entropy = 0
        
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
        self.picture_taken = False
        self.countdown_text = ""
        
        # Step-specific state
        self.blink_count = 0
        self.last_blink_time = 0
        self.step_start_time = 0
        self.action_detected = False
        self.action_start_time = 0
        self.stable_pose_start_time = 0
        
        # Head movement state
        self.target_direction = ""
        self.wrong_movement_count = 0
        self.consecutive_correct_frames = 0
        
    def start_verification(self):
        """Initialize new verification sequence"""
        self.reset_state()
        self.verification_start_time = time.time()
        
        # Create randomized verification steps
        step_types = ["blink", "head_movement", "action"]
        random.shuffle(step_types)
        
        for step_type in step_types:
            if step_type == "blink":
                blinks_required = random.choice([2, 3])
                self.verification_steps.append({
                    "type": "blink", 
                    "target": blinks_required,
                    "timeout": 10
                })
            elif step_type == "head_movement":
                direction = random.choice(["left", "right", "center"])
                self.verification_steps.append({
                    "type": "head_movement", 
                    "target": direction,
                    "timeout": 8
                })
            elif step_type == "action":
                action = random.choice(VERIFICATION_ACTIONS)
                self.verification_steps.append({
                    "type": "action", 
                    "target": action,
                    "timeout": 10
                })
        
        self.step_start_time = time.time()
        logger.info(f"Started verification with {len(self.verification_steps)} steps")

    def get_instruction_text(self):
        """Get current instruction for display"""
        if self.verification_failed:
            return self.failure_message
        
        if self.current_step >= len(self.verification_steps):
            if self.verification_complete and not self.picture_taken and hasattr(self, 'countdown_text') and self.countdown_text:
                return self.countdown_text
            return "VERIFICATION COMPLETE!"
        
        step = self.verification_steps[self.current_step]
        
        if step["type"] == "blink":
            remaining = step["target"] - self.blink_count
            return f"Blink {remaining} more time{'s' if remaining != 1 else ''}"
        elif step["type"] == "head_movement":
            direction = step["target"].upper()
            if direction == "CENTER":
                return "Look straight at the camera"
            else:
                return f"Turn your head {direction}"
        elif step["type"] == "action":
            return step["target"]["text"]
        
        return "Processing..."

    def update_head_pose_buffer(self, yaw, pitch):
        """Maintain head pose history for stability analysis"""
        self.head_pose_buffer.append((yaw, pitch, time.time()))
        if len(self.head_pose_buffer) > self.buffer_size:
            self.head_pose_buffer.pop(0)

    def is_head_pose_stable(self, target_direction):
        """Check if head pose is stable in target direction"""
        if len(self.head_pose_buffer) < self.min_stable_frames:
            return False
        
        recent_poses = self.head_pose_buffer[-self.min_stable_frames:]
        
        # Check if all recent poses are in correct direction
        for yaw, pitch, timestamp in recent_poses:
            if not check_head_movement(yaw, target_direction):
                return False
        
        # Check stability (low variance)
        yaws = [pose[0] for pose in recent_poses]
        if len(yaws) > 1:
            yaw_std = np.std(yaws)
            if yaw_std > self.stability_threshold:
                return False
        
        return True

    def check_facial_action(self, landmarks, action_type, ear_left, ear_right):
        """Check for specific facial actions"""
        try:
            if action_type == "smile":
                return detect_smile(landmarks)
            elif action_type == "close_eyes":
                return ear_left < BLINK_THRESHOLD and ear_right < BLINK_THRESHOLD
            elif action_type == "mouth_open":
                return detect_mouth_opening(landmarks)
            elif action_type == "eyebrow_raise":
                return detect_eyebrow_raising(landmarks)
            elif action_type == "tongue":
                return detect_tongue_visibility(landmarks)
            elif action_type == "single_eye":
                return detect_single_eye_closing(landmarks, ear_left, ear_right)
        except Exception as e:
            logger.warning(f"Facial action detection failed: {e}")
        return False

    def process_frame(self, landmarks, image_width, image_height):
        """Process a single frame and update verification state"""
        if self.verification_failed:
            # Auto-restart after failure timeout
            if time.time() - self.failure_start_time > 3.0:
                self.start_verification()
            return False

        if self.current_step >= len(self.verification_steps):
            if not self.verification_complete:
                self.verification_complete = True
                logger.info("Verification completed successfully!")
            return True

        # Calculate metrics
        left_ear = eye_aspect_ratio(landmarks, LEFT_EYE)
        right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE)
        avg_ear = (left_ear + right_ear) / 2.0
        yaw, pitch = calculate_head_pose(landmarks, image_width, image_height)
        
        # Update tracking
        self.update_head_pose_buffer(yaw, pitch)
        
        # Check for timeout
        current_time = time.time()
        step_duration = current_time - self.step_start_time
        step = self.verification_steps[self.current_step]
        
        if step_duration > step.get("timeout", 10):
            logger.warning(f"Step {self.current_step + 1} timed out")
            self.verification_failed = True
            self.failure_message = "TIMEOUT - Please try again"
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
        """Process blink detection step"""
        if ear < BLINK_THRESHOLD:
            current_time = time.time()
            if current_time - self.last_blink_time > 0.3:  # Prevent double counting
                self.blink_count += 1
                self.total_blinks += 1
                self.last_blink_time = current_time
                logger.debug(f"Blink detected! Count: {self.blink_count}/{step['target']}")
        
        if self.blink_count >= step["target"]:
            self._advance_step()
            return True
        return False

    def _process_head_movement_step(self, yaw, step):
        """Process head movement detection step"""
        target_direction = step["target"]
        
        # Check for wrong movement
        if detect_wrong_movement(yaw, target_direction):
            self.wrong_movement_count += 1
            self.consecutive_correct_frames = 0
            
            if self.wrong_movement_count > 10:  # Allow some tolerance
                logger.warning(f"Too many wrong movements for {target_direction}")
                self.verification_failed = True
                self.failure_message = f"WRONG DIRECTION! Please turn {target_direction.upper()}"
                self.failure_start_time = time.time()
                return False
        else:
            self.wrong_movement_count = max(0, self.wrong_movement_count - 1)
            
            # Check for correct movement
            if check_head_movement(yaw, target_direction):
                self.consecutive_correct_frames += 1
                
                # Require stable pose for completion
                if self.consecutive_correct_frames >= 8:  # About 0.5 seconds at 15fps
                    logger.debug(f"Head movement completed: {target_direction}")
                    self._advance_step()
                    return True
            else:
                self.consecutive_correct_frames = 0
        
        return False

    def _process_action_step(self, landmarks, ear_left, ear_right, step):
        """Process facial action detection step"""
        action = step["target"]
        action_detected = self.check_facial_action(
            landmarks, action["detector"], ear_left, ear_right
        )
        
        current_time = time.time()
        required_duration = action.get("duration", 1.0)
        
        if action_detected:
            if not self.action_detected:
                self.action_detected = True
                self.action_start_time = current_time
                logger.debug(f"Action started: {action['detector']}")
            elif current_time - self.action_start_time >= required_duration:
                logger.debug(f"Action completed: {action['detector']}")
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
        self.wrong_movement_count = 0
        self.consecutive_correct_frames = 0
        
        logger.info(f"Advanced to step {self.current_step + 1}/{len(self.verification_steps)}")

    def get_debug_info(self, yaw, pitch, ear):
        """Get debug information for display"""
        return {
            'ear': ear,
            'yaw': yaw,
            'pitch': pitch,
            'blinks': self.blink_count,
            'step': f"{self.current_step + 1}/{len(self.verification_steps)}" if self.verification_steps else "0/0",
            'target': self.target_direction if hasattr(self, 'target_direction') else "",
            'consecutive_frames': self.consecutive_correct_frames
        }

    def get_verification_results(self):
        """Get final verification results"""
        verification_time = time.time() - self.verification_start_time
        return {
            'success': self.verification_complete,
            'steps_completed': len(self.verification_steps),
            'total_blinks': self.total_blinks,
            'verification_time': verification_time,
            'timestamp': datetime.now().isoformat()
        }

def main():
    """Main execution function"""
    # Initialize detector
    detector = EnhancedLivenessDetector()
    detector.start_verification()
    
    # Initialize camera with better settings
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    print("🚀 Enhanced Face Liveness Detection Started")
    print("📋 Follow the on-screen instructions to verify you're a real person")
    print("⌨️  Press 'r' to restart, 'q' or 'ESC' to quit")
    print("=" * 60)

    frame_count = 0
    fps_counter = time.time()

    try:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                logger.error("Failed to read from camera")
                break

            frame_count += 1
            image = cv2.flip(image, 1)  # Mirror the image
            image_height, image_width = image.shape[:2]
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe
            results = face_mesh.process(rgb_image)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # Process the frame
                    step_completed = detector.process_frame(
                        face_landmarks.landmark, image_width, image_height
                    )
                    
                    # Get debug info for display
                    left_ear = eye_aspect_ratio(face_landmarks.landmark, LEFT_EYE)
                    right_ear = eye_aspect_ratio(face_landmarks.landmark, RIGHT_EYE)
                    avg_ear = (left_ear + right_ear) / 2.0
                    yaw, pitch = calculate_head_pose(face_landmarks.landmark, image_width, image_height)
                    
                    debug_info = detector.get_debug_info(yaw, pitch, avg_ear)
                    
                    # Handle verification completion
                    if detector.verification_complete and not detector.picture_taken:
                        # Add 5-second countdown before taking picture
                        if not hasattr(detector, 'completion_time'):
                            detector.completion_time = time.time()
                        
                        # Wait 5 seconds after completion before taking picture
                        elapsed_time = time.time() - detector.completion_time
                        if elapsed_time >= 3.0:
                            results_data = detector.get_verification_results()
                            saved_path = save_verification_image(image, results_data)
            
                            detector.picture_taken = True
                            if saved_path:
                                print(f"Verification image saved: {saved_path}")
                        else:
                            # Show countdown
                            countdown = 3 - int(elapsed_time)
                            # detector.countdown_text = f"Taking picture in {countdown} seconds..."

            else:
                debug_info = {'ear': 0, 'yaw': 0, 'pitch': 0, 'blinks': 0, 'step': "0/0"}

            # Draw UI
            draw_ui(image, detector, debug_info, results.multi_face_landmarks is not None)
            
            # Show the image
            cv2.imshow('Enhanced Liveness Detection', image)
            
            # Handle key presses
            key = cv2.waitKey(5) & 0xFF
            if key == 27 or key == ord('q'):  # ESC or Q to quit
                break
            elif key == ord('r'):  # R to restart
                detector.start_verification()
                print("Verification restarted")

    except KeyboardInterrupt:
        print("Detection stopped by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("👋 Goodbye!")

def draw_ui(image, detector, debug_info, face_detected):
    """Draw user interface elements"""
    height, width = image.shape[:2]
    
    # Colors
    green = (0, 255, 0)
    red = (0, 0, 255)
    yellow = (0, 255, 255)
    white = (255, 255, 255)
    blue = (255, 0, 0)
    
    # Main instruction
    instruction = detector.get_instruction_text()
    font_scale = 0.8
    thickness = 2
    
    # Multi-line text support for long instructions
    if len(instruction) > 50:
        words = instruction.split()
        line1 = ' '.join(words[:len(words)//2])
        line2 = ' '.join(words[len(words)//2:])
        cv2.putText(image, line1, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, font_scale, yellow, thickness)
        cv2.putText(image, line2, (30, 90), cv2.FONT_HERSHEY_SIMPLEX, font_scale, yellow, thickness)
    else:
        cv2.putText(image, instruction, (30, 70), cv2.FONT_HERSHEY_SIMPLEX, font_scale, yellow, thickness)
    
    # Status indicator
    if not face_detected:
        status_text = "NO FACE DETECTED"
        status_color = red
    elif detector.verification_complete:
        status_text = "VERIFICATION COMPLETE"
        status_color = green
    elif detector.verification_failed:
        status_text = "VERIFICATION FAILED"
        status_color = red
    else:
        status_text = "VERIFICATION IN PROGRESS"
        status_color = blue
    
    cv2.putText(image, status_text, (30, height - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
    
    # Debug information (top right)
    debug_y = 30
    debug_x = width - 250
    cv2.putText(image, f"EAR: {debug_info['ear']:.3f}", (debug_x, debug_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, white, 1)
    cv2.putText(image, f"Yaw: {debug_info['yaw']:.3f}", (debug_x, debug_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, white, 1)
    cv2.putText(image, f"Pitch: {debug_info['pitch']:.3f}", (debug_x, debug_y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, white, 1)
    cv2.putText(image, f"Blinks: {debug_info['blinks']}", (debug_x, debug_y + 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, green, 1)
    cv2.putText(image, f"Step: {debug_info['step']}", (debug_x, debug_y + 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, yellow, 1)
    
    # Progress bar
    if detector.verification_steps:
        progress = detector.current_step / len(detector.verification_steps)
        bar_width = 200
        bar_height = 20
        bar_x = (width - bar_width) // 2
        bar_y = height - 100
        
        # Background
        cv2.rectangle(image, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), white, 2)
        # Fill
        fill_width = int(bar_width * progress)
        if fill_width > 0:
            cv2.rectangle(image, (bar_x + 2, bar_y + 2), (bar_x + fill_width - 2, bar_y + bar_height - 2), green, -1)
        
        # Progress text
        cv2.putText(image, f"Progress: {int(progress * 100)}%", (bar_x, bar_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, white, 1)

if __name__ == "__main__":
    main()
