# Liveness Detection Service

A comprehensive face liveness detection system using computer vision and AI to verify real human presence. This service provides multiple detection methods including blink detection, head movement tracking, and facial action recognition.

## Features

- **Real-time Face Detection**: MediaPipe-based facial landmark detection
- **Multi-step Verification**: Randomized challenge sequences for enhanced security
- **Multiple Detection Methods**:
  - Eye blink detection
  - Head movement (left, right, center)
  - Smile detection
  - Mouth opening detection
  - Eyebrow raising
  - Tongue visibility
  - Single eye winking
- **Anti-spoofing Measures**: Dynamic verification steps prevent replay attacks
- **Image Verification**: Automatic capture and storage of verified images
- **REST API**: Complete HTTP API with session management
- **WebSocket Support**: Real-time SocketIO implementation available
- **SSL/TLS Support**: Secure HTTPS communication

## Architecture

### Server Implementations

The project includes multiple server implementations for different use cases:

1. **liveness_server.py** (Recommended)
   - Full-featured REST API server
   - Port: 5002
   - Session management with expiry
   - Comprehensive verification steps
   - Image saving with metadata
   - Health monitoring endpoints

2. **li_ssl.py** (Alternative)
   - SocketIO-based real-time server
   - Head turn verification
   - Built-in SSL support
   - Port: 5000

3. **li.py** (Basic)
   - Simple SocketIO server
   - Blink-based verification
   - Lightweight implementation
   - Port: 5000

### Core Components

- **Liveness_AI.py**: Core detection algorithms and enhanced liveness detector class
- **lbfmodel.yaml**: Pre-trained facial landmark detection model (56MB)

## Requirements

### System Requirements
- Python 3.7+
- Webcam/Camera access
- 2GB+ RAM
- OpenSSL (for certificate generation)

### Python Dependencies

```bash
pip install opencv-python
pip install mediapipe
pip install numpy
pip install flask
pip install flask-socketio
pip install flask-cors
```

Or install all at once:
```bash
pip install opencv-python mediapipe numpy flask flask-socketio flask-cors
```

## Installation

1. **Clone or download the repository**
   ```bash
   cd Liveness_service
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Generate SSL certificates (optional but recommended)**
   ```bash
   chmod +x generate_ssl_cert.sh
   ./generate_ssl_cert.sh
   ```

   Or on Windows:
   ```bash
   bash generate_ssl_cert.sh
   ```

4. **Verify model file exists**
   - Ensure `lbfmodel.yaml` is present in the project root
   - File size should be approximately 56MB

## Usage

### Running the Main Server (Recommended)

```bash
python liveness_server.py
```

Server will start on `http://0.0.0.0:5002`

Access the web interface: `http://localhost:5002`

### Running Alternative Servers

**SSL-enabled SocketIO server:**
```bash
python li_ssl.py
```

**Basic SocketIO server:**
```bash
python li.py
```

### Running Standalone Detection

```bash
python Liveness_AI.py
```

This launches a desktop application with GUI for testing.

## API Documentation

### REST API Endpoints (liveness_server.py)

#### 1. Health Check
```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "service": "liveness-detection-only",
  "version": "1.0",
  "active_sessions": 0,
  "max_sessions": 100
}
```

#### 2. Start Verification Session
```http
POST /start
```

**Response:**
```json
{
  "success": true,
  "session_id": "liveness_1699123456_1234",
  "instruction": "Blink 2 more times",
  "current_step": 1,
  "total_steps": 3,
  "expires_in_minutes": 10
}
```

#### 3. Process Frame
```http
POST /process
Content-Type: application/json

{
  "session_id": "liveness_1699123456_1234",
  "image": "base64_encoded_image_data"
}
```

**Response:**
```json
{
  "success": true,
  "face_detected": true,
  "instruction": "Turn your head left",
  "step_completed": false,
  "verification_complete": false,
  "verification_failed": false,
  "current_step": 2,
  "total_steps": 3
}
```

#### 4. Complete Verification
```http
POST /complete
Content-Type: application/json

{
  "session_id": "liveness_1699123456_1234"
}
```

**Response:**
```json
{
  "success": true,
  "session_id": "liveness_1699123456_1234",
  "verified_image": "base64_encoded_image",
  "verified_image_path": "verified_images/liveness_verified_xxx.jpg",
  "verification_time": 12.5,
  "steps_completed": 3,
  "total_blinks": 4,
  "timestamp": "2024-11-05T15:30:45"
}
```

#### 5. Get Session Status
```http
GET /status/<session_id>
```

**Response:**
```json
{
  "success": true,
  "session_id": "liveness_1699123456_1234",
  "created_at": "2024-11-05T15:30:00",
  "completed": false,
  "current_step": 2,
  "total_steps": 3,
  "instruction": "Turn your head left"
}
```

#### 6. Server Statistics
```http
GET /stats
```

### WebSocket API (li.py, li_ssl.py)

#### Events

**Client -> Server:**
- `connect`: Establish connection and receive first challenge
- `frame`: Send camera frame (binary JPEG data)
- `disconnect`: Close connection

**Server -> Client:**
- `challenge`: Receive new challenge instruction
- `feedback`: Real-time feedback on current action
- `result`: Final verification result with token

#### Example Usage (JavaScript)
```javascript
const socket = io('http://localhost:5000');

socket.on('challenge', (data) => {
  console.log('Challenge:', data.challenge);
});

socket.on('feedback', (data) => {
  console.log('Feedback:', data.message);
});

socket.on('result', (data) => {
  console.log('Result:', data.status, data.token);
});

// Send frame
socket.emit('frame', jpegBinaryData);
```

## Configuration

### Server Settings

**liveness_server.py:**
```python
SESSION_EXPIRY_MINUTES = 10  # Session timeout
MAX_SESSIONS = 100           # Maximum concurrent sessions
PORT = 5002                  # Server port
```

**li_ssl.py / li.py:**
```python
PORT = 5000
CHALLENGE_SEQUENCE = ["CENTER", "BLINK", "CENTER"]
EAR_THRESHOLD = 0.25        # Eye aspect ratio for blink detection
BLINK_REQUIRED = 6          # Number of blinks required
```

### Detection Thresholds

Edit `Liveness_AI.py` to adjust detection sensitivity:

```python
BLINK_THRESHOLD = 0.2           # Lower = easier to blink
LEFT_THRESHOLD = -0.15          # Head turn left sensitivity
RIGHT_THRESHOLD = 0.1           # Head turn right sensitivity
```

## SSL/TLS Configuration

### Generate Self-Signed Certificate

```bash
./generate_ssl_cert.sh
```

This creates:
- `certs/server.key` - Private key (keep secure!)
- `certs/server.crt` - Certificate
- `certs/server.pem` - Certificate for mobile apps

### Certificate Details
- **Validity**: 365 days
- **Key Size**: 2048-bit RSA
- **IP Address**: 5.22.215.77 (update in script if needed)
- **Alternative Names**: localhost, 127.0.0.1

### Using Custom Certificate

Replace files in `certs/` directory:
```
certs/
├── server.key
├── server.crt
└── server.pem
```

## Directory Structure

```
Liveness_service/
├── Liveness_AI.py              # Core detection algorithms
├── liveness_server.py          # Main REST API server (recommended)
├── li_ssl.py                   # SSL SocketIO server
├── li.py                       # Basic SocketIO server
├── lbfmodel.yaml               # Facial landmark model (56MB)
├── generate_ssl_cert.sh        # Certificate generation script
├── cer.sh                      # Alternative cert script
├── README.md                   # This file
├── certs/                      # SSL certificates
│   ├── server.key
│   ├── server.crt
│   ├── server.pem
│   └── cert.conf
├── verified_images/            # Saved verification images
├── verification_images/        # Alternative storage location
├── output/                     # Output directory (diagnostics)
│   ├── diagnostics/
│   ├── enhanced/
│   ├── faces/
│   └── results/
└── logs/                       # Server logs
    └── security.log
```

## Verification Process Flow

1. **Session Initialization**
   - Client requests new session
   - Server generates unique session ID
   - Random challenge sequence created

2. **Challenge Execution**
   - Client receives first challenge
   - Client sends camera frames
   - Server detects facial landmarks
   - Server validates challenge completion

3. **Multi-step Verification**
   - Blink detection (2-3 blinks)
   - Head movement (left/right/center)
   - Facial action (smile/mouth open/etc.)

4. **Completion**
   - All steps verified
   - Image captured and saved
   - Verification token issued

## Security Features

- **Anti-spoofing**: Random challenge sequences
- **Session Management**: Time-limited sessions with automatic cleanup
- **SSL/TLS Encryption**: Secure data transmission
- **Image Verification**: Timestamped verification images
- **Rate Limiting**: Maximum session limits
- **Timeout Protection**: Automatic session expiry

## Troubleshooting

### Camera Not Detected
```python
# Check available cameras
import cv2
cap = cv2.VideoCapture(0)  # Try 0, 1, 2...
print(cap.isOpened())
```

### Model File Not Found
- Verify `lbfmodel.yaml` exists in project root
- Check file size (should be ~56MB)
- Re-download if corrupted

### SSL Certificate Errors
```bash
# Regenerate certificate
./generate_ssl_cert.sh

# Test certificate
openssl x509 -in certs/server.crt -text -noout
```

### Port Already in Use
```bash
# Find process using port
netstat -ano | findstr :5002  # Windows
lsof -i :5002                 # Linux/Mac

# Kill process or change port in code
```

### MediaPipe Errors
```bash
# Reinstall MediaPipe
pip uninstall mediapipe
pip install mediapipe
```

## Performance Optimization

### For Production:
1. Use production WSGI server (gunicorn/uwsgi)
   ```bash
   pip install gunicorn
   gunicorn -w 4 -b 0.0.0.0:5002 liveness_server:app
   ```

2. Enable frame rate limiting
3. Implement caching for model loading
4. Use Redis for session storage
5. Add load balancing for multiple instances

### For Development:
- Use debug mode: `app.run(debug=True)`
- Lower image resolution for faster processing
- Reduce detection thresholds for easier testing

## Integration Examples

### Python Client
```python
import requests
import base64

# Start session
response = requests.post('http://localhost:5002/start')
session_id = response.json()['session_id']

# Send frame
with open('image.jpg', 'rb') as f:
    image_data = base64.b64encode(f.read()).decode()

requests.post('http://localhost:5002/process', json={
    'session_id': session_id,
    'image': image_data
})
```

### JavaScript/React
```javascript
// Start session
const response = await fetch('http://localhost:5002/start', {
  method: 'POST'
});
const { session_id } = await response.json();

// Capture and send frame
const canvas = document.createElement('canvas');
const image = canvas.toDataURL('image/jpeg').split(',')[1];

await fetch('http://localhost:5002/process', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ session_id, image })
});
```

### Flutter/Dart
```dart
import 'dart:convert';
import 'package:http/http.dart' as http;

// Start session
final response = await http.post(
  Uri.parse('http://localhost:5002/start')
);
final sessionId = json.decode(response.body)['session_id'];

// Send frame
final bytes = await image.readAsBytes();
final base64Image = base64Encode(bytes);

await http.post(
  Uri.parse('http://localhost:5002/process'),
  headers: {'Content-Type': 'application/json'},
  body: json.encode({'session_id': sessionId, 'image': base64Image})
);
```

## License

This project is provided as-is for liveness detection purposes.

## Support

For issues, questions, or contributions:
- Check the troubleshooting section
- Review server logs in `logs/`
- Test with standalone `Liveness_AI.py` first

## Version History

- **v1.0**: Initial release with REST API, WebSocket support, and SSL
- Multiple server implementations for different use cases
- Enhanced anti-spoofing with randomized challenges

## Notes

- All servers use the same core detection logic from `Liveness_AI.py`
- `liveness_server.py` is recommended for production use
- SSL certificates are self-signed and meant for development
- For production, use certificates from a trusted CA
- Model file (`lbfmodel.yaml`) is required and must not be deleted

---

**Last Updated**: November 2024
