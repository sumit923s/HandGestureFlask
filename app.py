from flask import Flask, render_template, Response, request, jsonify
import cv2
from hand_gesture_detection import detect_gesture

app = Flask(__name__)

# Function to generate video frames
def generate_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break

        frame_with_gesture = detect_gesture(frame)

        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame_with_gesture + b'\r\n')
 
    cap.release()

# Route to render the index page
@app.route('/')
def index():
    return render_template('index.html')

# Route to stream video frames
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Route to handle gesture detection
@app.route('/detect-gesture', methods=['POST'])
def detect_gesture_route():
    frame_data = request.json.get('frame')
    # Process the frame data using detect_gesture function
    # Example: result = detect_gesture(frame_data)
    # Return the result
    return jsonify({'result': 'Detected gesture'})  # Example response

if __name__ == "__main__":
    app.run(debug=True)
