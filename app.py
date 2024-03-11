from flask import Flask, render_template, Response
import cv2
from hand_gesture_detection import detect_gesture

app = Flask(__name__)

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

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# if __name__ == "__main__":
#     app.run(debug=True)
