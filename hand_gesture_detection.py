import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf

# Load the gesture recognizer model
model = tf.keras.models.load_model('mp_hand_gesture')  # Path to the directory containing the SavedModel

# Load class names
with open('gesture.names', 'r') as f:
    classNames = f.read().split('\n')

# Initialize mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

def detect_gesture(frame):
    x, y, c = frame.shape
    frame = cv2.flip(frame, 1)
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(framergb)
    className = ''

    if result.multi_hand_landmarks:
        landmarks = []
        for handslms in result.multi_hand_landmarks:
            for lm in handslms.landmark:
                lmx = int(lm.x * x)
                lmy = int(lm.y * y)
                landmarks.append([lmx, lmy])

            mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)

            prediction = model.predict(np.array([landmarks]))  # Convert to numpy array
            classID = np.argmax(prediction)
            className = classNames[classID]

    cv2.putText(frame, className, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 
               1, (0,0,255), 2, cv2.LINE_AA)

    ret, buffer = cv2.imencode('.jpg', frame)
    frame = buffer.tobytes()

    return frame
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf

# Load the gesture recognizer model
model = tf.keras.models.load_model('mp_hand_gesture')  # Path to the directory containing the SavedModel

# Load class names
with open('gesture.names', 'r') as f:
    classNames = f.read().split('\n')

# Initialize mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

def detect_gesture(frame):
    x, y, c = frame.shape
    frame = cv2.flip(frame, 1)
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(framergb)
    className = ''

    if result.multi_hand_landmarks:
        landmarks = []
        for handslms in result.multi_hand_landmarks:
            for lm in handslms.landmark:
                lmx = int(lm.x * x)
                lmy = int(lm.y * y)
                landmarks.append([lmx, lmy])

            mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)

            prediction = model.predict(np.array([landmarks]))  # Convert to numpy array
            classID = np.argmax(prediction)
            className = classNames[classID]

    cv2.putText(frame, className, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 
               1, (0,0,255), 2, cv2.LINE_AA)

    ret, buffer = cv2.imencode('.jpg', frame)
    frame = buffer.tobytes()

    return frame
