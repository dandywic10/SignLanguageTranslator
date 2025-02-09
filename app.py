# Backend (app.py)
from flask import Flask, render_template, Response, jsonify
import cv2
import mediapipe as mp
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Dictionary untuk 10 gesture
gesture_dict = {
    'thumbs_up': 'Halo',
    'palm_open': 'Terima Kasih',
    'fist': 'Tolong',
    'peace': 'Selamat Tinggal',
    'pointing': 'Ya',
    'thumbs_down': 'Tidak',
    'ok_sign': 'Oke',
    'phone_hand': 'Telepon',
    'wave_hand': 'Sampai Jumpa',
    'eat_gesture': 'Makan'
}

def detect_gesture(hand_landmarks):
    # Implementasi logika deteksi yang lebih kompleks
    landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
    
    # Definisi titik-titik landmark penting
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    middle_tip = landmarks[12]
    ring_tip = landmarks[16]
    pinky_tip = landmarks[20]
    
    # Deteksi gesture berdasarkan posisi jari
    if thumb_tip[1] < landmarks[3][1] and all(landmarks[tip][1] > landmarks[tip-2][1] for tip in [8,12,16,20]):
        return 'thumbs_up'
    elif all(landmarks[tip][1] < landmarks[tip-2][1] for tip in [8,12,16,20]):
        return 'palm_open'
    elif all(landmarks[tip][1] > landmarks[tip-2][1] for tip in [8,12,16,20]):
        return 'fist'
    elif landmarks[8][1] < landmarks[6][1] and landmarks[12][1] < landmarks[10][1] and landmarks[16][1] > landmarks[14][1]:
        return 'peace'
    elif landmarks[8][1] < landmarks[6][1] and all(landmarks[tip][1] > landmarks[tip-2][1] for tip in [12,16,20]):
        return 'pointing'
    elif thumb_tip[1] > landmarks[3][1] and all(landmarks[tip][1] > landmarks[tip-2][1] for tip in [8,12,16,20]):
        return 'thumbs_down'
    elif distance(thumb_tip, index_tip) < 0.05 and all(landmarks[tip][1] < landmarks[tip-2][1] for tip in [12,16,20]):
        return 'ok_sign'
    elif thumb_tip[2] < landmarks[3][2] and index_tip[2] < landmarks[6][2] and all(landmarks[tip][1] > landmarks[tip-2][1] for tip in [12,16,20]):
        return 'phone_hand'
    elif all(landmarks[tip][1] < landmarks[tip-2][1] for tip in [8,12,16,20]) and max(abs(landmarks[tip][0] - landmarks[0][0]) for tip in [8,12,16,20]) > 0.3:
        return 'wave_hand'
    elif thumb_tip[2] < landmarks[3][2] and all(landmarks[tip][1] < landmarks[tip-2][1] for tip in [8,12,16]) and landmarks[20][1] > landmarks[18][1]:
        return 'eat_gesture'
    return None

def distance(p1, p2):
    return np.sqrt(np.sum((p1 - p2) ** 2))

def generate_frames():
    camera = cv2.VideoCapture(0)
    while True:
        success, frame = camera.read()
        if not success:
            break
        
        # Proses frame
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                gesture = detect_gesture(hand_landmarks)
                if gesture:
                    cv2.putText(frame, f"Gesture: {gesture_dict.get(gesture, 'Unknown')}", 
                              (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)

