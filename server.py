# server.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import mediapipe as mp
import numpy as np
import base64
import math

app = Flask(__name__)
CORS(app)

# Inicializar MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(model_complexity=1, min_detection_confidence=0.7, min_tracking_confidence=0.7, max_num_hands=2)

def decode_image(base64_string):
    """
    Decodifica una imagen en base64 a un array de OpenCV (NumPy).
    """
    image_data = base64.b64decode(base64_string.split(",")[1])  # Quitar el encabezado "data:image/jpeg;base64,"
    np_array = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    return image

def mano_abierta(hand_landmarks, image_height):
    """
    Detecta el gesto de "Hola" si la mano está abierta y por encima del mentón.
    """
    middle_finger_tip = hand_landmarks.landmark[12]
    chin_threshold = image_height / 3
    if middle_finger_tip.y * image_height > chin_threshold:
        return False
    finger_tips = [8, 12, 16, 20]
    finger_pips = [6, 10, 14, 18]
    return all(hand_landmarks.landmark[tip].y < hand_landmarks.landmark[pip].y for tip, pip in zip(finger_tips, finger_pips))

def detectar_por_favor(hand_landmarks, image_height):
    """
    Detecta el gesto de "Por Favor" con la palma abierta y la mano debajo del mentón.
    """
    finger_tips = [8, 12, 16, 20]
    chin_threshold = image_height / 2
    open_hand = all(hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y for tip in finger_tips)
    middle_finger_tip = hand_landmarks.landmark[12]
    below_chin = middle_finger_tip.y * image_height > chin_threshold
    return open_hand and below_chin

def detectar_cuanto(hand_landmarks, image_height):
    """
    Detecta el gesto de "Cuanto" con los dedos ligeramente doblados y la palma hacia el frente.
    """
    finger_tips = [8, 12, 16, 20]
    finger_pips = [6, 10, 14, 18]
    fingers_bent = all(hand_landmarks.landmark[tip].y > hand_landmarks.landmark[pip].y * 0.95
                       for tip, pip in zip(finger_tips, finger_pips))
    wrist_y = hand_landmarks.landmark[0].y * image_height
    index_y = hand_landmarks.landmark[5].y * image_height
    palm_facing_front = wrist_y > index_y
    return fingers_bent and palm_facing_front

def detectar_cuesta(hand_landmarks, image_height):
    """
    Detecta el gesto de "Cuesta" cuando los dedos índice y pulgar están formando una pinza.
    """
    thumb_tip = hand_landmarks.landmark[4]
    index_tip = hand_landmarks.landmark[8]
    distance = math.sqrt(
        (thumb_tip.x - index_tip.x) ** 2 * image_height ** 2 + 
        (thumb_tip.y - index_tip.y) ** 2 * image_height ** 2
    )
    return distance < 20

def detectar_laptop(hand_landmarks_list, image_width, image_height):
    if len(hand_landmarks_list) < 2:
        return False
    
    hand1_landmarks = hand_landmarks_list[0]
    hand2_landmarks = hand_landmarks_list[1]
    min_distance_threshold = image_width * 0.3

    finger_tips = [8, 12, 16, 20]
    open_hand1 = all(hand1_landmarks.landmark[tip].y < hand1_landmarks.landmark[tip - 2].y for tip in finger_tips)
    open_hand2 = all(hand2_landmarks.landmark[tip].y < hand2_landmarks.landmark[tip - 2].y for tip in finger_tips)

    hand1_palm_facing_forward = hand1_landmarks.landmark[5].y < hand1_landmarks.landmark[0].y
    hand2_palm_facing_forward = hand2_landmarks.landmark[5].y < hand2_landmarks.landmark[0].y

    hand1_x = hand1_landmarks.landmark[0].x * image_width
    hand2_x = hand2_landmarks.landmark[0].x * image_width
    horizontal_distance = abs(hand1_x - hand2_x)

    return (
        open_hand1 and open_hand2 and
        hand1_palm_facing_forward and hand2_palm_facing_forward and
        horizontal_distance > min_distance_threshold
    )

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    base64_image = data.get("image")

    if not base64_image:
        return jsonify({"error": "No image data provided"}), 400

    # Decodificar la imagen y obtener sus dimensiones
    image = decode_image(base64_image)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_height, image_width, _ = image.shape

    # Procesar la imagen con MediaPipe
    results = hands.process(image_rgb)

    # Lista para almacenar los gestos detectados
    detections = []

    if results.multi_hand_landmarks:
        num_hands = len(results.multi_hand_landmarks) 
        for hand_landmarks in results.multi_hand_landmarks:
            if num_hands == 1:
                if mano_abierta(hand_landmarks, image_height):
                    detections.append("Hola")
                if detectar_por_favor(hand_landmarks, image_height):
                    detections.append("Por Favor")
                if detectar_cuesta(hand_landmarks, image_height):
                    detections.append("Cuesta")

            if num_hands == 2:
                if detectar_cuanto(hand_landmarks, image_height):
                    detections.append("Cuanto")

                if detectar_laptop(results.multi_hand_landmarks, image_width, image_height):
                    detections.append("Laptop")

    return jsonify({"detections": detections})

if __name__ == "__main__":
    pass
