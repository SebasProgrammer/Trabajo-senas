# server.py
from flask import Flask, render_template, Response
from flask_cors import CORS
import cv2
import mediapipe as mp
import math

app = Flask(__name__)
CORS(app)  # O el origen de tu frontend


# Inicializar Mediapipe
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)

def mano_abierta(hand_landmarks, image_height):
    """
    Verifica si la mano está abierta para indicar "Hola" y si la mano está por encima del mentón.
    """
    middle_finger_tip = hand_landmarks.landmark[12]  # Punta del dedo medio
    chin_threshold = image_height / 3  # Umbral para la "zona del mentón"

    # Si la punta del dedo medio está por debajo del umbral, la mano no está por encima del mentón
    if middle_finger_tip.y * image_height > chin_threshold:
        return False

    finger_tips = [8, 12, 16, 20]  # Puntas de los dedos
    finger_pips = [6, 10, 14, 18]  # Articulaciones PIP correspondientes

    # Comprueba si cada punta de dedo está por encima de su articulación PIP
    fingers_extended = all(hand_landmarks.landmark[tip].y < hand_landmarks.landmark[pip].y for tip, pip in zip(finger_tips, finger_pips))

    # Retorna True si todos los dedos están extendidos (mano abierta) y por encima del mentón
    return fingers_extended

def detectar_por_favor(hand_landmarks, image_width, image_height):
    """
    Detecta el gesto de "Por Favor" - palma abierta y mano debajo del mentón.
    """
    finger_tips = [8, 12, 16, 20]  # Puntas de dedo (excepto el pulgar)
    chin_threshold = image_height / 2  # Umbral para la "zona del mentón"

    # Verifica si la palma está abierta (dedos extendidos) y la mano está debajo del mentón
    open_hand = all(hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y for tip in finger_tips)
    
    # Verifica si la mano está debajo del mentón
    middle_finger_tip = hand_landmarks.landmark[12]
    below_chin = middle_finger_tip.y * image_height > chin_threshold

    return open_hand and below_chin

def detectar_cuanto(hand_landmarks, image_height, image_width):
    """
    Detecta el gesto de "Cuanto" con las manos orientadas hacia arriba,
    dedos levemente doblados y palmas hacia el frente.
    """
    finger_tips = [8, 12, 16, 20]
    finger_pips = [6, 10, 14, 18]
    
    # Dedos ligeramente doblados
    fingers_bent = all(hand_landmarks.landmark[tip].y > hand_landmarks.landmark[pip].y * 0.95
                       for tip, pip in zip(finger_tips, finger_pips))

    # Palma orientada hacia el frente
    wrist_y = hand_landmarks.landmark[0].y * image_height
    index_y = hand_landmarks.landmark[5].y * image_height
    palm_facing_front = wrist_y > index_y  # Indica que la palma está hacia arriba

    return fingers_bent and palm_facing_front

def detectar_cuesta(hand_landmarks, image_width, image_height):
    """
    Detecta el gesto de "Cuesta" con los dedos índice y pulgar formando una pinza.
    """
    thumb_tip = hand_landmarks.landmark[4]
    index_tip = hand_landmarks.landmark[8]

    # Calcula la distancia euclidiana entre el pulgar y el índice
    distance = math.sqrt(
        (thumb_tip.x - index_tip.x) ** 2 * image_width ** 2 + 
        (thumb_tip.y - index_tip.y) ** 2 * image_height ** 2
    )

    # Define un umbral para determinar si están formando la pinza
    return distance < 20  # Ajusta este valor de umbral según sea necesario

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

# Aquí puedes incluir las demás funciones de detección

def generate_frames():
    with mp_hands.Hands(model_complexity=1, min_detection_confidence=0.7, min_tracking_confidence=0.7, max_num_hands=2) as hands:
        while True:
            success, image = cap.read()
            if not success:
                break
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            image_height, image_width, _ = image.shape

            if results.multi_hand_landmarks:
                num_hands = len(results.multi_hand_landmarks)  # Contar el número de manos detectadas

                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )
                    
                    # Detectar "Hola" y "Por Favor" solo si hay una mano detectada
                    if num_hands == 1:
                        if mano_abierta(hand_landmarks, image_height):
                            cv2.putText(image, "Hola!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
                        
                        if detectar_por_favor(hand_landmarks, image_width, image_height):
                            cv2.putText(image, "Por Favor", (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 165, 0), 4, cv2.LINE_AA)

                        if detectar_cuesta(hand_landmarks, image_width, image_height):
                            cv2.putText(image, "Cuesta", (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 4, cv2.LINE_AA)

                    if num_hands == 2:
                        # Detectar "Cuanto" independientemente del número de manos
                        if detectar_cuanto(hand_landmarks, image_height, image_width):
                            cv2.putText(image, "Cuanto", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4, cv2.LINE_AA)

                        if detectar_laptop(results.multi_hand_landmarks, image_width, image_height):
                            cv2.putText(image, "Laptop", (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 4, cv2.LINE_AA)

            # Codificar la imagen para enviar al frontend
            ret, buffer = cv2.imencode('.jpg', image)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(port=5005,debug=True)
