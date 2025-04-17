import cv2
import mediapipe as mp
import pygame
import os

# ------------------------- Inicialización -------------------------

# Inicializar pygame mixer para reproducir sonidos
pygame.mixer.init()
path = os.path.join(os.getcwd(), 'sounds')

# Cargar sonidos para cada dedo y cada mano.
sounds = {
    "Left": {
        "thumb": pygame.mixer.Sound(f"{path}/left_thumb.wav"),
        "index": pygame.mixer.Sound(f"{path}/left_index.wav"),
        "middle": pygame.mixer.Sound(f"{path}/left_middle.wav"),
        "ring": pygame.mixer.Sound(f"{path}/left_ring.wav"),
        "pinky": pygame.mixer.Sound(f"{path}/left_pinky.wav")
    },
    "Right": {
        "thumb": pygame.mixer.Sound(f"{path}/right_thumb.wav"),
        "index": pygame.mixer.Sound(f"{path}/right_index.wav"),
        "middle": pygame.mixer.Sound(f"{path}/right_middle.wav"),
        "ring": pygame.mixer.Sound(f"{path}/right_ring.wav"),
        "pinky": pygame.mixer.Sound(f"{path}/right_pinky.wav")
    }
}

# Configurar MediaPipe Hands
mp_hands = mp.solutions.hands
hands_detector = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# Iniciar la captura de video
cap = cv2.VideoCapture(0)

# Diccionario para almacenar el estado previo de cada dedo (por mano)
# False: dedo "arriba" (no presionado); True: dedo "abajo" (presionado)
finger_states = {
    "Left": {"thumb": False, "index": False, "middle": False, "ring": False, "pinky": False},
    "Right": {"thumb": False, "index": False, "middle": False, "ring": False, "pinky": False}
}

# Índices de landmarks para cada dedo:
# Para el pulgar: tip (4) y su articulación IP (3)
# Para los demás dedos: tip (8, 12, 16, 20) y PIP (6, 10, 14, 18) respectivamente.
finger_tips = {"thumb": 4, "index": 8, "middle": 12, "ring": 16, "pinky": 20}
finger_pips = {"thumb": 3, "index": 6, "middle": 10, "ring": 14, "pinky": 18}

# ------------------------- Bucle Principal -------------------------

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Voltear la imagen horizontalmente para obtener efecto espejo
    frame = cv2.flip(frame, 1)

    # Convertir la imagen de BGR a RGB para MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Procesar la imagen para detectar manos
    results = hands_detector.process(rgb_frame)

    # Verificar si se detectaron manos
    if results.multi_hand_landmarks and results.multi_handedness:
        # Se recorre cada mano detectada junto con su clasificación (Left/Right)
        for hand_landmarks, hand_handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            # Obtener la etiqueta de la mano ("Left" o "Right")
            hand_label = hand_handedness.classification[0].label

            # Dibujar los landmarks y las conexiones en la imagen
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extraer la lista de landmarks (cada uno con atributos x, y, z)
            landmarks = hand_landmarks.landmark

            # Diccionario para guardar el estado actual de cada dedo en esta mano
            current_states = {}

            # --- Detección para el pulgar ---
            # Para el pulgar se utiliza la coordenada x, ya que su movimiento es horizontal.
            # La lógica es: para la mano derecha, si el tip del pulgar está a la izquierda del IP, el pulgar está extendido (arriba).
            # Para la mano izquierda, la condición se invierte.
            if hand_label == "Right":
                if landmarks[finger_tips["thumb"]].x < landmarks[finger_pips["thumb"]].x:
                    current_states["thumb"] = False  # Pulgar arriba
                else:
                    current_states["thumb"] = True   # Pulgar abajo (presionado)
            else:  # Mano izquierda
                if landmarks[finger_tips["thumb"]].x > landmarks[finger_pips["thumb"]].x:
                    current_states["thumb"] = False  # Pulgar arriba
                else:
                    current_states["thumb"] = True   # Pulgar abajo

            # --- Detección para los otros dedos ---
            # Se compara la coordenada y del tip y la del PIP.
            # Si el tip está más abajo (mayor valor de y) que la articulación PIP, se asume que el dedo está doblado (abajo).
            for finger in ["index", "middle", "ring", "pinky"]:
                if landmarks[finger_tips[finger]].y > landmarks[finger_pips[finger]].y:
                    current_states[finger] = True   # Dedo abajo (presionado)
                else:
                    current_states[finger] = False  # Dedo arriba

            # --- Comparar con el estado previo para detectar transiciones de "arriba" a "abajo" ---
            for finger in current_states:
                # Si anteriormente el dedo estaba arriba (False) y ahora se detecta como abajo (True)
                if not finger_states[hand_label][finger] and current_states[finger]:
                    print(f"{hand_label} {finger} presionado")
                    # Reproducir el sonido asociado a este dedo
                    sounds[hand_label][finger].play()

                # Actualizar el estado previo con el estado actual
                finger_states[hand_label][finger] = current_states[finger]

            # --- Opcional: Mostrar en pantalla el estado de cada dedo ---
            y0 = 30
            dy = 20
            for idx, finger in enumerate(["thumb", "index", "middle", "ring", "pinky"]):
                text = f"{hand_label} {finger}: {'Abajo' if current_states[finger] else 'Arriba'}"
                cv2.putText(frame, text, (10, y0 + idx * dy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # Mostrar la imagen con los landmarks y estados
    cv2.imshow("Piano con Vision Artificial", frame)

    # Salir al presionar la tecla ESC (código 27)
    key = cv2.waitKey(1)
    if key == 27:
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
