import cv2
import mediapipe as mp
import numpy as np
import joblib

# === Load trained model ===
model = joblib.load('sign_model.pkl')

# === Setup MediaPipe ===
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# === Start webcam ===
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Draw landmarks
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract landmark list
            data = []
            for lm in hand_landmarks.landmark:
                data.extend([lm.x, lm.y, lm.z])

            # Convert to numpy & reshape
            X = np.array(data).reshape(1, -1)

            # Predict
            pred = model.predict(X)[0]

            # Display prediction
            cv2.putText(frame, f'Prediction: {pred}', (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Sign Language Translator", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
word_map = {
    'H': 'Hi',
    'E': 'Hello',
    'T': 'Thanks',
    'Y': 'Yes',
    'N': 'No'
}

# Then after predicting:
word = word_map.get(pred, pred)

cv2.putText(frame, f'Prediction: {word}', (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
