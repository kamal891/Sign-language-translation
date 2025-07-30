import cv2
import mediapipe as mp
import csv

# Setup webcam and Mediapipe
cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

# Open CSV file to save data
file = open('data/sign_data.csv', mode='w', newline='')
csv_writer = csv.writer(file)
# Write header row
csv_writer.writerow(['label'] + [f'x{i}' for i in range(21)] + [f'y{i}' for i in range(21)] + [f'z{i}' for i in range(21)])

print("Press a letter key (A-Z) when your hand is in the right sign position.")
print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(image)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Wait for key press for label
            key = cv2.waitKey(1) & 0xFF
            if key != 255:
                if key == ord('q'):
                    cap.release()
                    file.close()
                    cv2.destroyAllWindows()
                    print("Finished collecting data.")
                    exit()
                else:
                    # Convert key to uppercase letter
                    label = chr(key).upper()
                    landmarks = []
                    for lm in hand_landmarks.landmark:
                        landmarks.extend([lm.x, lm.y, lm.z])
                    # Save to CSV
                    row = [label] + landmarks
                    csv_writer.writerow(row)
                    print(f"Saved sign for label: {label}")

    cv2.imshow('Collecting Data', frame)

