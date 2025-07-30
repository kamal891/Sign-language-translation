import cv2
import mediapipe as mp
import os
import csv

# === Setup ===
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

# === Output CSV ===
output_csv = 'data/kaggle_landmarks.csv'

# === Your ASL images ===
dataset_path = 'C:/Users/Bhaskar/.cache/kagglehub/grassknoted/asl-alphabet/v1/asl_alphabet_train'  # <-- adjust this if needed

# === Prepare CSV ===
with open(output_csv, mode='w', newline='') as f:
    writer = csv.writer(f)
    # Write header
    header = []
    for i in range(21):
        header += [f'x{i}', f'y{i}']
    header.append('label')
    writer.writerow(header)

    # Loop through folders A-Z
    for label in os.listdir(dataset_path):
        folder_path = os.path.join(dataset_path, label)
        if not os.path.isdir(folder_path):
            continue

        print(f'Processing {label} ...')

        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)
            image = cv2.imread(img_path)
            if image is None:
                continue

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            result = hands.process(image_rgb)

            if result.multi_hand_landmarks:
                landmarks = result.multi_hand_landmarks[0]
                row = []
                for lm in landmarks.landmark:
                    row += [lm.x, lm.y]
                row.append(label)
                writer.writerow(row)

print(f'✅ Landmarks saved to {output_csv}')
