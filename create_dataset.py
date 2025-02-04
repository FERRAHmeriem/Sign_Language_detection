import os
import cv2
import mediapipe as mp
import pickle

DATA_DIR = './collected_data'

# Detecting hands using MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Initializing the data
data = []
labels = []

for dir_ in os.listdir(DATA_DIR):
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        if img is None:
            print(f"Warning: Unable to read image {img_path} in directory {dir_}")
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = hands.process(img_rgb)
        data_aux = []
        x_ = []
        y_ = []

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                # Uncomment to visualize the hand landmarks
                # mp_drawing.draw_landmarks(
                #     img_rgb,
                #     hand_landmarks,
                #     mp_hands.HAND_CONNECTIONS,
                #     mp_drawing_styles.get_default_hand_landmarks_style(),
                #     mp_drawing_styles.get_default_hand_connections_style(),
                # )

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_.append(x)
                    y_.append(y)

                # Normalize the hand position
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

            data.append(data_aux)
            labels.append(dir_)
# Save the data to a pickle file
with open('./data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print("Data collection and processing complete.")