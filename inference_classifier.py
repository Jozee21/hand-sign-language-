import pickle

import cv2
import mediapipe as mp
import numpy as np

model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {0: 'L Right', 1: 'L Left', 2: 'G Right', 3: 'G Left', 4: 'M Right', 5: 'M Left', 6: 'H Right', 7: 'H Left', 8: 'X Right', 9: 'X Left', 10: 'Right Hand', 11: 'Left Hand', 12: 'Right Hand', 13: 'Left Hand', 14: 'Right Hand', 15: 'Left Hand', 16: 'Right Hand', 17:'Left Hand', 18: 'Right Hand', 19: 'Left Hand', 20: 'Right Hand', 21: 'Left Hand', 22: 'Right Hand', 23: 'Left Hand', 24:'Right Hand', 25:'Left Hand', 26: 'Right Hand', 27:'Left Hand', 28: 'Right Hand', 29: 'Left Hand', 30: 'Right Hand', 31: 'Left Hand', 32:'Right Hand', 33:'Left Hand', 34:'Right Hand', 35: 'Left Hand', 36: 'Right Hand', 37: 'Left Hand', 38: 'Right Hand', 39: 'Left Hand', 40:'Right Hand', 41:'Left Hand', 42:'Right Hand', 43:'Left Hand', 44: 'Right Hand', 45: 'Left Hand', 46:'Right Hand', 47:'Left Hand', 48: 'Right Hand', 49: 'Left Hand', 50: 'Right Hand', 51: 'Left Hand'} 
while True:

    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()

    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,  # image to draw
                hand_landmarks,  # model output
                mp_hands.HAND_CONNECTIONS,  # hand connections
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10

        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10

        prediction = model.predict([np.asarray(data_aux)])

        predicted_character = labels_dict[int(prediction[0])]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3,
                    cv2.LINE_AA)

    cv2.imshow('frame', frame)
    cv2.waitKey(1)


cap.release()
cv2.destroyAllWindows()