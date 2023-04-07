import os
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf

from preprocess import preprocess, training_data
from mp_functions import draw_landmarks, extract_keypoints

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils 
mp_drawing_styles = mp.solutions.drawing_styles

# Path where exported data, numpy arrays will be stored
DATA_PATH = os.path.join('/Users/alifiyabatterywala/Desktop/SLR_project/FinalSLR/MP_DATA') 

# Actions that we try to detect
ACTIONS = np.array(['Thank_you','Hello','bye-bye'])

#Instance
sequences, labels = preprocess()
X_train, X_val, y_train, y_val, X_test, y_test = training_data(sequences, labels)

# Load model
model = tf.keras.models.load_model('/Users/alifiyabatterywala/Desktop/SLR_project/FinalSLR/model2.h5')

res = model.predict(X_test)

def prob_viz(res, actions, input_frame):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.putText(output_frame, f'{actions[num]}: {int(prob*100)}%', (10, 60+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
        
    return output_frame

# detection variables
sequence = []
sentence = []
predictions = []
THRESHOLD = 0.5

cap = cv2.VideoCapture(0)

# Set mediapipe model 
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():

        # Read feed
        ret, frame = cap.read()

        # Make detections and get landmarks
        results = holistic.process(frame)
        image, results = draw_landmarks(frame, results)
        
        # Prediction logic
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-30:]
        
        if len(sequence) == 30:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            print(ACTIONS[np.argmax(res)])
            predictions.append(np.argmax(res))
            
            
            # Vizualization logic
            if np.unique(predictions[-10:])[0]==np.argmax(res): 
                if res[np.argmax(res)] > THRESHOLD: 
                    
                    if len(sentence) > 0: 
                        if ACTIONS[np.argmax(res)] != sentence[-1]:
                            sentence.append(ACTIONS[np.argmax(res)])
                    else:
                        sentence.append(ACTIONS[np.argmax(res)])

            if len(sentence) > 5: 
                sentence = sentence[-5:]

            # Vizualize probabilities
            image = prob_viz(res, ACTIONS, image)

            # Display only one word at a time
            if len(sentence) > 0:
                cv2.putText(image, sentence[-1], (3,30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
        
        # Show to screen
        cv2.imshow('OpenCV Feed', image)

        # Break gracefully
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

