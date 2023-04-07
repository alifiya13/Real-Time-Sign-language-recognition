import os
import cv2
import numpy as np
import mediapipe as mp

from mp_functions import draw_landmarks, extract_keypoints

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils 
mp_drawing_styles = mp.solutions.drawing_styles

# Path where exported data, numpy arrays will be stored
DATA_PATH = os.path.join('/Users/alifiyabatterywala/Desktop/SLR_project/FinalSLR/MP_DATA') 

# Actions that we try to detect
ACTIONS = np.array(['Thank_you','Hello','bye-bye'])

# Thirty videos 
NO_SEQ = 30

# Videos are going to be 30 frames in length
SEQ_LENGTH = 30

# mediapipe to extract keypoints 
cap = cv2.VideoCapture(0)
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:

  for action in ACTIONS:
    action_path = os.path.join(DATA_PATH, action)
   
    if not os.path.exists(action_path):
        os.mkdir(action_path)
   
    for sequence in range(NO_SEQ):       
        sequence_path = os.path.join(action_path, str(sequence))
       
        if not os.path.exists(sequence_path):
            os.mkdir(sequence_path)

            # Loop through video length aka sequence length
        for frame_num in range(SEQ_LENGTH):

          # Read feed
          ret, frame = cap.read()

          # Make detections and get landmarks
          results = holistic.process(frame)
          image, results = draw_landmarks(frame, results)
                
                # NEW Apply wait logic
          if frame_num == 0: 
              cv2.putText(image, 'STARTING COLLECTION', (120,200), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
              cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15,12), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
              # Show to screen
              cv2.imshow('OpenCV Feed', image)
              cv2.waitKey(500)
          else: 
              cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15,12), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
              # Show to screen
              cv2.imshow('OpenCV Feed', image)
                
          # NEW Export keypoints
          keypoints = extract_keypoints(results)
          npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
          np.save(npy_path, keypoints)

          # Break gracefully
          if cv2.waitKey(10) & 0xFF == ord('q'):
              break
                    
  cap.release()
  cv2.destroyAllWindows()
