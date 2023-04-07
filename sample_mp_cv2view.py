import cv2
import mediapipe as mp
from mp_functions import draw_landmarks

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils 
mp_drawing_styles = mp.solutions.drawing_styles

#setting up mediapipe model for sample view of landmarks on video
cap = cv2.VideoCapture(0) 

with mp_holistic.Holistic(
    static_image_mode = False, min_detection_confidence = 0.5, model_complexity = 1) as holistic:

    while cap.isOpened():
      ret, frame = cap.read()
      if not ret:
        break
      
      #make detection and draw lanmarks
      results = holistic.process(frame)
      image, results = draw_landmarks(frame,results)

      cv2.imshow('sample keypoints view',image)


      if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()