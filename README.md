# Real Time Sign language recognition 
## Dataset
The project is trained on a dataset consisting of 3 sign actions each containg of 30 videos, with 30 frames each video. Each frame contains 1662 MediaPipe landmarks, which represent the keypoints of the left hand, right hand pose and face, while performing sign language gestures. The dataset is used for training, testing, and validating the LSTM model.

## Dependencies
List of dependencies are mentioned in the `requirement.txt` file

You can install the dependencies using the following commands:

```
pip install -r requirements.txt

```

## How to Install and Run the Project

To use the Sign Language Detection project, follow these steps:

1. Clone the GitHub repository to your local machine.
2. Make sure you have installed all the required dependencies.
3. Run realtime_predict.py to start making real-time sign language predictions using the trained LSTM model.
4. Perform sign language gestures in front of the webcam, and the predicted gestures will be displayed in real-time.
5. You can also modify the project files to train the LSTM model on your own dataset or customize the project as needed.

**Note**: If you want to use the pretrained LSTM model (model.h5) for inference, make sure to update the file path in realtime_predict.py to the correct location of the model.h5 file on your local machine.
