import os
import numpy as np

from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# Path for exported data, numpy arrays
DATA_PATH = os.path.join('/Users/alifiyabatterywala/Desktop/SLR_project/FinalSLR/MP_DATA') 

# Actions that we try to detect
ACTIONS = np.array(['Thank_you','Hello','bye-bye'])

# Thirty videos 
NO_SEQ = 30

# Videos are going to be 30 frames in length
SEQ_LENGTH = 30

LABEL_MAP = {label:num for num, label in enumerate(ACTIONS)}

def preprocess():
    sequences, labels = [], []
    for action in ACTIONS:
        for sequence in np.array(os.listdir(os.path.join(DATA_PATH, action))).astype(int):
            window = []
            for frame_num in range(SEQ_LENGTH):
                res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
                window.append(res)
            sequences.append(window)
            labels.append(LABEL_MAP[action])
    return sequences, labels

sequences, labels = preprocess()
print('sequence shape', np.array(sequences).shape)
print('label shape', np.array(labels).shape)

def training_data(sequences, labels):
    X = np.array(sequences)
    Y = to_categorical(labels).astype(int)

# Split data into training 90% and testing 10%
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.10, random_state=42)
# Split training set 90% into training 80% and validation 10%
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.10, random_state=42)
    return X_train, X_val, y_train, y_val, X_test, y_test

X_train, X_val, y_train, y_val, X_test, y_test = training_data(sequences, labels)

print('X_train shape = ',X_train.shape,'X_test shape = ',X_test.shape, 'X_val shape = ',X_val.shape)
print('y_train shape = ',y_train.shape,'y_test shape = ',y_test.shape, 'y_val shape = ',y_val.shape)