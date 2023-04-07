import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from preprocess import preprocess, training_data
from sklearn.metrics import multilabel_confusion_matrix

# Actions that we try to detect
ACTIONS = np.array(['Thank_you','Hello','bye-bye'])

#Instance
sequences, labels = preprocess()
X_train, X_val, y_train, y_val, X_test, y_test = training_data(sequences, labels)


# Load model
model = tf.keras.models.load_model('/Users/alifiyabatterywala/Desktop/SLR_project/FinalSLR/model2.h5')

res = model.predict(X_test)
print('original sign',ACTIONS[np.argmax(res[4])])
print('predicted sign',ACTIONS[np.argmax(y_test[4])])

ytrue = np.argmax(y_test, axis=1).tolist()
yhat = np.argmax(res, axis=1).tolist()

confusion_matrix = multilabel_confusion_matrix(ytrue, yhat)
print('confusion matrix',confusion_matrix.shape)

# Iterate over the confusion matrices and plot each one
for i in range(confusion_matrix.shape[0]):
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(confusion_matrix[i], cmap='Blues')
    ax.set_title(f'Confusion Matrix for Class {ACTIONS[i]}')
    cbar = ax.figure.colorbar(im, ax=ax)
    plt.show()

