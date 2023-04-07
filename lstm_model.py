import os
import numpy as np

from preprocess import preprocess, training_data
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard

# Actions that we try to detect
ACTIONS = np.array(['Thank_you','Hello','bye-bye'])

# Instance
sequences, labels = preprocess()
X_train, X_val, y_train, y_val, X_test, y_test = training_data(sequences, labels)

model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,1662)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(ACTIONS.shape[0], activation='softmax'))

# Compile model
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# Print model summary
model.summary()

#Tensorflow board
log_dir = os.path.join('Logs2')
tb_callback = TensorBoard(log_dir=log_dir)

# Train model
history = model.fit(X_train, y_train, epochs=80, batch_size=32, validation_data=(X_val, y_val), callbacks=[tb_callback])

# save model
model.save('model2.h5') 
