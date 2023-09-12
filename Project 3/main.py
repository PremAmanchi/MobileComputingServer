import numpy as np
import matplotlib.pyplot as plt
import keras

# Unzip MNIST Dataset JPG format.zip
# First run 'python divide_images.py'
# Then run 'python prepare_data.py'
# Then run this file to create and save model

l1_data = np.load('l1_data.npy', allow_pickle=True)
l2_data = np.load('l2_data.npy', allow_pickle=True)
r1_data = np.load('r1_data.npy', allow_pickle=True)
r2_data = np.load('r2_data.npy', allow_pickle=True)
labels = np.load('labels.npy', allow_pickle=True)

l1_data = np.asarray(l1_data).astype('float32')
l2_data = np.asarray(l2_data).astype('float32')
r1_data = np.asarray(r1_data).astype('float32')
r2_data = np.asarray(r2_data).astype('float32')
labels = np.array(labels, dtype=int)
new_labels = np.zeros((labels.size, 10))
new_labels[np.arange(labels.size), labels] = 1


def getModel():
    return keras.Sequential([
    keras.layers.Input(shape=(14,14)),
    keras.layers.Flatten(),
    keras.layers.Dense(200,activation="relu"),
    keras.layers.Dense(100,activation="relu"),
    keras.layers.Dense(10,activation="softmax")
])


l1_model = getModel()
optimizer = keras.optimizers.Adam(learning_rate=0.0001)
l1_model.compile(optimizer=optimizer, loss='categorical_crossentropy',metrics=['accuracy'])
history = l1_model.fit(l1_data, new_labels, epochs=100)
plt.plot(history.history['accuracy'])
plt.plot(history.history['loss'])
plt.show()
l1_model.save('l1_model_updated')

l2_model = getModel()
optimizer = keras.optimizers.Adam(learning_rate=0.0001)
l2_model.compile(optimizer=optimizer, loss='categorical_crossentropy',metrics=['accuracy'])
history = l2_model.fit(l2_data, new_labels, epochs=100)
plt.plot(history.history['accuracy'])
plt.plot(history.history['loss'])
plt.show()
l2_model.save('l2_model_updated')


r1_model = getModel()
optimizer = keras.optimizers.Adam(learning_rate=0.0001)
r1_model.compile(optimizer=optimizer, loss='categorical_crossentropy',metrics=['accuracy'])
history = r1_model.fit(r1_data, new_labels, epochs=100)
plt.plot(history.history['accuracy'])
plt.plot(history.history['loss'])
plt.show()
r1_model.save('r1_model_updated')



r2_model = getModel()
optimizer = keras.optimizers.Adam(learning_rate=0.0001)
r2_model.compile(optimizer=optimizer, loss='categorical_crossentropy',metrics=['accuracy'])
history = r2_model.fit(r2_data, new_labels, epochs=100)
plt.plot(history.history['accuracy'])
plt.plot(history.history['loss'])
plt.show()
r2_model.save('r2_model_updated')


# ==================== Read images for from mnist jpg format ==============================
# loop through each folder, divide image into 4 parts and save to corresponding folder with subfolders as 1,2,3,4

# train model 1,2,3 and 4 for all 1, 2, 3, and 4 parts
