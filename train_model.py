
'''
input > weight > hidden layer 1 (activation function) > weights  > hidden layer 2
(activation function) > weights > output layer

straight through feed forward

compare the output to the intended output > cost or loss function (cross entropy)
optimization function (optimizer) > minimize cost (AdamOptimizer...SGD, AdaGrad)
this optimizer goes backwards and manipulates the weights > backpropagation

feed forward + backprop = epoch
'''
import keras
import tensorflow as tf
# tf.disable_v2_behavior()
import numpy as np
import idx2numpy
import matplotlib.pyplot as plt

image_file = 'MNIST Data/train-images.idx3-ubyte'
train_images = idx2numpy.convert_from_file(image_file)
label_file = 'MNIST Data/train-labels.idx1-ubyte'
raw_labels = idx2numpy.convert_from_file(label_file)
# converting to one hot
train_labels = np.zeros((raw_labels.size, raw_labels.max() + 1))
train_labels[np.arange(raw_labels.size), raw_labels] = 1


image_file = 'MNIST Data/t10k-images.idx3-ubyte'
test_images = idx2numpy.convert_from_file(image_file)
label_file = 'MNIST Data/t10k-labels.idx1-ubyte'
raw_labels = idx2numpy.convert_from_file(label_file)
# converting to one hot
test_labels = np.zeros((raw_labels.size, raw_labels.max() + 1))
test_labels[np.arange(raw_labels.size), raw_labels] = 1

# 10 classes, 0-9
'''
0 = [1,0,0,0,0,0,0,0,0,0]
1 = [0,1,0,0,0,0,0,0,0,0]
.
.
.
'''

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(500, activation='relu'),
    keras.layers.Dense(500, activation='relu'),
    keras.layers.Dense(10, activation='softmax'),
])

model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=10)

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=1)

print("Test accuracy", test_acc)

model.save('my_model')