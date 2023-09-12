import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def classify(path):
    image = cv2.imread(path)
    grey = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(grey.copy(), 125, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    preprocessed_digits = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(image, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=2)
        digit = thresh[y:y + h, x:x + w]
        resized_digit = cv2.resize(digit, (18, 18))
        padded_digit = np.pad(resized_digit, ((5, 5), (5, 5)), "constant", constant_values=0)
        preprocessed_digits.append(padded_digit)

    inp = np.array(preprocessed_digits)

    model = tf.keras.models.load_model('my_model')
    probabilities = []
    predictions = []
    for digit in inp:
        prediction = model.predict(digit.reshape(1, 28, 28, 1))
        print("\n---------------------------------------")
        print(prediction)
        print("=========PREDICTION============")
        res = np.argmax(prediction)
        predictions.append(res)
        probabilities.append(np.max(prediction))

    print("\n\nprobabilities",probabilities)
    print("predictions",predictions)

    return predictions[np.argmax(probabilities)]

# path = "/home/r00tus3r/asu/cse535/Proj1/mobile-computing-server/static/images/[3]/3.jpeg"
print("Res:", classify('6.jpeg'))
