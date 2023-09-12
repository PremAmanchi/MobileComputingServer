import cv2
import os
import numpy as np

my_path = os.path.abspath(os.path.dirname(__file__))
imgs_path = my_path + '\Divided Images'

def prepare_data():
    l1_data = []
    l2_data = []
    r1_data = []
    r2_data = []
    labels = []
    for i in range(10):
        dir = os.path.join(imgs_path,  str(i))
        for digits in os.listdir(dir):
            digit_folder = os.path.join(dir, digits)
            # read l1
            l1_file = os.path.join(digit_folder, 'l1.jpg')
            if os.path.isfile(l1_file):
                img = cv2.imread(l1_file)
                grey = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
                l1_data.append(grey)
            l2_file = os.path.join(digit_folder, 'l2.jpg')
            if os.path.isfile(l2_file):
                img = cv2.imread(l2_file)
                grey = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
                l2_data.append(grey)
            r1_file = os.path.join(digit_folder, 'r1.jpg')
            if os.path.isfile(r1_file):
                img = cv2.imread(r1_file)
                grey = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
                r1_data.append(grey)
            r2_file = os.path.join(digit_folder, 'r2.jpg')
            if os.path.isfile(r2_file):
                img = cv2.imread(r2_file)
                grey = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
                r2_data.append(grey)
            labels.append(i)
    return np.array(l1_data), np.array(l2_data), np.array(r1_data), np.array(r2_data), np.array(labels)


l1_data, l2_data, r1_data, r2_data, labels = prepare_data()

np.save('l1_data.npy', np.array(l1_data, dtype=object), True)
np.save('l2_data.npy', np.array(l2_data, dtype=object), True)
np.save('r1_data.npy', np.array(r1_data, dtype=object), True)
np.save('r2_data.npy', np.array(r2_data, dtype=object), True)
np.save('labels.npy', np.array(labels, dtype=object), True)