import os
import cv2

my_path = os.path.abspath(os.path.dirname(__file__))
path = my_path + '\Divided Images'
UPLOAD_FOLDER = 'Divided Images'

def divideimg(subfolder,filename, img):
    loc = UPLOAD_FOLDER + '/' + subfolder + '/' + filename
    if not os.path.isdir(loc):
        os.mkdir(loc)

    height = img.shape[0]
    width = img.shape[1]

    width_cutoff = width // 2
    left1 = img[:, :width_cutoff]
    right1 = img[:, width_cutoff:]

    img = cv2.rotate(left1, cv2.ROTATE_90_CLOCKWISE)
    height = img.shape[0]
    width = img.shape[1]

    width_cutoff = width // 2
    l1 = img[:, :width_cutoff]
    l2 = img[:, width_cutoff:]
    l1 = cv2.rotate(l1, cv2.ROTATE_90_COUNTERCLOCKWISE)
    # cv2.imwrite("l1.jpg", l1)
    l1_path = os.path.join(loc, 'l1.jpg')
    cv2.imwrite(l1_path, l1)
    print("l1 saved at === ",l1_path)
    # cv2.imshow('L1', l1)
    l2 = cv2.rotate(l2, cv2.ROTATE_90_COUNTERCLOCKWISE)
    # cv2.imwrite("l2.jpg", l2)
    l2_path = os.path.join(loc,  'l2.jpg')
    cv2.imwrite(l2_path, l2)
    print("l2 saved at ===", l2_path)
    # cv2.imshow('L2', l2)
    img = cv2.rotate(right1, cv2.ROTATE_90_CLOCKWISE)
    height = img.shape[0]
    width = img.shape[1]
    width_cutoff = width // 2
    r1 = img[:, :width_cutoff]
    r2 = img[:, width_cutoff:]
    # rotate image to 90 COUNTERCLOCKWISE
    r1 = cv2.rotate(r1, cv2.ROTATE_90_COUNTERCLOCKWISE)
    # cv2.imshow('R1', r1)
    # cv2.imwrite("second_vhorisont_1.jpg", r1)
    r1_path = os.path.join(loc,  'r1.jpg')
    cv2.imwrite(r1_path, r1)
    print("r1 saved at ===", r1_path)
    # rotate image to 90 COUNTERCLOCKWISE
    r2 = cv2.rotate(r2, cv2.ROTATE_90_COUNTERCLOCKWISE)
    # cv2.imshow('R2', r2)
    # cv2.waitKey(0)
    # cv2.imwrite("second_horisont_2.jpg", r2)
    r2_path = os.path.join(loc,  'r2.jpg')
    cv2.imwrite(r2_path, r2)
    print("r2 saved at ===", r2_path)

mnist_path =  my_path + '\MNIST Dataset JPG format\MNIST - JPG - training'

def readAndDivideMNISTImages():
    for i in range(10):
        dir = os.path.join(mnist_path,  str(i))
        for filename in os.listdir(dir):
            f = os.path.join(dir, filename)
            # checking if it is a file
            if os.path.isfile(f):
                img = cv2.imread(f)
                divideimg(str(i), os.path.splitext(filename)[0], img)

readAndDivideMNISTImages()