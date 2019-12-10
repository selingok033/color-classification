from PIL import Image
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import itemfreq
import knn_classifier


def color_histogram_of_test_image(test_src_image):

    for f in sorted(os.listdir("test_dataset")):
        # load the image
        image = cv2.imread(test_src_image+f)

        chans = cv2.split(image)
        colors = ('b', 'g', 'r')
        features = []
        feature_data = ''
        counter = 0
        for (chan, color) in zip(chans, colors):
            counter = counter + 1

            hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
            features.extend(hist)

            # find the peak pixel values for R, G, and B
            elem = np.argmax(hist)

            if counter == 1:
                blue = str(elem)
            elif counter == 2:
                green = str(elem)
            elif counter == 3:
                red = str(elem)
                feature_data = red + ',' + green + ',' + blue
                print(feature_data)

            with open('test.data', 'a') as myfile:
                myfile.write(feature_data)
        with open('test.data', 'a') as myfile:
            myfile.write('\n')

def color_histogram_of_training_image(img_name):

    # detect image color by using image file name to label training data
    if 'Red' in img_name:
        data_source = 'Red'
    elif 'Yellow' in img_name:
        data_source = 'Yellow'
    elif 'Green' in img_name:
        data_source = 'Green'
    elif 'Orange' in img_name:
        data_source = 'Orange'
    elif 'White' in img_name:
        data_source = 'White'
    elif 'Black' in img_name:
        data_source = 'Black'
    elif 'Blue' in img_name:
        data_source = 'Blue'
    elif 'Violet' in img_name:
        data_source = 'Violet'

    # load the image
    image = cv2.imread(img_name)

    chans = cv2.split(image)
    colors = ('b', 'g', 'r')
    features = []
    feature_data = ''
    counter = 0

    for (chan, color) in zip(chans, colors):
        counter = counter + 1

        hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
        features.extend(hist)

        # find the peak pixel values for R, G, and B
        elem = np.argmax(hist)

        if counter == 1:
            blue = str(elem)
        elif counter == 2:
            green = str(elem)
        elif counter == 3:
            red = str(elem)
            feature_data = red + ',' + green + ',' + blue

    with open('training.data', 'a') as myfile:
        myfile.write(feature_data + ',' + data_source + '\n')


def training():

    # red color training images
    for f in sorted(os.listdir('./Dataset/Red')):
        color_histogram_of_training_image('./Dataset/Red/' + f)

    # yellow color training images
    for f in sorted(os.listdir('./Dataset/Yellow')):
        color_histogram_of_training_image('./Dataset/Yellow/' + f)

    # green color training images
    for f in sorted(os.listdir('./Dataset/Green')):
        color_histogram_of_training_image('./Dataset/Green/' + f)

    # orange color training images
    for f in sorted(os.listdir('./Dataset/Orange')):
        color_histogram_of_training_image('./Dataset/Orange/' + f)

    # white color training images
    for f in sorted(os.listdir('./Dataset/White')):
        color_histogram_of_training_image('./Dataset/White/' + f)

    # black color training images
    for f in sorted(os.listdir('./Dataset/Black')):
        color_histogram_of_training_image('./Dataset/Black/' + f)

    # blue color training images
    for f in sorted(os.listdir('./Dataset/Blue')):
        color_histogram_of_training_image('./Dataset/Blue/' + f)

if __name__ == '__main__':
    training()
    color_histogram_of_test_image("test_dataset/")