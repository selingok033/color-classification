import os
import cv2
import numpy as np

def color_histogram_of_test_image(test_src_image):
        chans = cv2.split(test_src_image)
        colors = ('b', 'g', 'r')
        features = []
        feature_data = ''
        counter = 0
        for (chan, color) in zip(chans, colors):
            counter = counter + 1

            hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
            features.extend(hist)
            max = np.argmax(hist)

            if counter == 1:
                blue = max
            elif counter == 2:
                green = max
            elif counter == 3:
                red = max
                feature_data = str(red) + ',' + str(green) + ',' + str(blue)
                print(feature_data)

            with open('test.data', 'w') as myfile:
                myfile.write(feature_data)


def color_histogram_of_training_image(img_name,label):

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
        max = np.argmax(hist)

        if counter == 1:
            blue = str(max)
        elif counter == 2:
            green = str(max)
        elif counter == 3:
            red = str(max)
            feature_data = red + ',' + green + ',' + blue
            if (red == "255") & (green == "255") & (blue == "255"):
                print(img_name)

    with open('training.data', 'a') as myfile:
        myfile.write(feature_data + ',' + label + '\n')

def training():
    for dir in sorted(os.listdir('./Dataset')):
        for file in sorted(os.listdir("./Dataset/"+dir)):
            if(file == ".DS_Store"):
                continue
            color_histogram_of_training_image("./Dataset/"+dir+"/"+file,dir)

if __name__ == '__main__':
    training()