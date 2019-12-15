import cv2
import color_histogram_feature_extraction
import knn_classifier
import os.path


for dir in sorted(os.listdir('./test_dataset')):
    correct = 0
    fail = 0
    if (dir == ".DS_Store"):
        continue
    for file in sorted(os.listdir("./test_dataset/" + dir)):
        if (file == ".DS_Store"):
            continue
        test_image = cv2.imread('test_dataset/'+dir+"/"+file)

        color_histogram_feature_extraction.color_histogram_of_test_image(test_image)
        prediction = knn_classifier.main('training.data', 'test.data')

        if(prediction == dir):
            correct += 1
        else:
            fail += 1

        cv2.putText(test_image,'Predicted: ' + prediction,(10, 30),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 0, 0) ,2,cv2.LINE_AA)

        cv2.imshow('color classifier', test_image)
        cv2.waitKey(3000)

    print(dir+" Results : "+str(correct)+" Correct"+str(fail)+" Fail")