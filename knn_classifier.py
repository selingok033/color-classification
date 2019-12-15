import csv
import math
import operator

def calculateEuclideanDistance(variable1, variable2, length):
    distance = 0
    for x in range(length):
        distance += pow(variable1[x] - variable2[x], 2)
    return math.sqrt(distance)

def kNearestNeighbors(training_feature_vector, testInstance):
    distances = []
    length = len(testInstance)
    for x in range(len(training_feature_vector)):
        dist = calculateEuclideanDistance(testInstance,
                training_feature_vector[x], length)
        distances.append((training_feature_vector[x], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(0,3):
        neighbors.append(distances[x][0])
    return neighbors

def responseOfNeighbors(neighbors):
    n1 = neighbors[0][3]
    n2 = neighbors[1][3]
    n3 = neighbors[2][3]

    if(n1==n2) | (n1==n3):
        label = n1
    elif(n2==n3):
        label = n2
    else:
        label = n3
    return label

# Load image feature data to training feature vectors and test feature vector
def loadDataset(
    filename,
    filename2,
    training_feature_vector=[],
    test_feature_vector=[],
    ):
    with open(filename) as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        for x in range(len(dataset)):
            for y in range(3):
                dataset[x][y] = float(dataset[x][y])
            training_feature_vector.append(dataset[x])

    with open(filename2) as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        for x in range(len(dataset)):
            for y in range(3):
                dataset[x][y] = float(dataset[x][y])
            test_feature_vector.append(dataset[x])

def main(training_data, test_data):
    training_feature_vector = []  # training feature vector
    test_feature_vector = []  # test feature vector
    loadDataset(training_data, test_data, training_feature_vector, test_feature_vector)
    classifier_prediction = []  # predictions
    k = 3
    for x in range(len(test_feature_vector)):
        neighbors = kNearestNeighbors(training_feature_vector, test_feature_vector[x])
        result = responseOfNeighbors(neighbors)
        classifier_prediction.append(result)
        print(classifier_prediction[0])
    return classifier_prediction[0]

if __name__ == '__main__':
    main("./training.data","./test.data")