"""
Breast Cancer Prediction Problem:

Given a patient's tumor features, predict whether the person has a malignant or benign tumor.

0 implies the tumor is benign and non cancerous.

1 implies the tumor is malignant and cancerous.

Please refer here for the dataset: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html#sklearn.datasets.load_breast_cancer

For a deeper dive, refer to the original data set source: https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic

For setting up a sklearn env, refer here: For creating an sklearn virtual environment refer here: https://scikit-learn.org/stable/install.html
"""

"""
The Algorithm:

The approach used here is the weighted knn algorithm. 

The power of the weighted knn algorithm is that closer related tumors get more weight than non-similar tumors. 

This is powerful because a similar tumor should get more voting power than another tumor since cancer can vary so much in features.

1. The first step is to initialize an empty distances list to store the neighboring data points distance.
2. The second step is to compute the distance from each point in the training dataset to the test point.
3. The third step is to sort the distances array in ascending order so the closest neighboring data points are first.
4. The fourth step is to select the k nearest neighbors from the distances array.
5. The fifth step is to compute the weighted votes of each of the nearest neighbors. That means for each neighbor, compute the weight, and add it to that class.
6. The sixth step is to make the prediction and choose the class / set of neighbors with the highest weight.
7. The final step is to return the prediction.

Here's how it works intuitively:

1. How will I store distances for each data point?
2. How similar is everything to my test point?
3. Who are the closest neighbors?
4. Focus on only the top k / closest neighbors
5. Closer neighbors get more influence
6. Pick the strongest signal

"""

# The first step is to use sklearn to import the breast cancer dataset
from sklearn.datasets import load_breast_cancer

# The second / third step is to import the math and random library because I'm retarded
import math
import random

# Before I do anything, I load in the data set
data = load_breast_cancer()

# Now I define x to be the 30 features of the data set
X = data.data

# I define y to be the labels of each data point, ie whether the tumor is cancerous or not
y = data.target

# Now its time to define the cross validation function for training and testing the model
def train_test_split(X, y, test_size = 0.2):

    # First I turn the data set into a list
    data = list(zip(X, y))

    # Then I shuffle the data
    random.shuffle(data)

    # Then I split the dataset
    split = int(len(data) * (1 - test_size))

    # Now I define the training data set which will train the model
    train = data[:split]

    # Then I define the testing dataset to test the accuracy of the model
    test = data[split:]

    # Now its time to define the training and testing variables
    X_train, y_train = zip(*train)

    X_test, y_test = zip(*test)

    # Then I return the train / test data sets for the model to use
    return list(X_train), list(X_test), list(y_train), list(y_test)

# Now I define a function to compute the distances from each data point in my training set to the test set
def euclidian_distance(p1, p2):

    total = 0

    for i in range(len(p1)):

        total += (p1[i] - p2[i]) ** 2

    return math.sqrt(total)

# Now its time to define the model to predict cancer
def weighted_knn(X_train, y_train, test_point, k = 5):

    # First I define the array to store the distances
    distances = []

    # Then I compute the distances in the training set from the test point
    for i in range(len(X_train)):

        dist = euclidian_distance(test_point, X_train[i])

        distances.append( (dist, y_train[i]) )
    
    # Now its time to sort the array
    distances.sort(key = lambda x: x[0])

    # Then I take the k nearest neighbors which is 5 neighbors by default
    neighbors = distances[:k]

    # Then I sum up the votes using weights
    weight_sum = {0: 0, 1: 0}

    # Then I compute the weight of each class for each neighbor
    for dist, label in neighbors:

        if dist == 0:

            weight = 1
        
        else:

            weight = 1 / dist
        
        weight_sum[label] += weight

    # Finally I return the prediction
    return 0 if weight_sum[0] > weight_sum[1] else 1

# Main Program for testing
def main():

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    correct = 0

    for i in range(len(X_test)):

        prediction = weighted_knn(X_train, y_train, X_test[i], k = 5)

        if prediction == y_test[i]:

            correct += 1
    
    accuracy = correct / len(X_test)

    print(f"Model Accuracy: {accuracy: .2f}")

# Now its time to run it
if __name__ == "__main__":

    main()