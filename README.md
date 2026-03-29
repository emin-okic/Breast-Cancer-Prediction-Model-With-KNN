# Breast Cancer Prediction with Weighted KNN

This project implements a simple machine learning model to predict whether a tumor is malignant or benign using the Breast Cancer Wisconsin Diagnostic dataset from scikit-learn. Each data point represents a tumor described by 30 numerical features such as radius, texture, and smoothness. The goal of the model is to take a new patient’s tumor data and classify it as either benign (0) or malignant (1).

The approach used in this project is the weighted k-nearest neighbors (KNN) algorithm. Unlike standard KNN, which treats all neighbors equally, weighted KNN assigns more importance to closer data points and less importance to those farther away. This is particularly useful in healthcare scenarios where small differences in tumor characteristics can significantly impact diagnosis. By giving more influence to similar tumors, the model produces more intuitive and often more accurate predictions.

The algorithm works by first splitting the dataset into training and testing sets. For each test point, the model computes the distance to every point in the training set using Euclidean distance. These distances are sorted, and the top k closest neighbors are selected. Instead of using a simple majority vote, each neighbor contributes a weighted vote based on the inverse of its distance, meaning closer neighbors have more influence. The class with the highest total weight is then returned as the prediction.

This implementation is written from scratch using only basic Python libraries such as math and random, along with scikit-learn for loading the dataset. The goal is to provide a clear and intuitive understanding of how KNN works under the hood, rather than relying on high-level machine learning libraries.

To run the project, create a Python environment with scikit-learn installed, then execute the main script. The program will train the model, evaluate it on a test set, and output the prediction accuracy. This project is a great starting point for understanding distance-based algorithms and how simple models can be applied to real-world problems like medical diagnosis.

# References
Please refer here for the dataset: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html#sklearn.datasets.load_breast_cancer

For a deeper dive, refer to the original data set source: https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic

For setting up a sklearn env, refer here: For creating an sklearn virtual environment refer here: https://scikit-learn.org/stable/install.html