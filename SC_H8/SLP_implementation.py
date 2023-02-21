import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#
# Perceptron implementation
#
class CustomPerceptron(object):

    def __init__(self, n_iterations=100, random_state=1, learning_rate=0.01):
        self.n_iterations = n_iterations
        self.random_state = random_state
        self.learning_rate = learning_rate
    '''
    Stochastic Gradient Descent

    1. Weights are updated based on each training examples.
    2. Learning of weights can continue for multiple iterations
    3. Learning rate needs to be defined
    '''

    def fit(self, X, y,check,Thresh):
        if check == True :
            rgen = np.random.RandomState(self.random_state)
            self.coef_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
            self.errors_ = []
            Xi = np.array(X)
            yi = np.array(y)
            for _ in range(self.n_iterations):
                errors = 0
                for i in range(Xi.shape[0]):
                    predicted_value = self.predict(Xi[i])
                    self.coef_[1:] = self.coef_[1:] + self.learning_rate * (
                                yi[i] - predicted_value) * Xi[i]  # w1 = w0 + l( y-h(x)) * Xi
                    self.coef_[0] = self.coef_[0] + self.learning_rate * (
                                yi[i] - predicted_value) * 1  # b1 = b0 + l( y-h(x))
                    update = self.learning_rate * (yi[i] - predicted_value)
                    errors += ((yi[i] - predicted_value)**2)
                self.errors_.append((1/(2*Xi.shape[0])) * errors)
                if self.errors_[-1] < Thresh:
                    break

        else :
            rgen = np.random.RandomState(self.random_state)
            self.coef_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
            self.errors_ = []
            Xi = np.array(X)
            yi = np.array(y)
            for _ in range(self.n_iterations):
                errors = 0
                for i in range(Xi.shape[0]):
                    predicted_value = self.predict(Xi[i])
                    self.coef_[1:] = self.coef_[1:] + self.learning_rate * (
                            yi[i] - predicted_value) * Xi[i]  # w1 = w0 + l( y-h(x)) * Xi
                    self.coef_[0] = 0 * (self.coef_[0] + self.learning_rate * (
                            yi[i] - predicted_value) * 1 ) # b1 = b0 + l( y-h(x))
                    update = self.learning_rate * (yi[i] - predicted_value)
                    errors += int(update != 0.0)
                self.errors_.append(errors)

    '''
    Net Input is sum of weighted input signals
    '''

    def net_input(self, X):
        weighted_sum = np.dot(X, self.coef_[1:]) + self.coef_[0]
        return weighted_sum

    '''
    Activation function is fed the net input and the unit step function
    is executed to determine the output.
    '''

    def activation_function(self, X):
        weighted_sum = self.net_input(X)
        return weighted_sum

    '''
    Prediction is made on the basis of output of activation function
    '''

    def predict(self, X):
        return np.where(self.activation_function(X) >=0.0, 1, -1)

    '''
    Model score is calculated based on comparison of
    expected value and predicted value
    '''

    def score(self, X, y):
        Xi = np.array(X)
        yi = np.array(y)
        misclassified_data_count = 0
        for i in range(Xi.shape[0]):
            output = self.predict(Xi[i])
            if (yi[i] != output):
                misclassified_data_count += 1
        total_data_count = len(X)
        self.score_ = (total_data_count - misclassified_data_count) / total_data_count
        return self.score_

    def confusion_matrix(self, X, y):  # pass predicted and original labels to this function
        Xi = np.array(X)
        yi = np.array(y)
        misclassified_data_count = 0
        matrix = np.zeros((2, 2))  # form an empty matric of 2x2
        for i in range(Xi.shape[0]):  # the confusion matrix is for 2 classes: 1,0
            # 1=positive, 0=negative
            pred = self.predict(Xi[i])
            if int(pred) == 1 and int(yi[i]) == 1:
                matrix[0, 0] += 1  # True Positives
            elif int(pred) == -1 and int(yi[i]) == 1:
                matrix[0, 1] += 1  # False Positives
            elif int(pred) == 1 and int(yi[i]) == -1:
                matrix[1, 0] += 1  # False Negatives
            elif int(pred) == -1 and int(yi[i]) == -1:
                matrix[1, 1] += 1  # True Negatives
        precision = matrix[0, 0] / (matrix[0, 0] + matrix[0, 1])
        print("Precision:", precision)
        recall = matrix[0, 0] / (matrix[0, 0] + matrix[1, 0])
        print("Recall:", recall)
        specificity = matrix[1, 1] / (matrix[0, 1] + matrix[1, 1])
        print("Specificity:", specificity)
        negative_pred_value = matrix[1, 1] / (matrix[1, 0] + matrix[1, 1])
        print("Negative Predicted Value:", negative_pred_value)
        f1 = 2 * (precision * recall) / (precision + recall)
        print("F1 score:", f1)

        # the above code adds up the frequencies of the tps,tns,fps,fns and a matrix is formed
        return matrix

