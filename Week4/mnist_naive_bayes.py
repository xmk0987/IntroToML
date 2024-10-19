import matplotlib.pyplot as plt
import tensorflow as tf
from random import random
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# Original
mnist = tf.keras.datasets.mnist
# New
#mnist = tf.keras.datasets.fashion_mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Reshape the train and test data to the correct format 1x784
x_train_vector = x_train.reshape(60000, 28*28)
x_test_vector = x_test.reshape(10000,28*28)
print(f'x_train_vector shape {x_train_vector.shape}')
print(f'x_test_vector shape {x_test_vector.shape}')


# Create the noise version of train data
def add_gaussian_noise(x_train, scale=1.0):
    noise = np.random.normal(loc=0.0, scale=scale, size=x_train.shape)
    x_train_noisy = x_train + noise
    return x_train_noisy

scale = 0.1
x_train_noisy = add_gaussian_noise(x_train_vector, scale)

# Calculate the mean and variance for each class
def calculate_mean_variance(x_train_noisy, y_train):
    means = []
    variances = []
    for c in range(10):
        class_data = x_train_noisy[y_train == c]
        means.append(np.mean(class_data, axis=0))
        variances.append(np.var(class_data, axis=0) + 0.01)
    return np.array(means), np.array(variances)

means, variances = calculate_mean_variance(x_train_noisy, y_train)

def calculate_log_likelihood(var, mean, x):
    return -0.5 * np.sum(np.log(2*np.pi) + np.log(var) + ((x - mean) ** 2) / var)
   
# Get the best prediction based on the log likelihood for each class for each test value 
def get_predictions():
    predictions = []
    for x in x_test_vector:
        best_prediction = None
        best_class = None
        for c in range(10):
            log_likelihood = calculate_log_likelihood(variances[c], means[c], x)
            if (best_prediction == None or log_likelihood > best_prediction):
                best_prediction = log_likelihood
                best_class = c 
        predictions.append(best_class)
    return predictions

predictions = get_predictions()

# Return accuracy of the prediction to the ground truth
def class_acc(pred, gt):
    pred = np.array(pred)
    gt = np.array(gt)
    
    correct_predictions = np.sum(pred == gt)
    
    accuracy = correct_predictions / len(gt)
    
    return accuracy


accuracy = class_acc(predictions, y_test)
print(f'Noise applied {scale}')
print(f'Classification accuracy is {accuracy:.2f}')
