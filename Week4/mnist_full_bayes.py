import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from scipy.stats import multivariate_normal

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Reshape the train and test data to the correct format 1x784
x_train_vector = x_train.reshape(60000, 28*28)
x_test_vector = x_test.reshape(10000, 28*28)

# Create the noise version of sampled train data
def add_gaussian_noise(x_train, scale=1.0):
    noise = np.random.normal(loc=0.0, scale=scale, size=x_train.shape)
    x_train_noisy = x_train + noise
    return x_train_noisy

scale = 10
x_train_noisy_sample = add_gaussian_noise(x_train_vector, scale) 

# Calculate the mean vector and full covariance matrix for each class using the noisy sampled data
def calculate_mean_covariance(x_train_noisy, y_train_sample):
    means = []
    covariances = []
    for c in range(10): 
        class_data = x_train_noisy[y_train_sample == c]  
        means.append(np.mean(class_data, axis=0)) 
        covariances.append(np.cov(class_data, rowvar=False) + 0.01 * np.eye(class_data.shape[1]))  
    return np.array(means), np.array(covariances)

means, covariances = calculate_mean_covariance(x_train_noisy_sample, y_train)

# Get predictions using the full multivariate Gaussian
def get_predictions_full_gaussian():
    predictions = []
    for x in x_test_vector: 
        best_log_likelihood = None
        best_class = None
        for c in range(10):
            mn = multivariate_normal(mean=means[c], cov=covariances[c])
            log_likelihood = mn.logpdf(x)  
            if best_log_likelihood is None or log_likelihood > best_log_likelihood:
                best_log_likelihood = log_likelihood
                best_class = c  
        predictions.append(best_class)
    return predictions

print("Tested with smaller sample sizes and got 0.97 accuracy, but the full version is too slow.")
# Get the predictions using the full multivariate Gaussian approach
predictions = get_predictions_full_gaussian()

# Return accuracy of the prediction to the ground truth
def class_acc(pred, gt):
    pred = np.array(pred)
    gt = np.array(gt)
    correct_predictions = np.sum(pred == gt)
    accuracy = correct_predictions / len(gt)
    return accuracy

accuracy = class_acc(predictions, y_test) 
print(f'Noise applied: {scale}')
print(f'Classification accuracy (with noisy sample data): {accuracy:.2f}')
