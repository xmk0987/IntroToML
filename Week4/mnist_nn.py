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

# Return accuracy of the prediction to the ground truth
def class_acc(pred, gt):
    pred = np.array(pred)
    gt = np.array(gt)
    
    correct_predictions = np.sum(pred == gt)
    
    accuracy = correct_predictions / len(gt)
    
    return accuracy


print(f'x_train shape {x_train.shape}')
print(f'y_train shape {y_train.shape}')
print(f'x_test shape {x_test.shape}')
print(f'y_test shape {y_test.shape}')

x_train_vector = x_train.reshape(60000, 28*28)
x_test_vector = x_test.reshape(10000,28*28)
print(f'x_train_vector shape {x_train_vector.shape}')
print(f'x_test_vector shape {x_test_vector.shape}')

knn = KNeighborsClassifier()
knn.fit(x_train_vector, y_train)

predictions = knn.predict(x_test_vector)
accuracy = class_acc(predictions, y_test)

print(f'Classification accuracy is {accuracy:.2f}')
