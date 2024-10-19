import matplotlib.pyplot as plt
import tensorflow as tf
from random import random
import numpy as np


# Original
mnist = tf.keras.datasets.mnist
# New
#mnist = tf.keras.datasets.fashion_mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Print the size of training and test data
""" print(f'x_train shape {x_train.shape}')
print(f'y_train shape {y_train.shape}')
print(f'x_test shape {x_test.shape}')
print(f'y_test shape {y_test.shape}')

for i in range(x_test.shape[0]):
    # Show some images randomly
    if random() > 0.999:
        plt.figure(1)
        plt.clf()
        plt.imshow(x_test[i], cmap='gray_r')
        plt.title(f"Image {i} label num {y_test[i]} predicted {0}")
        plt.pause(1)

for i in range(10):
    plt.figure()
    plt.imshow(x_test[i], cmap='gray_r')
    plt.title(f"Image {i} - Actual: {y_test[i]}, Predicted: {predictions[i]}")
    plt.axis('off')  # Hide axes
    plt.show()
 """
 
print(f"Label of the first image: {y_train[0]}")


plt.figure(figsize=(4, 4)) 
plt.imshow(x_train[0], cmap='gray_r')  
plt.title(f"Label: {y_train[0]}")  
plt.axis('off')  
plt.show() 

def class_acc(pred, gt):
    pred = np.array(pred)
    gt = np.array(gt)
    
    correct_predictions = np.sum(pred == gt)
    
    accuracy = correct_predictions / len(gt)
    
    return accuracy


pred = np.random.randint(0, 10, size=y_test.shape) 
accuracy = class_acc(pred, y_test)

print(f"Random accuracy: {accuracy:.2%}") 