import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns


# Original
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

num_classes = 10

# One hot encoding for the y train and test data
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

# Normalize data
x_train = x_train /255.0
x_test = x_test / 255.0

# Reshape to one dimensional vector
x_train = x_train.reshape(60000, 28*28)
x_test = x_test.reshape(10000,28*28)

# Create model
model = Sequential()

# 
model.add(Dense(128, input_dim=784, activation='sigmoid'))

model.add(Dropout(0.25))
model.add(Dense(10, activation='sigmoid'))
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
model.summary()


# Train model
epochs = 30
tr_hist = model.fit(x=x_train, y=y_train, epochs=epochs, verbose=1)


# Confirm learning, plot training loss curve
plt.plot(tr_hist.history['loss'], label='Training Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

# Classifcation accuracy numbers for training and test data
train_loss, train_acc = model.evaluate(x_test, y_test)
test_loss, test_acc = model.evaluate(x_test, y_test)
print("Train Loss: {}, Train Accuracy: {}".format(train_loss, train_acc))
print("Test Loss: {}, Test Accuracy: {}".format(test_loss, test_acc))


# Predict values
y_pred = model.predict(x_test)
y_pred_classes= np.argmax(y_pred, axis=1)

# Test on random sample
random_idx = np.random.choice(len(x_test))
x_sample = x_test[random_idx]
y_true = np.argmax(y_test, axis= 1)
y_sample_true = y_true[random_idx]
y_sample_pred_class = y_pred_classes[random_idx]

plt.title("Predicted: {}, True: {}".format(y_sample_pred_class, y_sample_true), fontsize= 16)
plt.imshow(x_sample.reshape(28, 28), cmap='gray')
plt.show()


# Create confusion matrix to see the predictions spread out better
confusion_mtx = confusion_matrix(y_true, y_pred_classes)

fig, ax = plt.subplots(figsize=(15,10))
ax = sns.heatmap(confusion_mtx, annot=True, fmt='d', ax=ax, cmap="Blues")
ax.set_xlabel("Predicted Label")
ax.set_ylabel("True Label")
ax.set_title("Confuson Matirx")
plt.show()


