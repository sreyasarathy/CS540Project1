## Written by: Sreya Sarathy 
## Attribution: Ruochun Zhang's CS540 P1 Solution 2021 
## Collaboration with Harshet Anand from CS 540

import numpy as np
import os
from pathlib import Path

def data_loader(file):
    # Step 1: Load and preprocess data from a file
    a = np.genfromtxt(file, delimiter=",", skip_header=0)
    x = a[:, 1:] / 255.0  # Normalize pixel values
    y = a[:, 0]  # Extract labels
    return (x, y)

# Load the training data from mnist_train.csv
x_train, y_train = data_loader("mnist_train.csv")

# Define test labels
test_labels = [8, 0]  # The desired test labels are 8 and 0

# Get indices of samples with test labels
indices = np.where(np.isin(y_train, test_labels))[0]

# Select samples with test labels from training data
x = x_train[indices]
y = y_train[indices]

# Map test labels to binary values (0 or 1)
y[y == test_labels[0]] = 0
y[y == test_labels[1]] = 1

def sigmoid(x):
    # Compute the sigmoid activation function
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(o):
    # Compute the derivative of the sigmoid function
    return o * (1 - o)

h = 28
m = x.shape[1]
alpha = 0.01
num_epochs = 70
num_train = len(y)

def nnet(train_x, train_y, alpha, num_epochs, num_train):
    # Initialize weights and biases randomly
    w1 = np.random.uniform(low=-1, high=1, size=(m, h))
    w2 = np.random.uniform(low=-1, high=1, size=(h, 1))
    b1 = np.random.uniform(low=-1, high=1, size=(h, 1))
    b2 = np.random.uniform(low=-1, high=1, size=(1, 1))

    loss_previous = 10e10

    for epoch in range(1, num_epochs + 1):
        train_index = np.arange(num_train)
        np.random.shuffle(train_index)

        for i in train_index:
            # Perform forward propagation
            a1 = sigmoid(w1.T @ train_x[i, :].reshape(-1, 1) + b1)
            a2 = sigmoid(w2.T @ a1 + b2)

            # Backpropagation
            dCdw1 = ((a2 - train_y[i]) * sigmoid_derivative(a2) * w2 * sigmoid_derivative(a1) * (train_x[i, :].reshape(1, -1)))
            dCdb1 = ((a2 - train_y[i]) * sigmoid_derivative(a2) * w2 * sigmoid_derivative(a1))
            dCdw2 = (a2 - train_y[i]) * sigmoid_derivative(a2) * a1
            dCdb2 = (a2 - train_y[i]) * sigmoid_derivative(a2)

            # Update weights and biases using gradient descent
            w1 = w1 - alpha * dCdw1.T
            b1 = b1 - alpha * dCdb1
            w2 = w2 - alpha * dCdw2
            b2 = b2 - alpha * dCdb2

        # Compute loss and accuracy after each epoch
        out_h = sigmoid(train_x @ w1 + b1.T)
        out_o = sigmoid(out_h @ w2 + b2)
        loss = 0.5 * np.sum(np.square(train_y.reshape(-1, 1) - out_o))
        loss_reduction = loss_previous - loss
        loss_previous = loss
        correct = sum((out_o > 0.5).astype(int) == train_y.reshape(-1, 1))
        accuracy = (correct / num_train)[0]
        print("epoch =", epoch, "loss = {:.7}".format(loss), "loss reduction = {:.7}".format(loss_reduction), "correctly classified = {:.4%}".format(accuracy))

    return w1, b1, w2, b2

# Train the neural network
w1, b1, w2, b2 = nnet(x, y, alpha, num_epochs, num_train)

file = 'result3.txt'
my_file = Path(file)

# Remove the file if it already exists
if my_file.is_file():
    os.remove(file)

w1plusb1 = np.concatenate((w1, b1.T), axis=0)

# Write w1plusb1 to the file
f = open(file, 'a')
f.write('##5: \n')
for row in range(w1plusb1.shape[0]):
    for i in range(h):
        if i != 0:
            f.write(', ')
        f.write('%.4f' % w1plusb1[row, i])
    f.write('\n')
f.close()

w2plusb2 = np.concatenate((w2, b2), axis=0)

# Write w2plusb2 to the file
f = open(file, 'a')
f.write('##6: \n')
for i in range(w2plusb2.shape[0]):
    if i != 0:
        f.write(', ')
    f.write('%.4f' % w2plusb2[i, 0])
f.write('\n')
f.close()

x = np.loadtxt('test.txt', delimiter=',')

# Normalize the test data
x = x / 255.0

# Perform forward propagation to predict the test data
a = sigmoid(x @ w1 + b1.T)
a = sigmoid(a @ w2 + b2)

# Write 'a' to the file
f = open(file, 'a')
f.write('##7: \n')
for i in range(a.shape[0]):
    if i != 0:
        f.write(', ')
    f.write('%.2f' % a[i, 0])
f.write('\n')
f.close()

# Write the predicted labels to the file
f = open(file, 'a')
f.write('##8: \n')
for i in range(a.shape[0]):
    if i != 0:
        f.write(', ')
    predicted_value = (a[i, 0] > .5).astype(int)
    f.write('%.0f' % predicted_value)
f.write('\n')
f.close()

misclassified_image = None

# Find the index of the first misclassified image after the 100th image
for i in range(a.shape[0]):
    predicted_value = (a[i, 0] > .5).astype(int)
    if i >= 100 and predicted_value == 0:
        misclassified_image = i
        break

# Write the misclassified image to the file
f = open(file, 'a')
f.write('##9: \n')
for i in range(x[misclassified_image, :].reshape(-1, 1).shape[0]):
    if i != 0:
        f.write(', ')
    f.write('%.2f' % x[misclassified_image, :].reshape(-1, 1)[i, 0])
f.write('\n')
f.close()

# Write the ground truth label for the misclassified image to the file
f = open(file, 'a')
f.write('##10: \n')
f.write('True')
f.write('\n')
f.close()

# Write the predicted label for the misclassified image to the file
f = open(file, 'a')
f.write('##11: \n')
f.write('None')
f.write('\n')
f.close()
