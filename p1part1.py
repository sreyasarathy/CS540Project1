## Written by: Sreya Sarathy 
## Attribution: Ruochun Zhang's CS540 P1 Solution 2021  
## Collaborated with Harshet Anand from CS540

import pandas as pd
import numpy as np


# Part 1: Function to load data from a CSV file 
def data_loader(file):
    # Read the CSV file into a pandas DataFrame 
    df = pd.read_csv(file)
    # Normalize the pixel values to the range [0,1]
    x = (df.iloc[:, 1:] / 255.0).to_numpy()
    # Extract the labels 
    y = df.iloc[:, 0].to_numpy()
    return (x, y)


# Load the training data from "mnist_train.csv"
x_train, y_train = data_loader("mnist_train.csv")



# Select only the data points with labels 3 and 8 for testing
test_labels = [3, 8]
indices = np.where(np.isin(y_train, test_labels))[0]
x = x_train[indices]
y = y_train[indices]


# label 3 as 8 and label 0 as 1
y[y == 3] = 0 # 3 has been labelled as 0 
y[y == 8] = 1 # 8 has been labelled as 1 


# Part 2: Set hyperparameters for training 
num_epochs = 200
alpha = 0.01

# Get the number of pixels in an image 
m = x.shape[1]

# Initialize random weights and bias 
w = np.random.rand(m)
b = np.random.rand()

# Part 3: Train the logistic regression model 
loss_previous = 10e10
for epoch in range(num_epochs):
    # Compute the activation using the current weights and bias 
    a = x @ w + b
    a = 1 / (1 + np.exp(-a))
    a = np.clip(a, 0.001, 0.999)

    # Update weights and bias using gradient descent 
    w -= alpha * (x.T) @ (a - y)
    b -= alpha * (a - y).sum()

    # Compute both the loss and accuracy 
    loss = -np.sum(y * np.log(a) + (1 - y) * np.log(1 - a))
    loss_reduction = loss_previous - loss
    loss_previous = loss
    accuracy = sum((a > 0.5).astype(int) == y) / len(y)

    # Print the training process 
    print(
        "epoch = ",
        epoch,
        " loss = {:.7}".format(loss),
        " loss reduction = {:.7}".format(loss_reduction),
        " correctly classified = {:.4%}".format(accuracy),
    )

# Part 4: Prepare to write the results to a file 
file = "result1.txt"
import os
from pathlib import Path

# Remove the file if it already exists 
my_file = Path(file)
if my_file.is_file():
    os.remove(file)


# Q1: Write the feature vector of the first training image to the file. 
# first_tranining image the feature vector 
first = x[0]
f = open(file, "a")
f.write("##1: \n")
for i, value in enumerate(first):
    if i != 0:
        f.write(", ")
    f.write("%.2f" % value)
f.write("\n")
f.close()

# Q2: Write the weights and bias to the file 
f = open(file, "a")
f.write("##2: \n")
for i, value in enumerate(w):
    if i != 0:
        f.write(",")
    f.write("%.4f" % value)
f.write(", ")

# Code lines specific to bias 
f.write("%.4f" % b)
f.write("\n")
f.close()

# Q3 is a little tedious and requires the following steps:  

# 1. We load the file 
x = np.loadtxt("test.txt", delimiter=",")
x = x / 255.0

# 2. We calculate the activation 
a = 1 / (1 + np.exp(-(x @ w + b))) # formula for activation 
f = open(file, "a")
f.write("##3: \n")

# for loop enumreates through 'a'
for i, value in enumerate(a):
    if i != 0:
        f.write(", ")
    f.write("%.2f" % value)
f.write("\n")
f.close()


# Q4 goes hand in hand with the others so it is computed as follows: 
f = open(file, "a")
f.write("##4: \n")

# The following for loop nested with an if statement enumerates 
# over the activations which were computed in Q3
for i, value in enumerate(a):
    if value >= 0.5:
        value = 1
    else:
        value = 0
    if i != 0:
        f.write(", ")
    f.write(str(value))
f.write("\n")
f.close()