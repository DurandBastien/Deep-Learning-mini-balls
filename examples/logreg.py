#!/usr/bin/python
# ***************************************************************************
# Author: Christian Wolf
# christian.wolf@insa-lyon.fr
#
# Begin: 18.9.2019
# ***************************************************************************

import numpy as np
from numpy import genfromtxt
import torch
from torch.nn import functional as F

# Import the text file into a numpy array
n = genfromtxt('breast-cancer-wisconsin-cleaned.csv', delimiter=',')

# Convert to torch tensor
D = torch.tensor(n, dtype=torch.float32)
N_samples = D.size(0)

# The input is the full matrix without first and last column
# Plus the 1 column for the bias
X=D[:,1:-1]
X = torch.cat ((X, torch.ones((X.size(0),1))),1)
print (X.size())

# The targets
T=D[:,-1:]

# Change all 2->0 and 4->1
T[T==2]=0
T[T==4]=1
print (T.size())

class LogisticRegression(torch.nn.Module):
    def __init__(self):
        super(LogisticRegression, self).__init__()
        
        # The linear layer (input dim, output dim)
        # It also contains a weight matrix 
        # (here single output-> vector)
        self.fc1 = torch.nn.Linear(10, 1)

    # The forward pass of the network. x is the input
    def forward(self, x):        
        return F.sigmoid(self.fc1(x))

# Instantiate the model
model = LogisticRegression()

# The loss function: binary cross-entropy
criterion = torch.nn.BCELoss()

# Set up the optimizer: stochastic gradient descent
# with a learning rate of 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)




def calcAccuracy():    
    model.eval()
    correct = 0.0
    for sample in range(N_samples):
        y = 1 if model(X[sample,:]) > 0.5 else 0
        correct += (y == T[sample]).numpy()
        
    print ("Accuracy = ", 100.0*correct/N_samples)


# No batches yet !
# 1 epoch = 1 pass over the full dataset 
for epoch in range(200):

    print ("Starting epoch", epoch, " ",end='')
    calcAccuracy()

    # Go over the samples of the dataset
    for sample in range(N_samples):

        # Put the model into training mode and clear the gradients
        model.train()
        optimizer.zero_grad()

        # Forward pass (stimulate model with inputs)
        y = model(X[sample,:])
    
        # Compute Loss
        loss = criterion(y, T[sample])
        
        # Backward pass: calculate the gradients
        loss.backward()

        # Perform one step of stochastic gradient descent
        optimizer.step()

