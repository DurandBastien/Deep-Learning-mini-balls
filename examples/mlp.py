#!/usr/bin/python
# ***************************************************************************
# Author: Christian Wolf
# christian.wolf@insa-lyon.fr
#
# Begin: 20.9.2019
# ***************************************************************************

import glob
import os
import numpy as np
from skimage import io
from numpy import genfromtxt
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class MNISTDataset(Dataset):
    def __init__(self, dir, transform=None):
        self.no_images=0
        self.transform = transform

        arrarr = [None]*10
        for i in range(10):
            print (i)
            regex="%s/%i/*.png"%(dir,i)
            entries=glob.glob(regex)
            arr=[None]*len(entries)
            for j,filename in enumerate(entries):
                # arr[j] = torch.tensor(io.imread(filename))
                arr[j] = io.imread(filename)
                if self.transform:
                    arr[j] = self.transform(arr[j])
            arrarr[i] = arr
            self.no_images = self.no_images + len(entries)

        # Flatten into a single array
        self.images = [None]*self.no_images
        self.labels = [None]*self.no_images
        g_index=0
        for i in range(10):
            for t in arrarr[i]:
                self.images[g_index] = t
                self.labels[g_index] = i
                g_index += 1

    # The access is _NOT_ shuffled. The Dataloader will need
    # to do this.
    def __getitem__(self, index):
        return self.images[index], self.labels[index]

    # Return the dataset size
    def __len__(self):
        return self.no_images
        
BATCHSIZE=50

valid_dataset = MNISTDataset ("MNIST-png/testing", 
    transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))])) # mean, std of dataset
valid_loader = torch.utils.data.DataLoader(valid_dataset,
    batch_size=BATCHSIZE, shuffle=True)

train_dataset = MNISTDataset ("MNIST-png/training", 
    transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))])) # mean, std of dataset)
train_loader = torch.utils.data.DataLoader(train_dataset,
    batch_size=BATCHSIZE, shuffle=True)


class MLP(torch.nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        # input size to 300 units
        self.fc1 = torch.nn.Linear(28*28, 300)
        # 300 units to 10 output classes
        self.fc2 = torch.nn.Linear(300, 10)

    def forward(self, x):        
        # Reshape from a 3D tensor (batchsize, 28, 28)    
    	# to a flattened (batchsize, 28*28)
    	# 1 sample = 1 vector
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# Instantiate the model
model = MLP()

# This criterion combines LogSoftMax and NLLLoss in one single class.
crossentropy = torch.nn.CrossEntropyLoss()

# Set up the optimizer: stochastic gradient descent
# with a learning rate of 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Training
running_loss = 0.0
running_correct = 0
running_count = 0

# Cycle through epochs
for epoch in range(100):
    
    # Cycle through batches
    for batch_idx, (data, labels) in enumerate(train_loader):

        optimizer.zero_grad()
        y = model(data)
        loss = crossentropy(y, labels)
        loss.backward()
        running_loss += loss.item()
        optimizer.step()

        _, predicted = torch.max(y.data, 1)
        running_correct += (predicted == labels).sum().item()
        running_count += BATCHSIZE

		# Print statistics
        if (batch_idx % 100) == 0:
        # if True:
            # print ("predicted=", predicted)
            # print ("labels=", labels)
            # print ("predicted == labels=", predicted == labels)
            # print ("running_correct=", running_correct)
            # print ("running_count=", running_count)
            # print ("train_err=", 100.0*(1.0-running_correct / running_count))
                
            train_err = 100.0*(1.0-running_correct / running_count)
			# valid_err = calcError (net, dataloader_valid)
            print ('Epoch: %d batch: %5d ' % (epoch + 1, batch_idx + 1), end="")
            print ('train-loss: %.3f train-err: %.3f' % (running_loss / 100, train_err))
            running_loss = 0.0
            running_correct = 0.0
            running_count=0.0
        # exit (1)

