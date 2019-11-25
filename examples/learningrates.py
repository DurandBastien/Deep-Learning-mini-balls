#!/usr/bin/python
# ***************************************************************************
# Author: Christian Wolf
# christian.wolf@insa-lyon.fr
#
# Begin: 22.9.2019
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
from torch.utils.tensorboard import SummaryWriter

STATS_INTERVAL = 200

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
    def __init__(self, no_hidden):
        super(MLP, self).__init__()
        # input size to no of hidden units
        self.fc1 = torch.nn.Linear(28*28, no_hidden)
        # 300 units to 10 output classes
        self.fc2 = torch.nn.Linear(no_hidden, 10)

    def forward(self, x):        
        # Reshape from a 3D tensor (batchsize, 28, 28)    
    	# to a flattened (batchsize, 28*28)
    	# 1 sample = 1 vector
        x = x.view(-1, 28*28)
        x = F.tanh(self.fc1(x))
        return self.fc2(x)

# Instantiate the model
model = MLP(2000)

# This criterion combines LogSoftMax and NLLLoss in one single class.
crossentropy = torch.nn.CrossEntropyLoss(reduce='mean')

# Set up the optimizer: stochastic gradient descent
# with a learning rate of 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=1)

# Setting up tensorboard
writer = SummaryWriter('runs_lrates/mnist_lr=1')

# ************************************************************************
# Calculate the error of a model on data from a given loader
# This is used to calculate the validation error every couple of
# thousand batches
# ************************************************************************

def calcError (net, dataloader):
    vloss=0
    vcorrect=0
    vcount=0
    for batch_idx, (data, labels) in enumerate(dataloader):
        y = model(data)
        loss = crossentropy(y, labels)
        vloss += loss.item()
        _, predicted = torch.max(y.data, 1)
        vcorrect += (predicted == labels).sum().item()
        vcount += BATCHSIZE
    return vloss/len(dataloader), 100.0*(1.0-vcorrect/vcount)

# Training
running_loss = 0.0
running_correct = 0
running_count = 0

# Add the graph to tensorboard
dataiter = iter(train_loader)
data, labels = dataiter.next()
writer.add_graph (model, data)
writer.flush()

# Cycle through epochs
for epoch in range(38):
    
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
        if (batch_idx % STATS_INTERVAL) == 0:
            train_err = 100.0*(1.0-running_correct / running_count)
            valid_loss, valid_err = calcError (model, valid_loader)
            print ('Epoch: %d batch: %5d ' % (epoch + 1, batch_idx + 1), end="")
            print ('train-loss: %.3f train-err: %.3f' % (running_loss / STATS_INTERVAL, train_err), end="")
            print (' valid-loss: %.3f valid-err: %.3f' % (valid_loss, valid_err))

            # Write statistics to the log file
            writer.add_scalars ('Loss', {
                'training:': running_loss / STATS_INTERVAL,
                'validation:': valid_loss }, 
                epoch * len(train_loader) + batch_idx)

            writer.add_scalars ('Error', {
                'training:': train_err,
                'validation:': valid_err }, 
                epoch * len(train_loader) + batch_idx)
                            
            running_loss = 0.0
            running_correct = 0.0
            running_count=0.0

