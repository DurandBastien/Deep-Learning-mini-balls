import torch
import numpy as np
from torch.nn import functional as F
import tp_src.dataset_det as d_loader
import torch.utils.data as torch_d

STATS_INTERVAL = 50

class LeNet(torch.nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        # self.conv1 = torch.nn.Conv2d(1, 20, 5, 1)
        # self.conv2 = torch.nn.Conv2d(20, 50, 5, 1)
        self.fc1 = torch.nn.Linear(3*100*100, 500)
        self.fc2 = torch.nn.Linear(500, 9)

    def forward(self, x):
        # x = F.relu(self.conv1(x))
        # x = F.max_pool2d(x, 2, 2)
        # x = F.relu(self.conv2(x))
        # x = F.max_pool2d(x, 2, 2)
        # x = x.view(-1, 50, 3*100*100)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def transform(img):
    return img.flatten()

# ************************************************************************
# Calculate the error of a model on data from a given loader
# This is used to calculate the validation error every couple of
# thousand batches
# ************************************************************************

def calcError (net, dataloader):
    vloss=0
    vcorrect=0
    vcount=0
    for batch_idx, (data, labels, bb) in enumerate(dataloader):
        y = model(data)
        y = torch.squeeze(y, 1)
        labels = torch.squeeze(labels, 1)
        #labels=torch.LongTensor(np.array(labels.numpy(),np.long))
        loss = crossentropy(y, labels)
        vloss += loss.item()
        predicted = predictedBalls (y.data)
        #_, predicted = torch.max(y.data, 1)
        vcorrect += (predicted == labels).sum().item()
        vcount += BATCHSIZE
    print(vloss, len(dataloader), vloss/len(dataloader))
    return vloss/len(dataloader), 100.0*(1.0-vcorrect/vcount)

def predictedBalls (yBatches):
    _, maxItemsIndex = torch.topk(yBatches, k=3, dim=1)
    for i in range(len(yBatches)):
        for j in range(len(yBatches[i])):
            if j in maxItemsIndex[i]:
                yBatches[i][j] = 1
            else:
                yBatches[i][j] = 0
    return yBatches

if __name__ == "__main__":
    # train_dataset = Balls_CF_Detection ("../mini_balls/train", 20999,
    #     transforms.Normalize([128, 128, 128], [50, 50, 50]))
    dataset = d_loader.Balls_CF_Detection ("../../train/train/train", 20999, transform)

    train_dataset, test_dataset = torch_d.random_split(dataset, [16799, 4200]) 
    
    BATCHSIZE=50
    
    test_loader = torch.utils.data.DataLoader(test_dataset,
        batch_size=BATCHSIZE, shuffle=True)
    
    train_loader = torch.utils.data.DataLoader(train_dataset,
        batch_size=BATCHSIZE, shuffle=True)
    
    img, labels, bb = train_dataset.__getitem__(42)
    
    model = LeNet()

    pos_weight = torch.ones([9])

    # This criterion combines LogSoftMax and NLLLoss in one single class.
    crossentropy = torch.nn.BCEWithLogitsLoss()
    
    # Set up the optimizer: stochastic gradient descent
    # with a learning rate of 0.01
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    # Setting up tensorboard
    #writer = SummaryWriter('runs/mnist_lenet')
    
    # Training
    running_loss = 0.0
    running_correct = 0
    running_count = 0
    
    # Add the graph to tensorboard
    #dataiter = iter(train_loader)
    #data, labels = dataiter.next()
    #writer.add_graph (model, data)
    #writer.flush()
    
    # Cycle through epochs
    for epoch in range(1):
        
        # Cycle through batches
        for batch_idx, (data, labels, bb) in enumerate(train_loader):

            optimizer.zero_grad()
            y = model(data)
            y = torch.squeeze(y, 1)
            labels = torch.squeeze(labels, 1)
            #labels=torch.LongTensor(np.array(labels.numpy(),np.long))
            loss = crossentropy(y, labels)
            print(loss)
            loss.backward()
            running_loss += loss.item()
            optimizer.step()
    
            predicted = predictedBalls (y.data) #torch.max(y.data, 1)

            running_correct += (predicted == labels).sum().item()
            running_count += BATCHSIZE
    
    		# Print statistics
            if (batch_idx % STATS_INTERVAL) == 0:
                train_err = 100.0*(1.0-running_correct / running_count)
                valid_loss, valid_err = calcError (model, test_loader)
                print ('Epoch: %d batch: %5d ' % (epoch + 1, batch_idx + 1), end="")
                print ('train-loss: %.3f train-err: %.3f' % (running_loss / STATS_INTERVAL, train_err), end="")
                print (' valid-loss: %.3f valid-err: %.3f' % (valid_loss, valid_err))
    
                # Write statistics to the log file
                #writer.add_scalars ('Loss', {
                    #'training:': running_loss / STATS_INTERVAL,
                    #'validation:': valid_loss }, 
                   # epoch * len(train_loader) + batch_idx)
    
               # writer.add_scalars ('Error', {
                    #'training:': train_err,
                    #'validation:': valid_err }, 
                    #epoch * len(train_loader) + batch_idx)
                                
                running_loss = 0.0
                running_correct = 0.0
                running_count=0.0

