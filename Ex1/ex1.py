import tp_src.dataset_det as d_loader
import torch.utils.data as torch_d


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
        # x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

if __name__ == "__main__":
    # train_dataset = Balls_CF_Detection ("../mini_balls/train", 20999,
    #     transforms.Normalize([128, 128, 128], [50, 50, 50]))
    dataset = d_loader.Balls_CF_Detection ("train", 20999)

    train_dataset, test_dataset = torch_d.random_split(dataset, [20999*0.8, 20999*0.2]) 

    print(len(train_dataset))
 #    BATCHSIZE=50



	# valid_dataset = MNISTDataset ("MNIST-png/testing", 
 #    transforms.Compose([
 #    transforms.ToTensor(),
 #    transforms.Normalize((0.1307,), (0.3081,))])) # mean, std of dataset
	# valid_loader = torch.utils.data.DataLoader(valid_dataset,
 #    batch_size=BATCHSIZE, shuffle=True)














 #    img,p,b = train_dataset.__getitem__(42)

 #    print ("Presence:")
 #    print (p)

 #    print ("Pose:")
 #    print (b)