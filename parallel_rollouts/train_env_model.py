import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os.path
from env_model_dataset import EnvModelDataset
from torch.utils.data import DataLoader

num_epochs = 5
file_path = "env_model.pth"

class Net(nn.Module):
    def __init__(self, hidden_layer_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(158+19, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, hidden_layer_size)
        self.bn2 = nn.BatchNorm1d(hidden_layer_size)
        self.fc3 = nn.Linear(hidden_layer_size, 158)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.bn2(F.relu(self.bn1(self.fc2(x))))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

net = Net(128)
net.train()

# check if there is some already present state, if so, load it
if os.path.isfile(file_path):
    net.load_state_dict(torch.load(file_path))

criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.0001)

# load the data
dataset = EnvModelDataset("results_collected", 2400)

# print some training samples
rnd = np.random.randint(len(dataset), size=2)
for r in rnd:
    item = dataset[r]
    input = item["in"]
    print("input: {}".format(input))
    # print("target: {}".format(s_prime))

dataloader = DataLoader(dataset, batch_size=4,
                        shuffle=False, num_workers=1)

for epoch in range(num_epochs):
    print("Start of epoch {} of {}".format(epoch, num_epochs))
    running_loss = 0.0
    for i_batch, sample_batched in enumerate(dataloader):
        # get the inputs
        input = sample_batched["in"]
        target = sample_batched["target"]

        if not input.shape[1] == 177:
            print("skipped wrong shape at: {}".format(i_batch))
            continue
        
        # in your training loop:
        optimizer.zero_grad()   # zero the gradient buffers
        output = net(input)

        loss = criterion(output, target)
        loss.backward()

        optimizer.step()    # Does the update

        # print statistics
        running_loss += loss.item()
        if i_batch % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i_batch + 1, running_loss / 2000))
            running_loss = 0.0
print('Finished Training')

torch.save(net.state_dict(), file_path)
print("Saved model at: {}".format(file_path))