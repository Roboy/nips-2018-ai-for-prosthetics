import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os.path
from env_model_dataset import EnvModelDataset
from torch.utils.data import DataLoader
from env_net import Net

num_epochs = 1
file_path = "env_model.pth"
use_cuda = False
hidden_size=128
batch_size=16

net = Net(hidden_size)
net.train()
if use_cuda:
    net = net.cuda()

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

dataloader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=True, num_workers=1)

for epoch in range(num_epochs):
    print("Start of epoch {} of {}".format(epoch, num_epochs))
    running_loss = 0.0
    for i_batch, sample_batched in enumerate(dataloader):
        # get the inputs
        
        input = sample_batched["in"]
        target = sample_batched["target"]
        if use_cuda:
            input = input.cuda()
            target = target.cuda()
        if not input.shape[1] == 177:
            print("skipped wrong shape at: {}".format(i_batch))
            continue
        if not input.shape[0] == batch_size:
            print("bad batch size: {}... skipping".format(input.shape[0]))
            continue
        
        # in your training loop:
        optimizer.zero_grad()   # zero the gradient buffers
        output = net(input)

        loss = criterion(output, target)
        loss.backward()

        optimizer.step()    # Does the update

        # print statistics
        running_loss += loss.item()
        if i_batch == 0:
+            print('[%d, %5d] loss: %.3f' %
+                    (epoch + 1, i_batch + 1, running_loss))
        if i_batch % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i_batch + 1, running_loss / 2000))
            running_loss = 0.0
print('Finished Training')

torch.save(net.state_dict(), file_path)
print("Saved model at: {}".format(file_path))