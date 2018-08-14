from env_net import Net
import os
import torch
from env_model_dataset import EnvModelDataset
from torch.utils.data import DataLoader
import torch.nn as nn


hidden_size = 128
use_cuda = False
model_path = "env_model.pth"
validation_data_dir = "results_collected"
num_episodes_in_dir = 200

batch_size = 16

net = Net(hidden_size)
net.eval()
if use_cuda:
    net.cuda()

# check if there are network weights present
if os.path.isfile(model_path):
    net.load_state_dict(torch.load(model_path))


# load the data
dataset = EnvModelDataset(validation_data_dir, 10)

dataloader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=False, num_workers=1)


criterion = nn.MSELoss()

running_loss = 0
skipped_entries = 0
for i_batch, sample_batched in enumerate(dataloader):
    input = sample_batched["in"]
    target = sample_batched["target"]
    if use_cuda:
        input.cuda()
        target.cuda()
    if not input.shape[1] == 177:
        print("skipped wrong shape at: {}".format(i_batch))
        skipped_entries += 1
        continue
    
    output = net(input)
    loss = criterion(output, target)

    running_loss += loss.item()

average_loss = running_loss / float(len(dataloader))
print("The average loss is: {}".format(average_loss))