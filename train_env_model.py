import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os.path

num_epochs = 5
file_path = "env_model.pth"

class Net(nn.Module):
    def __init__(self, hidden_layer_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(158+19, 128)
        self.fc2 = nn.Linear(128, hidden_layer_size)
        self.fc3 = nn.Linear(hidden_layer_size, 158)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

net = Net(128)

# check if there is some already present state, if so, load it
if os.path.isfile(file_path):
    net.load_state_dict(torch.load(file_path))

criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.01)

# load the data
train_data = []
for i in range(20000):
    input = torch.randn(1, 158+19) # change this to use the loaded values
    target = torch.randn(1, 158)
    train_data.append ((input, target))

# print some training samples
rnd = np.random.randint(len(train_data), size=2)
for r in rnd:
    x, y = train_data[r]
    print("input: {}".format(x))
    print("target: {}".format(y))

for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(train_data):
        # get the inputs
        input, target = data
        input = torch.from_numpy(input)
        target = torch.from_numpy(target)
        
        # in your training loop:
        optimizer.zero_grad()   # zero the gradient buffers
        output = net(input)

        loss = criterion(output, target)
        loss.backward()

        optimizer.step()    # Does the update

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0
print('Finished Training')

torch.save(net.state_dict(), file_path)
print("Saved model at: {}".format(file_path))