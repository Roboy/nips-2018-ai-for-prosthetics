from torch.utils.data import Dataset, DataLoader
from serializer import CSVEpisodeDeserializer
import numpy as np
import os
import torch

class EnvModelDataset(Dataset):
    def __init__(self, base_folder, num_episodes):
        folder_name=base_folder
        episode_names=map(lambda x: os.path.join(folder_name, "episode{:05d}.csv".format(x)), np.arange(num_episodes))

        self.data = []
        loader = CSVEpisodeDeserializer()
        for name in episode_names:
            print("load episode: {}".format(name))
            self.data.extend(loader.deserialize_episode(name))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_elem = self.data[idx]
        

        state = data_elem.initial_state
        action = data_elem.action
        
        input = state+action

        target = data_elem.final_state

        input = torch.FloatTensor(input)
        target = torch.FloatTensor(target)

        if not input.shape[0] == 177:
            print("bad state len at {}: {}".format(idx, input))
        if not target.shape[0] == 158:
            print("bad target len at {}: {}".format(idx, target.size()))
        item = {"in": input, "target": target}
        return item
    
