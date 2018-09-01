from torch.utils.data import Dataset, DataLoader
from episode_serializer import CSVEpisodeParser
import numpy as np
import os
import torch

EPSILON = 0.00000001
# if True use x = (x - min)/(max - min), else use mean and std normalization (whitening) x = (x - mean)/stdd
MIN_MAX_NORM = True 

class EnvModelDataset(Dataset):
    def __init__(self, base_folder, num_episodes):
        folder_name=base_folder
        episode_names=map(lambda x: os.path.join(folder_name, "episode{:05d}.csv".format(x)), np.arange(num_episodes))

        data = []
        loader = CSVEpisodeParser()
        np_states = []
        np_targets = []
        np_actions = []
        for name in episode_names:
            print("load episode: {}".format(name))
            episode = loader.parse_episode(name)
            data.extend(episode.experience_tuples)
        for element in data:
            state = element.initial_state
            action = element.action
            target = element.final_state

            state = np.array(state)
            action = np.array(action)
            target = np.array(target)

            if not (state.shape[0] == 158):
                print("bad state len: {}".format(state.shape))
            if not (target.shape[0] == 158):
                print("bad target len: {}".format(target.shape))
            np_states.append(state)
            np_targets.append(target)
            np_actions.append(action)
        self.np_states = np.array(np_states, dtype=np.float32)
        self.np_targets = np.array(np_targets, dtype=np.float32)
        self.np_actions = np.array(np_actions, dtype=np.float32)

        if MIN_MAX_NORM:
            self.state_mins = self.np_states.min(axis=0)
            self.state_maxs = self.np_states.max(axis=0)

            self.target_mins = self.np_targets.min(axis=0)
            self.target_maxs = self.np_targets.max(axis=0)

            self.np_states = self.np_states - self.state_mins
            self.np_targets = self.np_targets - self.target_mins

            self.np_states = self.np_states / (self.state_maxs - self.state_mins + EPSILON)
            self.np_targets = self.np_targets / (self.target_maxs - self.target_mins + EPSILON)

            print("state max: {}".format(self.state_maxs))
            print("state min: {}".format(self.state_mins))
        else:
            self.state_means = self.np_states.mean(axis=0)
            self.state_stds = self.np_states.var(axis=0) + EPSILON

            self.targets_means = self.np_targets.mean(axis=0)
            self.targets_stds = self.np_targets.var(axis=0) + EPSILON

            self.np_states = self.np_states - self.state_means
            self.np_targets = self.np_targets - self.targets_means

            self.np_states = self.np_states / self.state_stds
            self.np_targets = self.np_targets / self.targets_stds

            print("state means: {}".format(self.state_means))
            print("state stds: {}".format(self.targets_stds))

        print("np_states shape: {}".format(self.np_states.shape))


    def __len__(self):
        return self.np_states.shape[0]

    def __getitem__(self, idx):
        

        state = self.np_states[idx]
        action = self.np_actions[idx]
        target = self.np_targets[idx]
        
        input = np.concatenate([state, action])

        input = torch.from_numpy(input)
        target = torch.from_numpy(target)

        if not input.shape[0] == 177:
            print("bad state len at {}: {}".format(idx, input))
        if not target.shape[0] == 158:
            print("bad target len at {}: {}".format(idx, target.size()))
        item = {"in": input, "target": target}
        return item
    
