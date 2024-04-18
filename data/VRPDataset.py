import numpy
import torch
from torch.utils.data import Dataset


class VRPDataset(Dataset):
    CAPACITIES = {
        5: numpy.float32(15),
        10: numpy.float32(20),
        20: numpy.float32(30),
        40: numpy.float32(40),
        50: numpy.float32(40),
        100: numpy.float32(50)
    }

    # code copied from https://github.com/wouterkool/attention-learn-to-route/blob/master/problems/vrp/problem_vrp.py
    def __init__(self, size, num_samples):

        self.graph_size = size
        # locations[0] is the depot, locations[1:] are the clients
        self.locations = []

        # depot demand : demands[0]= 0, client demands : demands[1:] uniform(1,9)
        self.demands = []

        self.capacities = []

        for sample in range(num_samples):
            self.capacities.append(self.CAPACITIES[self.graph_size])
            self.locations.append(torch.FloatTensor(size, 2).uniform_(0, 1))
            self.demands.append(torch.randint(low=0, high=9, size=(size,), dtype=torch.float32) + 1.0)
            self.demands[-1][0] = 0


        self.size = len(self.locations)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.locations[idx], self.demands[idx], self.capacities[idx]

