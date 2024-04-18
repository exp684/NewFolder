from typing import Optional

import cvrplib
import torch
from torch.utils.data import Dataset


class VRPLiteratureDataset(Dataset):
    def __init__(self,
                 low: Optional[int] = None,
                 high: Optional[int] = None):
        names = cvrplib.list_names(low=low, high=high, vrp_type="cvrp")

        # locations[0] is the depot, locations[1:] are the clients
        self.locations = []

        # depot demand : demands[0]= 0, client demands : demands[1:] uniform(1,9)
        self.demands = []

        self.capacities = []

        for name in names:
            print("Downloading ", name)
            try:
                instance = cvrplib.download(name)
                if instance.coordinates is None:
                    continue
                self.locations.append(instance.coordinates)
                self.demands.append(instance.demands)
                self.capacities.append(instance.capacity)
            except:
                print("Failed to download ", name)
                continue

        self.size = len(self.locations)
        assert len(self.locations) == len(self.demands) == len(self.capacities) == self.size
        self.locations = torch.FloatTensor(self.locations)
        self.demands = torch.Tensor(self.demands)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.locations[idx], self.demands[idx], self.capacities[idx]
