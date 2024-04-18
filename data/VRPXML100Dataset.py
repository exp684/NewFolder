import math
import random

import torch
from torch.utils.data import Dataset


def distance(x, y):
    return math.sqrt((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2)


class VRPXML100Dataset(Dataset):
    # code copied from the generator of XML100 dataset
    def __init__(self, size, num_samples, seed=1234):

        self.graph_size = size
        # locations[0] is the depot, locations[1:] are the clients
        self.locations = []

        # depot demand : demands[0]= 0, client demands : demands[1:] uniform(1,9)
        self.demands = []

        self.capacities = []

        # constants
        max_coord = 1000
        decay = 40
        n = size - 1
        random.seed(seed)

        for _ in range(math.ceil(num_samples / (3 * 3 * 7 * 6))):
            for rootPos in (1, 2, 3):
                for custPos in (1, 2, 3):
                    for demandType in (1, 2, 3, 4, 5, 6, 7):
                        for avgRouteSize in (1, 2, 3, 4, 5, 6):
                            n_seeds = random.randint(2, 6)
                            r_bornes = {1: (3, 5), 2: (5, 8), 3: (8, 12), 4: (12, 16), 5: (16, 25), 6: (25, 50)}
                            r = random.uniform(r_bornes[avgRouteSize][0], r_bornes[avgRouteSize][1])
                            coordinates = set()  # set of coordinates for the customers

                            # Root positioning
                            if rootPos == 1:
                                x_ = random.randint(0, max_coord)
                                y_ = random.randint(0, max_coord)
                            elif rootPos == 2:
                                x_ = y_ = int(max_coord / 2.0)
                            elif rootPos == 3:
                                x_ = y_ = 0
                            else:
                                print("Depot Positioning out of range!")
                                exit(0)
                            depot = (x_, y_)

                            # Customer positioning
                            if custPos == 3:
                                n_rand_cust = int(n / 2.0)
                            elif custPos == 2:
                                n_rand_cust = 0
                            elif custPos == 1:
                                n_rand_cust = n
                                n_seeds = 0
                            else:
                                print("Costumer Positioning out of range!")
                                exit(0)

                            n_clust_cust = n - n_rand_cust

                            # Generating random customers
                            for i in range(1, n_rand_cust + 1):
                                x_ = random.randint(0, max_coord)
                                y_ = random.randint(0, max_coord)
                                while (x_, y_) in coordinates or (x_, y_) == depot:
                                    x_ = random.randint(0, max_coord)
                                    y_ = random.randint(0, max_coord)
                                coordinates.add((x_, y_))

                            n_s = n_rand_cust

                            seeds = []
                            # Generation of the clustered customers
                            if n_clust_cust > 0:
                                if n_clust_cust < n_seeds:
                                    print("Too many seeds!")
                                    exit(0)

                                # Generate the seeds
                                for i in range(n_seeds):
                                    x_ = random.randint(0, max_coord)
                                    y_ = random.randint(0, max_coord)
                                    while (x_, y_) in coordinates or (x_, y_) == depot:
                                        x_ = random.randint(0, max_coord)
                                        y_ = random.randint(0, max_coord)
                                    coordinates.add((x_, y_))
                                    seeds.append((x_, y_))
                                n_s = n_s + n_seeds

                                # Determine the seed with maximum sum of weights (w.r.t. all seeds)
                                max_weight = 0.0
                                for i, j in seeds:
                                    w_ij = 0.0
                                    for i_, j_ in seeds:
                                        w_ij += 2 ** (- distance((i, j), (i_, j_)) / decay)
                                    if w_ij > max_weight:
                                        max_weight = w_ij

                                norm_factor = 1.0 / max_weight

                                # Generate the remaining customers using Accept-reject method
                                while n_s < n:
                                    x_ = random.randint(0, max_coord)
                                    y_ = random.randint(0, max_coord)
                                    while (x_, y_) in coordinates or (x_, y_) == depot:
                                        x_ = random.randint(0, max_coord)
                                        y_ = random.randint(0, max_coord)

                                    weight = 0.0
                                    for i_, j_ in seeds:
                                        weight += 2 ** (- distance((x_, y_), (i_, j_)) / decay)
                                    weight *= norm_factor
                                    rand = random.uniform(0, 1)

                                    if rand <= weight:  # Will we accept the customer?
                                        coordinates.add((x_, y_))
                                        n_s = n_s + 1

                            vertices = [depot] + list(coordinates)  # set of vertices (from now on, the ids are defined)

                            # Demands
                            demand_min_values = [1, 1, 5, 1, 50, 1, 51, 50, 1]
                            demand_max_values = [1, 10, 10, 100, 100, 50, 100, 100, 10]
                            demand_min = demand_min_values[demandType - 1]
                            demand_max = demand_max_values[demandType - 1]
                            demand_min_even_quadrant = 51
                            demand_max_even_quadrant = 100
                            demand_min_large = 50
                            demand_max_large = 100
                            large_per_route = 1.5
                            demand_min_small = 1
                            demand_max_small = 10

                            demands = []  # demands
                            sum_demands = 0
                            max_demand = 0

                            for i in range(2, n + 2):
                                j = int((demand_max - demand_min + 1) * random.uniform(0, 1) + demand_min)
                                if demandType == 6:
                                    if (vertices[i - 1][0] < max_coord / 2.0 and vertices[i - 1][1] < max_coord / 2.0) or (
                                            vertices[i - 1][0] >= max_coord / 2.0 and vertices[i - 1][1] >= max_coord / 2.0):
                                        j = int(
                                            (demand_max_even_quadrant - demand_min_even_quadrant + 1) * random.uniform(
                                                0,
                                                1) + demand_min_even_quadrant)
                                if demandType == 7:
                                    if i < (n / r) * large_per_route:
                                        j = int(
                                            (demand_max_large - demand_min_large + 1) * random.uniform(0,
                                                                                                       1) + demand_min_large)
                                    else:
                                        j = int(
                                            (demand_max_small - demand_min_small + 1) * random.uniform(0,
                                                                                                       1) + demand_min_small)
                                demands.append(j)
                                if j > max_demand:
                                    max_demand = j
                                sum_demands = sum_demands + j

                            # Generate capacity
                            if sum_demands == n:
                                capacity = math.floor(r)
                            else:
                                capacity = max(max_demand, math.ceil(r * sum_demands / n))

                            self.capacities.append(capacity)

                            nodes = []
                            for i, v in enumerate(vertices):
                                nodes.append((v[0], v[1]))
                            self.locations.append(nodes)

                            if demandType != 6:
                                random.shuffle(demands)
                            demands = [0] + demands

                            self.demands.append(demands)

                            assert len(demands) == len(nodes) == size

                            assert len(self.locations) == len(self.demands) == len(self.capacities)

        self.size = len(self.locations)
        self.locations = torch.FloatTensor(self.locations)
        self.demands = torch.Tensor(self.demands)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.locations[idx], self.demands[idx], self.capacities[idx]
