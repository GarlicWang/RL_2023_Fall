import torch
import numpy as np
from torch.utils.data import Dataset
from typing import *


class PseudoDataset(Dataset):
    def __init__(
        self,
        n_data: int = 10000,
        domain_1: Tuple[float, float, int] = (-10.0, 1.0, 1024),
        domain_2: Tuple[float, float, int] = (10.0, 2.0, 1024),
        num_cluster: int = 10,
        use_cluster: bool = True,
    ) -> None:
        """
        param:
            n_data: Number of data.
            domain_1: (mean_1, std_1, dim_1). Mean and std are for the normal distribution from which this dataset samples data. Dim is the latent feature dimension.
            domain_2: (mean_2, std_2, dim_2). Mean and std are for the normal distribution from which this dataset samples data. Dim is the latent feature dimension.
        """
        super().__init__()

        self.n_data = n_data
        self.num_cluster = num_cluster
        self.use_cluster = use_cluster

        self.data_dom_1 = torch.zeros((n_data, domain_1[2]))
        self.data_dom_2 = torch.zeros((n_data, domain_2[2]))

        if self.use_cluster:
            for multiplier in range(self.num_cluster):
                start = multiplier * (n_data // self.num_cluster)
                end = (multiplier + 1) * (n_data // self.num_cluster)
                size = end - start

                mean_1, std_1, dim_1 = domain_1
                mean_2, std_2, dim_2 = domain_2

                mean_1 -= multiplier
                mean_2 += multiplier

                std_1 += np.abs(np.random.normal(0, 1))
                std_2 += np.abs(np.random.normal(0, 1))

                self.data_dom_1[start:end] = torch.normal(mean=mean_1, std=std_1, size=(size, dim_1))
                self.data_dom_2[start:end] = torch.normal(mean=mean_2, std=std_2, size=(size, dim_2))

        else:
            mean_1, std_1, dim_1 = domain_1
            mean_2, std_2, dim_2 = domain_2

            self.data_dom_1 = torch.normal(mean=mean_1, std=std_1, size=(n_data, dim_1))
            self.data_dom_2 = torch.normal(mean=mean_2, std=std_2, size=(n_data, dim_2))

    def __len__(self) -> int:
        return self.n_data

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.data_dom_1[idx], self.data_dom_2[idx]
