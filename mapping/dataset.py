import torch
from torch.utils.data import Dataset
from typing import *


class PseudoDataset(Dataset):
    def __init__(
        self,
        n_data: int = 10000,
        domain_1: Tuple[float, float, int] = (0.0, 1.0, 1024),
        domain_2: Tuple[float, float, int] = (1.0, 2.0, 1024),
    ) -> None:
        """
        param:
            n_data: Number of data.
            domain_1: (mean_1, std_1, dim_1). Mean and std are for the normal distribution from which this dataset samples data. Dim is the latent feature dimension.
            domain_2: (mean_2, std_2, dim_2). Mean and std are for the normal distribution from which this dataset samples data. Dim is the latent feature dimension.
        """
        super().__init__()

        self.n_data = n_data

        mean_1, std_1, dim_1 = domain_1
        mean_2, std_2, dim_2 = domain_2

        self.data_dom_1 = torch.normal(mean=mean_1, std=std_1, size=(n_data, dim_1))
        self.data_dom_2 = torch.normal(mean=mean_2, std=std_2, size=(n_data, dim_2))

    def __len__(self) -> int:
        return self.n_data

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.data_dom_1[idx], self.data_dom_2[idx]
