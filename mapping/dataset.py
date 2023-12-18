import torch
import numpy as np
import pandas as pd
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

                self.data_dom_1[start:end] = torch.normal(
                    mean=mean_1, std=std_1, size=(size, dim_1)
                )
                self.data_dom_2[start:end] = torch.normal(
                    mean=mean_2, std=std_2, size=(size, dim_2)
                )

        else:
            mean_1, std_1, dim_1 = domain_1
            mean_2, std_2, dim_2 = domain_2

            self.data_dom_1 = torch.normal(mean=mean_1, std=std_1, size=(n_data, dim_1))
            self.data_dom_2 = torch.normal(mean=mean_2, std=std_2, size=(n_data, dim_2))

    def __len__(self) -> int:
        return self.n_data

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.data_dom_1[idx], self.data_dom_2[idx]


class MazeData(Dataset):
    def __init__(
        self, dataset_type: str, domain1_base_path: str, domain2_base_path: str
    ) -> None:
        super().__init__()

        assert dataset_type in ["train", "valid", "test"]
        self.dataset_type = dataset_type

        # init, trag
        if self.dataset_type == "train":
            self.tasks = [
                (12, 16),
                (12, 61),
                (12, 65),
                (15, 11),
                (15, 61),
                (15, 65),
                (21, 16),
                (21, 61),
                (21, 65),
                (26, 11),
                (26, 61),
                (26, 65),
                (46, 11),
                (46, 16),
                (46, 61),
                (51, 11),
                (51, 16),
                (51, 65),
            ]
        elif self.dataset_type == "valid":
            self.tasks = [
                (63, 11),
                (63, 16),
                # (63, 65),
            ]
        elif self.dataset_type == "test":
            self.tasks = [
                (65, 11),
                (65, 16),
                (65, 61),
            ]

        self.domain1_base_path = domain1_base_path
        self.domain2_base_path = domain2_base_path

        self.domain1_data = []
        self.domain2_data = []

        for task in self.tasks:
            domain1_path = self.domain1_base_path + f"/init{task[0]}_targ{task[1]}.csv"
            domain2_path = self.domain2_base_path + f"/init{task[0]}_targ{task[1]}.csv"

            domain1_raw_data = pd.read_csv(
                domain1_path,
                converters={"embedding": pd.eval, "sampled_embedding": pd.eval},
            )
            domain2_raw_data = pd.read_csv(
                domain2_path,
                converters={"embedding": pd.eval, "sampled_embedding": pd.eval},
            )

            for data1, data2 in zip(
                domain1_raw_data["embedding"],
                domain2_raw_data["embedding"],
            ):
                assert len(data1) == len(data2)

                domain1_data = torch.from_numpy(np.array(data1)).to(torch.float32)
                domain2_data = torch.from_numpy(np.array(data2)).to(torch.float32)

                self.domain1_data.append(domain1_data)
                self.domain2_data.append(domain2_data)

    def __len__(self) -> int:
        assert len(self.domain1_data) == len(self.domain2_data)
        return len(self.domain1_data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.domain1_data[idx], self.domain2_data[idx]
