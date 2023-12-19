import torch
from torch import optim
import pandas as pd
from torch.utils.data import DataLoader
import hydra
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf

from network import DomainTranslater
from static_flow_vae import VAE
from dataset import PseudoDataset, MazeInferenceData
from trainer import DomainTranslaterTrainer, FlowVAEDomainTranslaterTrainer
from transformer import Transformer


def set_seed(seed: int) -> None:
    torch.manual_seed(seed=seed)


@hydra.main(config_path="config", config_name="transformer")
def main(config: DictConfig) -> None:
    print("Configuration")
    print("=" * 20)
    print(OmegaConf.to_yaml(config), end="")
    print("=" * 20)

    device = config.train.device
    print("Using device:", device)
    print("=" * 20)

    set_seed(config.seed)

    test_set = MazeInferenceData(
        dataset_type="inference",
        domain1_base_path="test_data",
    )

    print("Test set length:", len(test_set))
    test_loader = DataLoader(test_set, 1, shuffle=False, pin_memory=True)

    model = Transformer(
        input_dim=config.model.input_dim,
        hidden_dim=config.model.hidden_dim,
        output_dim=config.model.output_dim,
        n_layer=config.model.layer,
        dim_head=config.model.dim_head,
        n_head=config.model.n_head,
        dropout=config.model.dropout,
    )

    model.load_state_dict(torch.load("checkpoint/transformer/best_model_1.5318.pt"))
    model = model.to("cuda:0")
    model.eval()

    df = pd.DataFrame(columns=["init", "targ", "rollout_id", "seq", "embedding"])

    for raw_data in tqdm(test_loader):
        data, init, targ, rollout, seq = raw_data
        init = init.item()
        targ = targ.item()
        rollout = rollout.item()
        seq = seq.item()

        data = data.to("cuda:0")
        data = data.reshape(1, 1, -1)
        y = model(data)
        y = y.reshape(-1).detach().cpu().numpy()
        df.loc[len(df)] = [init, targ, rollout, seq, y]

    # deal with list in dataframe
    df["seq"] = df["seq"].apply(lambda x: str(x))

    df.to_csv("inference.csv", index=False)
    df.to_parquet("inference.parquet", index=False)


if __name__ == "__main__":
    main()
