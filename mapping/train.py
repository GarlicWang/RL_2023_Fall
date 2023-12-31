import torch
from torch import optim
from torch.utils.data import DataLoader
import hydra
from omegaconf import DictConfig, OmegaConf

from network import DomainTranslater
from static_flow_vae import VAE
from dataset import PseudoDataset, MazeData
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

    train_set = MazeData(
        dataset_type="train",
        domain1_base_path=config.data.domain1_base_path,
        domain2_base_path=config.data.domain2_base_path,
    )
    valid_set = MazeData(
        dataset_type="valid",
        domain1_base_path=config.data.domain1_base_path,
        domain2_base_path=config.data.domain2_base_path,
    )
    test_set = MazeData(
        dataset_type="test",
        domain1_base_path=config.data.domain1_base_path,
        domain2_base_path=config.data.domain2_base_path,
    )

    print("Train set length:", len(train_set))
    print("Valid set length:", len(valid_set))
    print("Test set length:", len(test_set))
    train_loader = DataLoader(
        train_set, config.data.batch_size, shuffle=True, pin_memory=True
    )
    valid_loader = DataLoader(
        valid_set, config.data.batch_size, shuffle=True, pin_memory=True
    )
    test_loader = DataLoader(
        test_set, config.data.batch_size, shuffle=False, pin_memory=True
    )

    model = Transformer(
        input_dim=config.model.input_dim,
        hidden_dim=config.model.hidden_dim,
        output_dim=config.model.output_dim,
        n_layer=config.model.layer,
        dim_head=config.model.dim_head,
        n_head=config.model.n_head,
        dropout=config.model.dropout,
    )

    optimizer = optim.Adadelta(
        model.parameters(), lr=config.train.lr, weight_decay=config.train.weight_decay
    )

    trainer = DomainTranslaterTrainer(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        epoch=config.train.epoch,
        device=device,
        enable_wandb=config.enable_wandb,
        project=config.project,
        name=config.name,
        config=OmegaConf.to_object(config),
    )

    trainer.run()


if __name__ == "__main__":
    main()
