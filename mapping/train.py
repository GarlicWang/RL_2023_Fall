import torch
from torch import optim
from torch.utils.data import DataLoader
import hydra
from omegaconf import DictConfig, OmegaConf

from network import DomainTranslater
from static_flow_vae import VAE
from dataset import PseudoDataset
from trainer import DomainTranslaterTrainer, FlowVAEDomainTranslaterTrainer


def set_seed(seed: int) -> None:
    torch.manual_seed(seed=seed)


@hydra.main(config_path="config", config_name="pseudo_basic")
def main(config: DictConfig) -> None:
    print("Configuration")
    print("=" * 20)
    print(OmegaConf.to_yaml(config), end="")
    print("=" * 20)

    device = config.train.device
    print("Using device:", device)
    print("=" * 20)

    set_seed(config.seed)

    train_set = PseudoDataset(
        n_data=config.data.n_train_data,
        domain_1=(config.data.mean_1, config.data.std_1, config.data.dim_1),
        domain_2=(config.data.mean_2, config.data.std_2, config.data.dim_2),
    )
    valid_set = PseudoDataset(
        n_data=config.data.n_valid_data,
        domain_1=(config.data.mean_1, config.data.std_1, config.data.dim_1),
        domain_2=(config.data.mean_2, config.data.std_2, config.data.dim_2),
    )
    test_set = PseudoDataset(
        n_data=config.data.n_test_data,
        domain_1=(config.data.mean_1, config.data.std_1, config.data.dim_1),
        domain_2=(config.data.mean_2, config.data.std_2, config.data.dim_2),
    )
    print("Train set length:", len(train_set))
    print("Valid set length:", len(valid_set))
    print("Test set length:", len(test_set))
    train_loader = DataLoader(train_set, config.data.batch_size, shuffle=True, pin_memory=True)
    valid_loader = DataLoader(valid_set, config.data.batch_size, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_set, config.data.batch_size, shuffle=False, pin_memory=True)

    # model = DomainTranslater(
    #     input_dim=config.model.input_dim,
    #     output_dim=config.model.output_dim,
    #     latent_dim=config.model.latent_dim,
    # )

    model = VAE(
        dataset="pseudo",
        layer=config.model.layer,
        in_dim=config.model.input_dim,
        hidden_dim=config.model.hidden_dim,
        latent_dim=config.model.latent_dim,
        gate=True,
        flow='nice',
        length=config.model.n_layers,
    )

    optimizer = optim.Adadelta(model.parameters(), lr=config.train.lr, weight_decay=config.train.weight_decay)

    trainer = FlowVAEDomainTranslaterTrainer(
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
