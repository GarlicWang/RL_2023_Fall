from typing import Tuple
import torch
from torch import optim, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from typing import *
import wandb
from tqdm import tqdm

from tools import gradient_norm
from network import DomainTranslater


class BasicTrainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        test_loader: DataLoader,
        optimizer: optim.Optimizer,
        epoch: int,
        device: Union[str, torch.device] = torch.device("cuda"),
        enable_wandb: bool = True,
        project: str = "RL_final_mapping",
        name: str = "basic",
        config: object = None,
    ) -> None:
        self.model = model.to(device)

        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader

        self.optimizer = optimizer
        self.device = device
        self.epoch = epoch

        self.enable_wandb = enable_wandb
        if enable_wandb:
            self.record = wandb.init(project=project, name=name, config=config)

    def _train(self, data: Tuple[torch.Tensor, torch.Tensor]) -> float:
        """
        return:
            Training loss
        """
        raise NotImplementedError()

    def train(self) -> None:
        self.model.train()

        best_loss = 1e6
        loss_history = []
        grad_norm = []

        for data in tqdm(self.train_loader):
            data = [d.to(self.device) for d in data]
            loss = self._train(data)

            # record
            best_loss = min(best_loss, loss)
            loss_history.append(loss)
            grad_norm.append(gradient_norm(self.model))

        loss_avg = sum(loss_history) / len(loss_history)
        grad_norm = sum(grad_norm) / len(grad_norm)

        if self.enable_wandb:
            wandb.log(
                {
                    "train_avg_loss": loss_avg,
                    "train_best_loss": best_loss,
                    "train_grad_norm": grad_norm,
                }
            )

        print(
            f"[Train] Avg loss: {loss_avg:.4f}, best loss: {best_loss:.4f}, gradient norm: {grad_norm:.4f}"
        )

    def _valid(self, data: Tuple[torch.Tensor, torch.Tensor]) -> float:
        """
        return:
            Validation loss
        """
        raise NotImplementedError()

    @torch.no_grad()
    def valid(self) -> None:
        self.model.eval()

        best_loss = 1e6
        loss_history = []
        for data in tqdm(self.valid_loader):
            data = [d.to(self.device) for d in data]
            loss = self._valid(data)

            # record
            if loss < best_loss:
                best_loss = loss
                torch.save(self.model.state_dict(), f"best_model.pt")

            loss_history.append(loss)

        loss_avg = sum(loss_history) / len(loss_history)

        if self.enable_wandb:
            wandb.log(
                {
                    "valid_avg_loss": loss_avg,
                    "valid_best_loss": best_loss,
                }
            )

        print(f"[Valid] Avg loss: {loss_avg:.4f}, best loss: {best_loss:.4f}")

    def _test(self, data: Tuple[torch.Tensor, torch.Tensor]) -> float:
        """
        return:
            Testing loss
        """
        raise NotImplementedError()

    @torch.no_grad()
    def test(self) -> None:
        self.model.eval()

        best_loss = 1e6
        loss_history = []
        for data in tqdm(self.test_loader):
            data = [d.to(self.device) for d in data]
            loss = self._test(data)

            # record
            best_loss = min(best_loss, loss)
            loss_history.append(loss)

        loss_avg = sum(loss_history) / len(loss_history)

        if self.enable_wandb:
            wandb.log(
                {
                    "test_avg_loss": loss_avg,
                    "test_best_loss": best_loss,
                }
            )

        print(f"[Test] Avg loss: {loss_avg:.4f}, best loss: {best_loss:.4f}")

    def run(self) -> None:
        for epoch in range(self.epoch):
            print(f"[Epoch {epoch:04d}/{self.epoch:04d}]")
            self.train()
            self.valid()
            self.test()

        if self.enable_wandb:
            wandb.finish()


class DomainTranslaterTrainer(BasicTrainer):
    def __init__(
        self,
        model: DomainTranslater,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        test_loader: DataLoader,
        optimizer: optim.Optimizer,
        epoch: int,
        device: Union[str, torch.device] = torch.device("cuda"),
        enable_wandb: bool = True,
        project: str = "RL_final_mapping",
        name: str = "domain_translation",
        config: object = None,
    ) -> None:
        super().__init__(
            model,
            train_loader,
            valid_loader,
            test_loader,
            optimizer,
            epoch,
            device,
            enable_wandb,
            project,
            name,
            config,
        )

    def __loss_fn(self, source: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(source, gt)

    def _train(self, data: Tuple[torch.Tensor, torch.Tensor]) -> float:
        feat_dom_1, feat_dom_2 = data

        output = self.model(feat_dom_1)

        self.optimizer.zero_grad()

        loss = self.__loss_fn(output, feat_dom_2)

        loss.backward()
        self.optimizer.step()

        return loss.detach().item()

    def _valid(self, data: Tuple[torch.Tensor, torch.Tensor]) -> float:
        feat_dom_1, feat_dom_2 = data

        output = self.model(feat_dom_1)
        loss = self.__loss_fn(output, feat_dom_2)
        return loss.item()

    def _test(self, data: Tuple[torch.Tensor, torch.Tensor]) -> float:
        feat_dom_1, feat_dom_2 = data

        output = self.model(feat_dom_1)
        loss = self.__loss_fn(output, feat_dom_2)
        return loss.item()


class FlowVAEDomainTranslaterTrainer(BasicTrainer):
    def __init__(
        self,
        model: DomainTranslater,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        test_loader: DataLoader,
        optimizer: optim.Optimizer,
        epoch: int,
        device: Union[str, torch.device] = torch.device("cuda"),
        enable_wandb: bool = True,
        project: str = "RL_final_mapping",
        name: str = "domain_translation",
        config: object = None,
    ) -> None:
        super().__init__(
            model,
            train_loader,
            valid_loader,
            test_loader,
            optimizer,
            epoch,
            device,
            enable_wandb,
            project,
            name,
            config,
        )

    def __loss_fn(self, source: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(source, gt)

    def _train(self, data: Tuple[torch.Tensor, torch.Tensor]) -> float:
        feat_dom_1, feat_dom_2 = data

        y_hat, mean, log_var, log_det = self.model(feat_dom_1)

        # loss = self.model.loss(y_hat, feat_dom_2, mean, log_var, log_det).mean()
        loss = self.__loss_fn(y_hat, feat_dom_2)

        self.optimizer.zero_grad()

        loss.backward()
        self.optimizer.step()

        return loss.detach().item()

    def _valid(self, data: Tuple[torch.Tensor, torch.Tensor]) -> float:
        feat_dom_1, feat_dom_2 = data

        y_hat, mean, log_var, log_det = self.model(feat_dom_1)
        # loss = self.model.loss(y_hat, feat_dom_2, mean, log_var, log_det).mean()
        loss = self.__loss_fn(y_hat, feat_dom_2)
        return loss.item()

    def _test(self, data: Tuple[torch.Tensor, torch.Tensor]) -> float:
        feat_dom_1, feat_dom_2 = data

        y_hat, mean, log_var, log_det = self.model(feat_dom_1)
        # loss = self.model.loss(y_hat, feat_dom_2, mean, log_var, log_det).mean()
        loss = self.__loss_fn(y_hat, feat_dom_2)
        return loss.item()
