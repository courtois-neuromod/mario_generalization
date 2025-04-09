import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
import pytorch_lightning as pl
from sklearn.metrics import f1_score, accuracy_score


class BaseModel(nn.Module):
    """Common base class for all models"""

    def __init__(self, n_in):
        super().__init__()
        self.conv1 = nn.Conv2d(n_in, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.linear = nn.Linear(32 * 6 * 6, 512)
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, nn.init.calculate_gain("relu"))
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        return self.linear(x.view(x.size(0), -1))


class PPO(nn.Module):
    """PPO model"""

    def __init__(self, n_in, n_actions):
        super().__init__()
        self.base = BaseModel(n_in)
        self.critic_linear = nn.Linear(512, 1)
        self.actor_linear = nn.Linear(512, n_actions)
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, nn.init.calculate_gain("relu"))
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        x = F.relu(self.base(x))
        return self.actor_linear(x), self.critic_linear(x)


class ResnetProxyModel(pl.LightningModule):
    """ResNet proxy model."""

    def __init__(
        self, n_in, n_out, lr, weight_decay, lr_scheduler, lr_scheduler_kwargs
    ):
        super().__init__()
        self.base = BaseModel(n_in)
        self.linear_out = nn.Linear(512, n_out)
        self._initialize_weights()
        self.lr = lr
        self.weight_decay = weight_decay
        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_kwargs = lr_scheduler_kwargs
        self.save_hyperparameters()
        self.best_val_loss = float("inf")

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, nn.init.calculate_gain("relu"))
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        x = F.relu(self.base(x))
        return self.linear_out(x)

    def training_step(self, batch, batch_idx):
        x = batch["frames"]
        y = batch["resnet_emb"]
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log("tng_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch["frames"]
        y = batch["resnet_emb"]
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log("val_loss", loss, sync_dist=True)
        if loss < self.best_val_loss:
            self.best_val_loss = loss.item()
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = getattr(lr_scheduler, self.lr_scheduler)(
            optimizer, **self.lr_scheduler_kwargs
        )
        return [optimizer], [dict(scheduler=scheduler, interval="step", frequency=1)]


class ImitationModel(pl.LightningModule):
    """Imitation model."""

    def __init__(
        self,
        n_in,
        n_actions,
        lr,
        pos_weight,
        weight_decay,
        lr_scheduler,
        lr_scheduler_kwargs,
    ):
        super().__init__()
        self.base = BaseModel(n_in)
        self.linear_out = nn.Linear(512, n_actions)
        self._initialize_weights()
        self.lr = lr
        self.weight_decay = weight_decay
        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_kwargs = lr_scheduler_kwargs
        self.pos_weight = pos_weight
        if pos_weight is not None:
            pos_weight = torch.tensor(pos_weight)
        self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.bce_val = nn.BCELoss()
        self.save_hyperparameters()
        self.best_val_loss = float("inf")

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, nn.init.calculate_gain("relu"))
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        x = F.relu(self.base(x))
        actions = self.linear_out(x)
        if not self.training:
            actions = torch.sigmoid(actions)
            # During training the Sigmoid is done in the BCEwithLogits loss
        return actions

    def training_step(self, batch, batch_idx):
        x = batch["frames"]
        y = batch["target_action"]
        y_hat = self(x)
        loss = self.bce_loss(y_hat, y)
        self.log("tng_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch["frames"]
        y = batch["target_action"]
        y_hat = self(x)
        loss = self.bce_val(y_hat, y)
        if loss < self.best_val_loss:
            self.best_val_loss = loss.item()
        self.log("val_loss", loss, sync_dist=True)
        actions_thres05 = (y_hat > 0.5).cpu()
        y = y.cpu()
        for i, button in enumerate(("A", "UP", "DOWN", "LEF", "RIGHT", "B")):
            pred_button = actions_thres05[:, i]
            y_button = y[:, i]
            accuracy_thres05 = accuracy_score(y_button, pred_button)
            f1_score_thres05 = f1_score(y_button, pred_button, zero_division=0.0)
            self.log(
                f"val_{button}_f1_score_thres=0.5", f1_score_thres05, sync_dist=True
            )
            self.log(
                f"val_{button}_accuracy_thres=0.5", accuracy_thres05, sync_dist=True
            )
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = getattr(lr_scheduler, self.lr_scheduler)(
            optimizer, **self.lr_scheduler_kwargs
        )
        return [optimizer], [dict(scheduler=scheduler, interval="step", frequency=1)]
