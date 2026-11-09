"""World-model training loop with validation, logging and checkpointing.

The :class:`Trainer` keeps the training logic in one readable place:

* minibatch training over single-step transitions
* a validation pass each epoch (episode-level split -- see
  :func:`lab1.data.dataset.make_dataloaders`)
* TensorBoard logging of every loss term
* checkpoint saving (periodic + best-validation)
* deterministic seeding and tqdm progress bars
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from lab1.config import Config
from lab1.data import load_num_actions, make_dataloaders
from lab1.models import WorldModel
from lab1.utils.checkpoint import save_checkpoint
from lab1.utils.seed import set_seed


def resolve_device(device: str) -> torch.device:
    """Resolve a device string to a :class:`torch.device`.

    ``"auto"`` prefers CUDA, then Apple-Silicon MPS, then CPU. Any other value
    (``"cpu"``/``"cuda"``/``"mps"``) is used as-is.
    """
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


class Trainer:
    """Train a :class:`~lab1.models.WorldModel` from a config and data."""

    def __init__(
        self,
        cfg: Config,
        model: WorldModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
    ) -> None:
        self.cfg = cfg
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=cfg.training.learning_rate,
            weight_decay=cfg.training.weight_decay,
        )
        self.writer = SummaryWriter(log_dir=cfg.training.log_dir)
        self.checkpoint_dir = Path(cfg.training.checkpoint_dir)
        self.best_val = float("inf")
        self.history: list[dict[str, object]] = []

    def _move(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {k: v.to(self.device) for k, v in batch.items()}

    def _loss_kwargs(self) -> dict[str, float]:
        t = self.cfg.training
        return {
            "recon_weight": t.recon_loss_weight,
            "reward_weight": t.reward_loss_weight,
            "dynamics_weight": t.dynamics_loss_weight,
        }

    def train_epoch(self, epoch: int) -> dict[str, float]:
        """Run one training epoch and return mean loss terms."""
        self.model.train()
        totals: dict[str, float] = {}
        n = 0
        bar = tqdm(self.train_loader, desc=f"epoch {epoch} [train]", leave=False)
        for batch in bar:
            batch = self._move(batch)
            losses = self.model.compute_losses(batch, **self._loss_kwargs())
            self.optimizer.zero_grad()
            losses["total"].backward()
            if self.cfg.training.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg.training.grad_clip
                )
            self.optimizer.step()

            bs = batch["obs"].shape[0]
            n += bs
            for k, v in losses.items():
                totals[k] = totals.get(k, 0.0) + float(v.detach()) * bs
            bar.set_postfix(loss=float(losses["total"].detach()))
        return {k: v / max(n, 1) for k, v in totals.items()}

    @torch.no_grad()
    def validate(self, epoch: int) -> dict[str, float]:
        """Run one validation pass and return mean loss terms."""
        self.model.eval()
        totals: dict[str, float] = {}
        n = 0
        for batch in tqdm(self.val_loader, desc=f"epoch {epoch} [val]", leave=False):
            batch = self._move(batch)
            losses = self.model.compute_losses(batch, **self._loss_kwargs())
            bs = batch["obs"].shape[0]
            n += bs
            for k, v in losses.items():
                totals[k] = totals.get(k, 0.0) + float(v) * bs
        return {k: v / max(n, 1) for k, v in totals.items()}

    def fit(self) -> Path:
        """Run the full training loop; return the path to the best checkpoint."""
        best_path = self.checkpoint_dir / "world_model_best.pt"
        for epoch in range(1, self.cfg.training.epochs + 1):
            train_losses = self.train_epoch(epoch)
            val_losses = self.validate(epoch)

            for k, v in train_losses.items():
                self.writer.add_scalar(f"train/{k}", v, epoch)
            for k, v in val_losses.items():
                self.writer.add_scalar(f"val/{k}", v, epoch)

            tqdm.write(
                f"epoch {epoch:3d} | "
                f"train {train_losses['total']:.5f} | "
                f"val {val_losses['total']:.5f}"
            )
            self.history.append(
                {"epoch": epoch, "train": train_losses, "val": val_losses}
            )

            if val_losses["total"] < self.best_val:
                self.best_val = val_losses["total"]
                save_checkpoint(
                    best_path,
                    self.model,
                    self.optimizer,
                    epoch=epoch,
                    extra={"config": self.cfg.to_dict(), "val": val_losses},
                )
            if epoch % self.cfg.training.checkpoint_every == 0:
                save_checkpoint(
                    self.checkpoint_dir / f"world_model_epoch{epoch}.pt",
                    self.model,
                    self.optimizer,
                    epoch=epoch,
                    extra={"config": self.cfg.to_dict()},
                )
        self.writer.close()
        self._save_history()
        return best_path

    def _save_history(self) -> None:
        """Write per-epoch loss history to ``checkpoint_dir/history.json``."""
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = self.checkpoint_dir / "history.json"
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.history, fh, indent=2)


def train(cfg: Config, dataset_path: str | Path | None = None) -> Path:
    """High-level entry point: build data, model and trainer, then fit.

    Parameters
    ----------
    cfg:
        Full configuration.
    dataset_path:
        Optional dataset override; defaults to ``cfg.data.dataset_path``.

    Returns
    -------
    pathlib.Path
        Path to the best-validation checkpoint.
    """
    set_seed(cfg.seed)
    device = resolve_device(cfg.training.device)

    path = dataset_path or cfg.data.dataset_path
    train_loader, val_loader, _, _ = make_dataloaders(cfg, path=path)
    num_actions = load_num_actions(path)

    model = WorldModel.from_config(cfg.model, cfg.env, num_actions)
    trainer = Trainer(cfg, model, train_loader, val_loader, device)
    return trainer.fit()
