import os
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from .config import (
    DEVICE, BATCH_SIZE, RANDOM_SEED, GRID_SIZE,
    MLP_LR, MLP_EPOCHS, MLP_PATIENCE,
    CFP_LR, CFP_MOMENTUM, CFP_EPOCHS,
    MLP_TRAIN_SIZE, MLP_VAL_SIZE,
    CFP_TRAIN_SIZE, CFP_VAL_SIZE,
    MLP_ENSEMBLE_SIZE,
)
from .models.mlp_model import MLPModel, EnsembleMLPModel
from .models.cfp_model import CFPModel


class MSEPlusL1Loss(nn.Module):
    def __init__(self, l1_weight=0.1):
        super().__init__()
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()
        self.l1_weight = l1_weight

    def forward(self, pred, target):
        return self.mse(pred, target) + self.l1_weight * self.l1(pred, target)


def train_loop(model, train_loader, val_loader, criterion, optimizer,
               scheduler, epochs, patience, device, use_amp=False):
    """Training loop with optional early stopping.

    Returns:
        history: dict with 'train_loss' and 'val_loss' lists.
    """
    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None
    history = {"train_loss": [], "val_loss": []}
    scaler = torch.amp.GradScaler(enabled=use_amp)

    step_per_batch = isinstance(
        scheduler, torch.optim.lr_scheduler.OneCycleLR
    ) if scheduler is not None else False

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                output = model(batch_x)
                loss = criterion(output, batch_y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if step_per_batch:
                scheduler.step()

            total_train_loss += loss.item() * batch_x.size(0)

        train_loss = total_train_loss / len(train_loader.dataset)

        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    output = model(batch_x)
                    loss = criterion(output, batch_y)
                total_val_loss += loss.item() * batch_x.size(0)

        val_loss = total_val_loss / len(val_loader.dataset)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch + 1}/{epochs} - "
              f"Train: {train_loss:.6f} - Val: {val_loss:.6f} - LR: {lr:.2e}")

        if scheduler is not None and not step_per_batch:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1
            if patience is not None and patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return history


def train_mlp(data_dir="data", checkpoint_dir="checkpoints"):
    """Train an ensemble of MLP models with different seeds."""
    os.makedirs(checkpoint_dir, exist_ok=True)

    data = np.load(os.path.join(data_dir, "mlp_data.npz"))
    rss = data["rss"]
    theta_ideal = data["theta_ideal"]

    X_train = torch.FloatTensor(rss[:MLP_TRAIN_SIZE])
    y_train = torch.FloatTensor(theta_ideal[:MLP_TRAIN_SIZE])
    X_val = torch.FloatTensor(rss[MLP_TRAIN_SIZE:MLP_TRAIN_SIZE + MLP_VAL_SIZE])
    y_val = torch.FloatTensor(theta_ideal[MLP_TRAIN_SIZE:MLP_TRAIN_SIZE + MLP_VAL_SIZE])

    trained_members = []
    all_histories = []

    mlp_max_lrs = [2e-3, 3e-3, 5e-3]

    for i in range(MLP_ENSEMBLE_SIZE):
        seed = RANDOM_SEED + i * 1000
        torch.manual_seed(seed)
        np.random.seed(seed)

        member = MLPModel().to(DEVICE)

        train_loader = DataLoader(
            TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True,
            generator=torch.Generator().manual_seed(seed),
        )
        val_loader = DataLoader(
            TensorDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False,
        )

        max_lr = mlp_max_lrs[i % len(mlp_max_lrs)]
        criterion = MSEPlusL1Loss(l1_weight=0.1)
        optimizer = torch.optim.Adam(member.parameters(), lr=MLP_LR)
        steps_per_epoch = len(train_loader)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=max_lr, epochs=MLP_EPOCHS,
            steps_per_epoch=steps_per_epoch, pct_start=0.3,
        )

        print(f"\nTraining MLP member {i + 1}/{MLP_ENSEMBLE_SIZE} (seed={seed})...")
        history = train_loop(
            member, train_loader, val_loader, criterion, optimizer,
            scheduler, MLP_EPOCHS, MLP_PATIENCE, DEVICE,
        )
        trained_members.append(member)
        all_histories.append(history)

    ensemble = EnsembleMLPModel(trained_members).to(DEVICE)

    path = os.path.join(checkpoint_dir, "mlp_best.pt")
    torch.save(ensemble.state_dict(), path)
    print(f"MLP ensemble ({MLP_ENSEMBLE_SIZE} members) saved to {path}")

    best_idx = min(range(len(all_histories)),
                   key=lambda j: min(all_histories[j]["val_loss"]))
    return ensemble, all_histories[best_idx]


def generate_cfp_training_data(mlp_model, data_dir="data"):
    """Run trained MLP on CFP dataset to produce training pairs for CFP."""
    data = np.load(os.path.join(data_dir, "cfp_data.npz"))
    rss = data["rss"]
    theta_ideal = data["theta_ideal"]

    mlp_model.eval()
    n = rss.shape[0]
    mlp_output = np.zeros((n, 900))

    with torch.no_grad():
        for start in range(0, n, BATCH_SIZE):
            end = min(start + BATCH_SIZE, n)
            batch = torch.FloatTensor(rss[start:end]).to(DEVICE)
            mlp_output[start:end] = mlp_model(batch).cpu().numpy()

    mlp_images = mlp_output.reshape(-1, 1, GRID_SIZE, GRID_SIZE)
    ideal_images = theta_ideal.reshape(-1, 1, GRID_SIZE, GRID_SIZE)

    np.savez(
        os.path.join(data_dir, "cfp_training_pairs.npz"),
        mlp_images=mlp_images, ideal_images=ideal_images,
    )
    print(f"CFP training pairs saved ({mlp_images.shape[0]} samples)")

    return mlp_images, ideal_images


def train_cfp(data_dir="data", checkpoint_dir="checkpoints"):
    """Train CFP model on MLP output / ideal SLF pairs."""
    os.makedirs(checkpoint_dir, exist_ok=True)

    torch.manual_seed(RANDOM_SEED + 100)
    np.random.seed(RANDOM_SEED + 100)

    data = np.load(os.path.join(data_dir, "cfp_training_pairs.npz"))
    mlp_images = data["mlp_images"]
    ideal_images = data["ideal_images"]

    X_train = torch.FloatTensor(mlp_images[:CFP_TRAIN_SIZE])
    y_train = torch.FloatTensor(ideal_images[:CFP_TRAIN_SIZE])
    X_val = torch.FloatTensor(mlp_images[CFP_TRAIN_SIZE:CFP_TRAIN_SIZE + CFP_VAL_SIZE])
    y_val = torch.FloatTensor(ideal_images[CFP_TRAIN_SIZE:CFP_TRAIN_SIZE + CFP_VAL_SIZE])

    train_loader = DataLoader(
        TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False
    )

    model = CFPModel().to(DEVICE)
    criterion = MSEPlusL1Loss(l1_weight=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=CFP_LR)
    steps_per_epoch = len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=3e-3, epochs=CFP_EPOCHS,
        steps_per_epoch=steps_per_epoch, pct_start=0.3,
    )

    print("Training CFP...")
    history = train_loop(
        model, train_loader, val_loader, criterion, optimizer,
        scheduler, CFP_EPOCHS, None, DEVICE,
    )

    path = os.path.join(checkpoint_dir, "cfp_best.pt")
    torch.save(model.state_dict(), path)
    print(f"CFP model saved to {path}")

    return model, history
