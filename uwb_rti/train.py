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
)
from .models.mlp_model import MLPModel
from .models.cfp_model import CFPModel


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

        if scheduler is not None:
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
    """Train MLP model on MLP dataset."""
    os.makedirs(checkpoint_dir, exist_ok=True)

    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    data = np.load(os.path.join(data_dir, "mlp_data.npz"))
    rss = data["rss"]
    theta_ideal = data["theta_ideal"]

    X_train = torch.FloatTensor(rss[:MLP_TRAIN_SIZE])
    y_train = torch.FloatTensor(theta_ideal[:MLP_TRAIN_SIZE])
    X_val = torch.FloatTensor(rss[MLP_TRAIN_SIZE:MLP_TRAIN_SIZE + MLP_VAL_SIZE])
    y_val = torch.FloatTensor(theta_ideal[MLP_TRAIN_SIZE:MLP_TRAIN_SIZE + MLP_VAL_SIZE])

    train_loader = DataLoader(
        TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False
    )

    model = MLPModel().to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=MLP_LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)

    print("Training MLP...")
    history = train_loop(
        model, train_loader, val_loader, criterion, optimizer,
        scheduler, MLP_EPOCHS, MLP_PATIENCE, DEVICE,
    )

    path = os.path.join(checkpoint_dir, "mlp_best.pt")
    torch.save(model.state_dict(), path)
    print(f"MLP model saved to {path}")

    return model, history


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
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=CFP_LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=15)

    print("Training CFP...")
    history = train_loop(
        model, train_loader, val_loader, criterion, optimizer,
        scheduler, CFP_EPOCHS, 30, DEVICE,
    )

    path = os.path.join(checkpoint_dir, "cfp_best.pt")
    torch.save(model.state_dict(), path)
    print(f"CFP model saved to {path}")

    return model, history
