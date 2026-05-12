"""Training loop for M2TR deepfake detection model."""

import torch
import torch.nn as nn
from tqdm import tqdm


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct = 0.0, 0
    for x_rgb, x_dct, y in tqdm(loader, desc='Train'):
        x_rgb, x_dct, y = x_rgb.to(device), x_dct.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x_rgb, x_dct)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x_rgb.size(0)
        correct += (out.argmax(1) == y).sum().item()
    n = len(loader.dataset)
    return total_loss / n, correct / n


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct = 0.0, 0
    with torch.no_grad():
        for x_rgb, x_dct, y in tqdm(loader, desc='Eval'):
            x_rgb, x_dct, y = x_rgb.to(device), x_dct.to(device), y.to(device)
            out = model(x_rgb, x_dct)
            loss = criterion(out, y)
            total_loss += loss.item() * x_rgb.size(0)
            correct += (out.argmax(1) == y).sum().item()
    n = len(loader.dataset)
    return total_loss / n, correct / n


def fit(model, train_loader, val_loader, epochs=20, lr=1e-4,
        device='cuda', checkpoint_path='best_model.pt'):
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    best_val_loss = float('inf')
    for epoch in range(1, epochs + 1):
        print(f'\nEpoch {epoch}/{epochs}')
        tl, ta = train_epoch(model, train_loader, criterion, optimizer, device)
        vl, va = evaluate(model, val_loader, criterion, device)
        print(f'  Train loss={tl:.4f} acc={ta:.4f}')
        print(f'  Val   loss={vl:.4f} acc={va:.4f}')
        if vl < best_val_loss:
            best_val_loss = vl
            torch.save(model.state_dict(), checkpoint_path)
            print(f'  Saved checkpoint → {checkpoint_path}')
