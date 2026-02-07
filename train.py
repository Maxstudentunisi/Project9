import argparse
import torch
from torch import nn
from torch.optim import Adam

from models import get_model
from data import get_loaders


def accuracy(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            total += y.numel()
    return correct / total


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="cnn_light")
    p.add_argument("--aug", type=int, default=0)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--out", type=str, default="checkpoints/model_best.pt")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, test_loader = get_loaders(batch_size=args.batch, aug=bool(args.aug))
    model = get_model(args.model).to(device)

    opt = Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    best_val = -1.0

    for ep in range(1, args.epochs + 1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()

        val_acc = accuracy(model, val_loader, device)
        test_acc = accuracy(model, test_loader, device)
        print(f"Epoch {ep} | val_acc={val_acc:.4f} | test_acc={test_acc:.4f}")

        if val_acc > best_val:
            best_val = val_acc
            torch.save({
                "model_name": args.model,
                "aug": bool(args.aug),
                "state_dict": model.state_dict(),
                "val_acc": best_val
            }, args.out)
            print("  Salvato checkpoint BEST:", args.out)

    print("Fine. Best val acc:", best_val)


if __name__ == "__main__":
    main()
