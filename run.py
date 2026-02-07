import argparse
import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

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
    p.add_argument("--ckpt", type=str, default="checkpoints/cnn_light_noaug_best.pt")
    p.add_argument("--model", type=str, default="cnn_light")
    p.add_argument("--aug", type=int, default=0)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, test_loader = get_loaders(batch_size=256, aug=bool(args.aug))

    model = get_model(args.model).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["state_dict"])

    acc = accuracy(model, test_loader, device)
    print("Checkpoint:", args.ckpt)
    print("Test accuracy:", acc)

    # salva immagine demo
    x, y = next(iter(test_loader))
    x = x[:16].to(device)
    with torch.no_grad():
        preds = model(x).argmax(1).cpu().tolist()

    grid = make_grid(x.cpu(), nrow=4, padding=2)
    plt.figure(figsize=(5, 5))
    plt.axis("off")
    plt.imshow(grid.permute(1, 2, 0).squeeze(), cmap="gray")
    plt.title("Pred: " + " ".join(map(str, preds)), fontsize=9)
    plt.savefig("outputs/demo_grid.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Salvata immagine: outputs/demo_grid.png")


if __name__ == "__main__":
    main()
