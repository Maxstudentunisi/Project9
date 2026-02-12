from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import torch


def get_loaders(batch_size=128, aug=False):
    if aug:
        train_tf = transforms.Compose([
            transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
            transforms.ToTensor(),
        ])
    else:
        train_tf = transforms.ToTensor()

    test_tf = transforms.ToTensor()

    train_ds = datasets.MNIST("./data", train=True, download=True, transform=train_tf)
    test_ds = datasets.MNIST("./data", train=False, download=True, transform=test_tf)

    # piccola validation split
    n_val = 6000
    n_train = len(train_ds) - n_val
    train_ds, val_ds = random_split(train_ds, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader
