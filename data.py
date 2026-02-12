from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import torch

def get_loaders(batch_size=128, aug=False):
     """
    Crea e restituisce i DataLoader per MNIST: train, validation e test.

    Args:
        batch_size (int): numero di immagini per batch.
        aug (bool): se True applica data augmentation solamente per il training set.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    # Se aug=True, viene applicata una augmentation per rendere il modello più robusto
    if aug:
        train_tf = transforms.Compose([
            transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),        #degrees ruota l'immagine di +10 o -10 gradi, translate la trasla al max del 10% lungo x e y
            transforms.ToTensor(),                                            #l'immagine viene trasformata in un tensore e viene aggiunto il canale (1,28,28)
        ])
    else:
        train_tf = transforms.ToTensor()                                      #solo conversione in tensore

    test_tf = transforms.ToTensor()

    train_ds = datasets.MNIST("./data", train=True, download=True, transform=train_tf)            #se train=True prende 60000 immagini
    test_ds = datasets.MNIST("./data", train=False, download=True, transform=test_tf)             #se train=False prende 10000 immagini, quelle di test    

    # piccola validation split
    n_val = 6000
    n_train = len(train_ds) - n_val
    train_ds, val_ds = random_split(train_ds, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(42))                # random_split divide il dataset in due subset (train e val)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)                    # shuffle=True solo per training: mescola i dati ad ogni epoca
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader
