import torch
from torch import nn
from torch.optim import Adam
from models import get_model
from data import get_loaders


def accuracy(model, loader, device):
    model.eval()                                # Modalità evaluation: disattiva comportamenti del training (es. Dropout)
    correct = 0  # numero di predizioni corrette
    total = 0    # numero totale di esempi

    # Disattiva il calcolo dei gradienti: più veloce e meno memoria in validazione/test
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(1)                # argmax(1) prende l'indice della classe con logit massimo per ogni esempio
            correct += (pred == y).sum().item()      # Conta quante predizioni coincidono con le etichette vere
            total += y.numel()                       # Numero di elementi nel batch (equivalente a batch_size)
    return correct / total                           #ritorna accuracy 


def main():

    out='/content/drive/MyDrive/NeuralNetworksProject/checkpoints_new/mlp_aug_best.pt'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")                             # Se CUDA è disponibile usa GPU, altrimenti CPU
    out='/content/drive/MyDrive/NeuralNetworksProject/checkpoints_new/mlp_aug_best.pt'
    # Creo dataloader di train/val/test
    train_loader, val_loader, test_loader = get_loaders(batch_size=batch, aug=bool(aug))    # aug=bool(aug): se aug è 1 -> True, se 0 -> False
    model = get_model(model).to(device)                                                          # Istanzia il modello scelto e lo sposta su device
    
    opt = Adam(model.parameters(), lr=lr)                                                        # Ottimizzatore Adam per aggiornare i pesi del modello
    loss_fn = nn.CrossEntropyLoss()                                                                   # Funzione di loss per classificazione multi-classe

    best_val = -1.0

    for ep in range(epochs):
        model.train()                                                      # Modalità training: attiva ad es. dropout 
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()                                                # Azzera gradienti
            loss = loss_fn(model(x), y)                                    #il modello produce logits, la loss li confronta con le etichette
            loss.backward()                                                # Backprop: calcola i gradienti
            opt.step()                                                     # Step ottimizzatore: aggiorna i pesi
            
        #calcolo accuracy su validation e test set
        val_acc = accuracy(model, val_loader, device)
        test_acc = accuracy(model, test_loader, device)
        print(f"Epoch {ep} | val_acc={val_acc:.4f} | test_acc={test_acc:.4f}")
        
        # Se la validation accuracy migliora, salva il checkpoint "migliore"
        if val_acc > best_val:
            best_val = val_acc
            torch.save({
                "model_name": model,                        #nome modello
                "aug": bool(aug),                           #Augmentatio on/off
                "state_dict": model.state_dict(),                #pesi del modello 
                "val_acc": best_val                              #migliore validation 
            }, out)    
            print("  Salvato checkpoint BEST:", out)

    print("Fine. Best val acc:", best_val)                    # Fine training: stampa miglior risultato di validation



