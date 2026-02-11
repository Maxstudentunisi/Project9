import torch.nn as nn
import torch.nn.functional as F
import torch

"""
CNN 'leggera' per immagini 28x28 in scala di grigi (1 canale), es. MNIST.
 Architettura: 2 blocchi Conv+ReLU+MaxPool + 2 layer lineari.
"""
def CNNLight():  
    """
      # Convoluzione:
        - canali->1  (immagine grayscale)
        - output channels->32 (32 feature map)
        - kernel size->5 (filtro 5x5)
        - padding->2 mantiene dimensione spaziale (28x28 -> 28x28) a
        """
    return nn.Sequential(
        nn.Conv2d(1, 32, 5, padding=2),    
        nn.ReLU(),                             # Attivazione non lineare: applica ReLU elemento per elemento
        nn.MaxPool2d(2, 2),                    #prende l'output lo divide in sottomatrici 2x2 e prende l'elemento con valore più alto per ogni sotto matrice, si passa da 28x28 a 14x14 

        nn.Conv2d(32, 64, 5, padding=2),       # Seconda convoluzione:  32 canali in input -> 64 canali in output
        nn.ReLU(),
        nn.MaxPool2d(2, 2),                    # dimezza altezza e larghezza 14x14 -> 7x7

        nn.Flatten(),                          # Flatten: trasforma (batch, 64, 7, 7) in (batch, 64*7*7)
        nn.Linear(64 * 7 * 7, 128),
        nn.ReLU(),
        nn.Dropout(0.5),                       #“spegne” casualmente il 50% dei neuroni attivati durante il training
        nn.Linear(128, 10)                
    )
def CNNDeep():           
     """
    CNN più profonda:
    - 4 layer convoluzionali (due blocchi conv+conv+pool) + 2 lineari.
    - Kernel 3x3 con padding=1 mantiene la dimensione spaziale all’interno del blocco.
    """
    return nn.Sequential(
        nn.Conv2d(1, 32, 3, padding=1),      # #layer Convoluzionale, 1 img in ingresso, 32 feature map, kernel size 3x3, 2pixel di bordo extra
        nn.ReLU(),
        nn.Conv2d(32, 32, 3, padding=1),      #layer Convoluzionale, 32 img in ingresso, 32 feature map, kernel size 3x3, 2pixel di bordo extra
        nn.ReLU(),
        nn.MaxPool2d(2, 2),                   #si passa da img 28x28 a 14x14

        nn.Conv2d(32, 64, 3, padding=1),     
        nn.ReLU(),
        nn.Conv2d(64, 64, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),

        nn.Flatten(),                         # (batch, 64, 7, 7) -> (batch, 64*7*7)
        nn.Linear(64 * 7 * 7, 128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, 10)
    )

def MLP():
    """
    MLP (rete fully-connected) per immagini 28x28:
    Flatten -> 2 layer densi da 256 input -> output 10 classi.
    """
    return nn.Sequential(
        nn.Flatten(),                 #(Batchsize, 28*28)
        nn.Linear(28 * 28, 256),      #layer fully connected che combina tutte le feature in input per produrre quelle in output
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 10)
    )
    
    def get_model(name: str):
    """
    ritorna il modello richiesto in base al nome.
    valori validi: 'cnn_light', 'cnn_deep', 'mlp'
    """
    name = name.lower()
    if name == "cnn_light":
        return CNNLight()
    if name == "cnn_deep":
        return CNNDeep()
    if name == "mlp":
        return MLP()
        """
        se il nome non è tra quelli previsti da errore
        """
    raise ValueError("Modello non valido: usa cnn_light / cnn_deep / mlp")
