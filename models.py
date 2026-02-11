import torch.nn as nn
import torch.nn.functional as F
import torch


def CNNLight():
    return nn.Sequential(
        nn.Conv2d(1, 32, 5, padding=2),    #layer Convoluzionale, 1 img in ingresso, 32 feature map 5x5, 2pixel di bordo extra
        nn.ReLU(),                         #layer che applica funzioneRelu elemento per elemento    
        nn.MaxPool2d(2, 2),                #prende l'output lo divide in sottomatrici 2x2 e prende l'elemento con valore più alto per ogni sotto matrice, si passa da 28x28 a 14x14 

        nn.Conv2d(32, 64, 5, padding=2),   #layer Convoluzionale, 32 img in ingresso, 64 feature map 5x5, 2pixel di bordo extra
        nn.ReLU(),
        nn.MaxPool2d(2, 2),                #qua si passa da 14x4 a 7x7

        nn.Flatten(),                     # passa da tensore multidimensionale a vettore 1D di dimensioni (batchsize, 7x7x64)
        nn.Linear(64 * 7 * 7, 128),
        nn.ReLU(),
        nn.Dropout(0.5),                  #spegne a caso il 50% dei neuroni 
        nn.Linear(128, 10)                #layer che passa da 128 ingressi a 10 uscite, una per classe
    )
def CNNDeep():
    return nn.Sequential(
        nn.Conv2d(1, 32, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 32, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),

        nn.Conv2d(32, 64, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(64, 64, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),

        nn.Flatten(),                 # (B, 64*7*7)
        nn.Linear(64 * 7 * 7, 128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, 10)
    )
def MLP():
    return nn.Sequential(
        nn.Flatten(),                 # (B, 28*28)
        nn.Linear(28 * 28, 256),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 10)
    )
    
    def get_model(name: str):
    name = name.lower()
    if name == "cnn_light":
        return CNNLight()
    if name == "cnn_deep":
        return CNNDeep()
    if name == "mlp":
        return MLP()
    raise ValueError("Modello non valido: usa cnn_light / cnn_deep / mlp")
