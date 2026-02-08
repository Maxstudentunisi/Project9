import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from autoattack import AutoAttack
import matplotlib.pyplot as plt
import os

from models import get_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# parametri
eps = 0.3
n_test = 1000

# carico MNIST test
transform = transforms.Compose([transforms.ToTensor()])
test_data = datasets.MNIST('./data', train=False, transform=transform, download=True)
idx = np.random.choice(len(test_data), n_test, replace=False)
subset = Subset(test_data, idx)
loader = DataLoader(subset, batch_size=n_test)
x_test, y_test = next(iter(loader))
x_test = x_test.to(device)
y_test = y_test.to(device)

# carico i 6 modelli
models = {}

# cnn_light noaug
m = get_model("cnn_light").to(device)
m.load_state_dict(torch.load("checkpoints/cnn_light_noaug_best.pt", map_location=device)["state_dict"])
m.eval()
models["cnn_light_noaug"] = m

# cnn_light aug
m = get_model("cnn_light").to(device)
m.load_state_dict(torch.load("checkpoints/cnn_light_aug_best.pt", map_location=device)["state_dict"])
m.eval()
models["cnn_light_aug"] = m

# cnn_deep noaug
m = get_model("cnn_deep").to(device)
m.load_state_dict(torch.load("checkpoints/cnn_deep_noaug_best.pt", map_location=device)["state_dict"])
m.eval()
models["cnn_deep_noaug"] = m

# cnn_deep aug
m = get_model("cnn_deep").to(device)
m.load_state_dict(torch.load("checkpoints/cnn_deep_aug_best.pt", map_location=device)["state_dict"])
m.eval()
models["cnn_deep_aug"] = m

# mlp noaug
m = get_model("mlp").to(device)
m.load_state_dict(torch.load("checkpoints/mlp_noaug.pt", map_location=device)["state_dict"])
m.eval()
models["mlp_noaug"] = m

# mlp aug
m = get_model("mlp").to(device)
m.load_state_dict(torch.load("checkpoints/mlp_aug.pt", map_location=device)["state_dict"])
m.eval()
models["mlp_aug"] = m

print("Modelli caricati:", list(models.keys()))

# clean accuracy
print("\nClean Accuracy:")
for name, model in models.items():
    with torch.no_grad():
        pred = model(x_test).argmax(1)
        acc = (pred == y_test).float().mean().item() * 100
    print(f"{name}: {acc:.2f}%")

# genero adversarial examples
os.makedirs('adversarial_examples', exist_ok=True)
adv_examples = {}

print("\nGenerating adversarial examples:")
for name, model in models.items():
    print(f"  {name}...", end=" ")
    attacker = AutoAttack(model, norm='Linf', eps=eps, version='standard', device=device)
    x_adv = attacker.run_standard_evaluation(x_test, y_test, bs=250)
    adv_examples[name] = x_adv
    
    # robust accuracy
    with torch.no_grad():
        pred = model(x_adv).argmax(1)
        robust_acc = (pred == y_test).float().mean().item() * 100
    print(f"robust acc: {robust_acc:.2f}%")

# transferability test
print("\nTransferability matrix:")
model_names = list(models.keys())
results = np.zeros((len(models), len(models)))

for i, adv_source in enumerate(model_names):
    for j, test_target in enumerate(model_names):
        x_adv = adv_examples[adv_source]
        with torch.no_grad():
            pred = models[test_target](x_adv).argmax(1)
            acc = (pred == y_test).float().mean().item() * 100
        results[i, j] = acc

# stampo matrice
print("\n" + " "*20, end="")
for name in model_names:
    print(f"{name:18}", end=" ")
print()
for i, name in enumerate(model_names):
    print(f"{name:18}", end=" ")
    for j in range(len(model_names)):
        print(f"{results[i,j]:18.2f}", end=" ")
    print()

# salvo matrice
np.save('adversarial_examples/transfer_matrix.npy', results)

# visualizzazione
n_show = 10
for name in model_names:
    fig, axes = plt.subplots(3, n_show, figsize=(15, 4))
    x_adv = adv_examples[name]
    
    for i in range(n_show):
        # originale
        axes[0, i].imshow(x_test[i].cpu().squeeze(), cmap='gray')
        axes[0, i].axis('off')
        
        # adversarial
        axes[1, i].imshow(x_adv[i].cpu().squeeze(), cmap='gray')
        axes[1, i].axis('off')
        
        # differenza
        diff = (x_adv[i] - x_test[i]).cpu().squeeze().numpy()
        axes[2, i].imshow(diff * 10, cmap='seismic', vmin=-1, vmax=1)
        axes[2, i].axis('off')
    
    axes[0, 0].set_ylabel('Original')
    axes[1, 0].set_ylabel('Adversarial')
    axes[2, 0].set_ylabel('Diff x10')
    
    plt.suptitle(f'{name} - eps={eps}')
    plt.tight_layout()
    plt.savefig(f'adversarial_examples/{name}_vis.png', dpi=120)
    plt.close()

# heatmap
fig, ax = plt.subplots(figsize=(12, 10))
im = ax.imshow(results, cmap='RdYlGn', vmin=0, vmax=100)
plt.colorbar(im)
ax.set_xticks(range(len(model_names)))
ax.set_yticks(range(len(model_names)))
ax.set_xticklabels(model_names, rotation=45, ha='right')
ax.set_yticklabels(model_names)
ax.set_xlabel('Tested on')
ax.set_ylabel('Adversarial from')
ax.set_title('Transferability')

for i in range(len(model_names)):
    for j in range(len(model_names)):
        color = 'white' if results[i, j] < 50 else 'black'
        ax.text(j, i, f'{results[i,j]:.1f}', ha="center", va="center", color=color, fontsize=8)

plt.tight_layout()
plt.savefig('adversarial_examples/heatmap.png', dpi=120)
plt.close()

print("\nDone! Results in adversarial_examples/")
