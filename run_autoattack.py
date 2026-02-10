import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from autoattack import AutoAttack

from models import get_model

# PARAMETERS
eps = 0.3
n_test = 1000
device = 'cpu'

# LOAD MNIST TEST SET
transform = transforms.ToTensor()
test_data = datasets.MNIST('./data', train=False, transform=transform, download=True)

idx = np.random.choice(len(test_data), n_test, replace=False)
x_test = torch.stack([test_data[i][0] for i in idx]).to(device)
y_test = torch.tensor([test_data[i][1] for i in idx]).to(device)

# LOAD MODELS
models = {}

def load(name, arch, ckpt):
    m = get_model(arch)
    m.load_state_dict(torch.load(ckpt, map_location=device)["state_dict"])
    m.eval()
    models[name] = m

load("cnn_light_noaug", "cnn_light", "checkpoints/cnn_light_noaug_best.pt")
load("cnn_light_aug", "cnn_light", "checkpoints/cnn_light_aug_best.pt")
load("cnn_deep_noaug", "cnn_deep", "checkpoints/cnn_deep_noaug_best.pt")
load("cnn_deep_aug", "cnn_deep", "checkpoints/cnn_deep_aug_best.pt")
load("mlp_noaug", "mlp", "checkpoints/mlp_noaug.pt")
load("mlp_aug", "mlp", "checkpoints/mlp_aug.pt")

model_names = list(models.keys())

# CLEAN ACCURACY
print("\nCLEAN ACCURACY\n")

for name, model in models.items():
    with torch.no_grad():
        acc = (model(x_test).argmax(1) == y_test).float().mean().item() * 100
    print(f"{name:20} {acc:6.2f}%")

# ATTACKS TO TEST
attacks = {
    "APGD-CE": dict(version='custom', attacks=['apgd-ce']),
    "Square":  dict(version='custom', attacks=['square'])
}

all_results = {}
all_adv = {}

# RUN ATTACKS
for attack_name, cfg in attacks.items():
    print(f"\nTESTING ATTACK: {attack_name}\n")

    adv_examples = {}

    for mname, model in models.items():
        print(f"Generating adversarial for {mname}")
        attacker = AutoAttack(
            model,
            norm='Linf',
            eps=eps,
            version=cfg['version'],
            attacks_to_run=cfg['attacks'],
            device=device,
            verbose=False
        )
        adv_examples[mname] = attacker.run_standard_evaluation(x_test, y_test, bs=250)

    # TRANSFERABILITY MATRIX
    results = np.zeros((len(models), len(models)))

    for i, src in enumerate(model_names):
        for j, tgt in enumerate(model_names):
            with torch.no_grad():
                acc = (models[tgt](adv_examples[src]).argmax(1) == y_test).float().mean().item() * 100
            results[i, j] = acc

    all_results[attack_name] = results
    all_adv[attack_name] = adv_examples

    # PRINT MATRIX
    print("\nTransferability matrix:\n")
    print(f"{'':18}", end="")
    for name in model_names:
        print(f"{name[:11]:>13}", end="")
    print("\n" + "-" * (18 + 12 * len(model_names)))

    for i, name in enumerate(model_names):
        print(f"{name:18}", end="")
        for j in range(len(model_names)):
            print(f"{results[i,j]:12.2f}", end="")
        print()

# ROBUST ACCURACY
print("\nROBUST ACCURACY\n")
print(f"{'Model':20}", end="")
for a in attacks:
    print(f"{a:>12}", end="")
print()

for i, name in enumerate(model_names):
    print(f"{name:20}", end="")
    for a in attacks:
        print(f"{all_results[a][i,i]:12.2f}", end="")
    print()

# VISUALIZATION
n_show = 8

for attack_name in attacks:
    for model_name in model_names:
        x_adv = all_adv[attack_name][model_name]

        fig, axes = plt.subplots(3, n_show, figsize=(14, 4))
        for i in range(n_show):
            axes[0,i].imshow(x_test[i].cpu().squeeze(), cmap='gray')
            axes[1,i].imshow(x_adv[i].cpu().squeeze(), cmap='gray')
            diff = (x_adv[i] - x_test[i]).cpu().squeeze()
            axes[2,i].imshow(diff * 10, cmap='seismic', vmin=-1, vmax=1)
            for r in range(3):
                axes[r,i].axis('off')

        axes[0,0].set_ylabel("Original")
        axes[1,0].set_ylabel("Adversarial")
        axes[2,0].set_ylabel("Diff x10")

        plt.suptitle(f"{model_name} - {attack_name} - eps={eps}")
        plt.tight_layout()
        plt.show()

# HEATMAPS
for attack_name, results in all_results.items():
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(results, cmap='RdYlGn', vmin=0, vmax=100)
    plt.colorbar(im, label="Robust accuracy (%)")

    ax.set_xticks(range(len(model_names)))
    ax.set_yticks(range(len(model_names)))
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.set_yticklabels(model_names)
    ax.set_xlabel("Tested on")
    ax.set_ylabel("Adversarial from")
    ax.set_title(f"Transferability Matrix - {attack_name}")

    for i in range(len(model_names)):
        for j in range(len(model_names)):
            ax.text(j, i, f"{results[i,j]:.1f}",
                    ha='center', va='center',
                    color='white' if results[i,j] < 50 else 'black',
                    fontsize=8)

    plt.tight_layout()
    plt.show()

print("\nDONE...")

