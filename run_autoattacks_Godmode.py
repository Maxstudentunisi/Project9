import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import Subset, DataLoader
from autoattack import AutoAttack
import random
import os

from models import get_model

# Parameters
EPS_VALUES = [0.03, 0.3, 1.0]
N_TEST = 500  # TODO: maybe increase this later?
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Reproducibility for experiments

np.random.seed(42)


print(f"Device: {DEVICE}")
print(f"Test samples: {N_TEST}")
print(f"Epsilon values: {EPS_VALUES}")

os.makedirs("outputs", exist_ok=True)  # Create output folder

# Load MNIST test set: we pick N_TEST random images from MNIST
transform = transforms.ToTensor()
test_data = datasets.MNIST('./data', train=False, transform=transform, download=True)

idx = np.random.choice(len(test_data), N_TEST, replace=False)
subset = Subset(test_data, idx)
loader = DataLoader(subset, batch_size=N_TEST)

for x_test, y_test in loader:
    break

x_test = x_test.to(DEVICE)
y_test = y_test.to(DEVICE)

# Load models
models = {}

def load(name, arch, ckpt):
    if not os.path.exists(ckpt):
        print(f"Checkpoint {ckpt} not Found!")
        return
    m = get_model(arch)
    m.load_state_dict(torch.load(ckpt, map_location=DEVICE)["state_dict"])
    m = m.to(DEVICE)  # No mismatch device between model and input
    m.eval()
    models[name] = m

load("cnn_light_noaug", "cnn_light", "checkpoints/cnn_light_noaug_best.pt")
load("cnn_light_aug", "cnn_light", "checkpoints/cnn_light_aug_best.pt")
load("cnn_deep_noaug", "cnn_deep", "checkpoints/cnn_deep_noaug_best.pt")
load("cnn_deep_aug", "cnn_deep", "checkpoints/cnn_deep_aug_best.pt")
load("mlp_noaug", "mlp", "checkpoints/mlp_noaug.pt")
load("mlp_aug", "mlp", "checkpoints/mlp_aug.pt")

model_names = list(models.keys())
if len(models) == 0:
    raise RuntimeError("No models loaded.")  # If no checkpoints were found stop the script

# Clean accuracy
print("\nClean accuracy:\n")
for name, model in models.items():
    with torch.no_grad():
        acc = (model(x_test).argmax(1) == y_test).float().mean().item() * 100
    print(f"{name:20} {acc:6.2f}%")

# Attacks to test
attacks = {
    "APGD-CE": dict(version='custom', attacks=['apgd-ce']),
    "APGD-DLR": dict(version='custom', attacks=['apgd-dlr']),
    "FAB": dict(version='custom', attacks=['fab-t']),
    "Square": dict(version='custom', attacks=['square'])
}

# Experiments changing epsilon value
all_results_by_eps = {}  # To store transferability matrices
all_adv_by_eps = {}      # To store all adversarial examples

for eps in EPS_VALUES:
    print(f"\nTesting with EPS = {eps}\n")
    eps_folder = f"outputs/eps_{eps}"
    os.makedirs(eps_folder, exist_ok=True)

    all_results = {}
    all_adv = {}  # This will contain all adv examples for this epsilon

    for attack_name, cfg in attacks.items():
        print(f"\nTesting attack: {attack_name}\n")
        adv_examples = {}

        for mname, model in models.items():
            print(f"Generating adversarial for {mname}")
            bs = 100 if DEVICE == 'cpu' else 250  # For the sake of my cpu

            attacker = AutoAttack(
                model,
                norm='Linf',
                eps=eps,
                version=cfg['version'],
                attacks_to_run=cfg['attacks'],
                device=DEVICE,
                verbose=False
            )

            adv_examples[mname] = attacker.run_standard_evaluation(x_test, y_test, bs=bs)

        # Transferability matrix
        results = np.zeros((len(models), len(models)))
        for i, src in enumerate(model_names):
            for j, tgt in enumerate(model_names):
                with torch.no_grad():
                    acc = (models[tgt](adv_examples[src]).argmax(1) == y_test).float().mean().item() * 100
                results[i, j] = acc

        all_results[attack_name] = results
        all_adv[attack_name] = adv_examples  # Save all adv examples per attack

        # Print matrix
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

        # Heatmaps
        """
        This creates heatmap for transferability 
        The goal is to show robust accuracy for each models
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(results, cmap='RdYlGn', vmin=0, vmax=100)
        plt.colorbar(im, label="Robust accuracy (%)")

        ax.set_xticks(range(len(model_names)))
        ax.set_yticks(range(len(model_names)))
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.set_yticklabels(model_names)
        ax.set_xlabel("Tested on")
        ax.set_ylabel("Adversarial from")
        ax.set_title(f"Transferability Matrix - {attack_name} - eps={eps}")

        for i in range(len(model_names)):
            for j in range(len(model_names)):
                ax.text(j, i, f"{results[i,j]:.1f}",
                        ha='center', va='center',
                        color='white' if results[i,j] < 50 else 'black',
                        fontsize=8)

        plt.tight_layout()
        plt.savefig(f"{eps_folder}/heatmap_{attack_name}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    all_results_by_eps[eps] = all_results
    all_adv_by_eps[eps] = all_adv  # Save all adv examples for this epsilon

# Visualization of adversarial examples
"""
Show original, adversarial, and difference for each attack and model
We save these images in output folder
"""
n_show = 8  # Number of samples to visualize

for eps in EPS_VALUES:
    eps_folder = f"outputs/eps_{eps}"
    
    for attack_name in attacks:
        for model_name in model_names:
            x_adv = all_adv_by_eps[eps][attack_name][model_name]

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
            plt.savefig(f"{eps_folder}/visualization_{attack_name}_{model_name}.png", dpi=300, bbox_inches="tight")
            plt.close(fig)

# Robust vs accuracy summary
print("\nRobust accuracy summary:\n")
for eps in EPS_VALUES:
    print(f"\nEPSILON = {eps}")
    print(f"{'Model':20}", end="")
    for a in attacks:
        print(f"{a:>12}", end="")
    print()
    print("-" * (20 + 12 * len(attacks)))

    for i, name in enumerate(model_names):
        print(f"{name:20}", end="")
        for a in attacks:
            print(f"{all_results_by_eps[eps][a][i,i]:12.2f}", end="")
        print()

# Summary plot
"""
This plot shows robustness vs epsilon
We average over all attacks for each model and then we save them
"""
fig = plt.figure(figsize=(12, 6))
markers = ['o', 's', '^', 'D', 'v', '<']

for i, model_name in enumerate(model_names):
    robust_accs = []
    for eps in EPS_VALUES:
        accs = [all_results_by_eps[eps][a][i,i] for a in attacks]
        robust_accs.append(np.mean(accs))

    plt.plot(EPS_VALUES, robust_accs,
             marker=markers[i],
             label=model_name,
             linewidth=2)

plt.xlabel('Epsilon')
plt.ylabel('Robust Accuracy (%)')
plt.title('Robustness vs Epsilon - Mean over attacks')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.xscale('log')
plt.xticks(EPS_VALUES, [str(e) for e in EPS_VALUES])
plt.tight_layout()
plt.savefig("outputs/robustness_vs_epsilon.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("\nCompilation finished...")
