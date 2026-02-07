!pip install git+https://github.com/fra31/auto-attack
# autoattack_experiment.py
import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from autoattacks import AutoAttack
from models import CNNLight, CNNDeep, MLP
import matplotlib.pyplot as plt
import os

# ============================================
# CONFIGURAZIONE
# ============================================
device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_samples = 1000  # Numero di esempi da testare
epsilon = 0.3     # Budget di perturbazione
batch_size = 250  # Batch size per AutoAttack

print(f"Device: {device}")
print(f"Test samples: {n_samples}")
print(f"Epsilon: {epsilon}\n")

# ============================================
# CARICAMENTO DATASET
# ============================================
transform = transforms.Compose([transforms.ToTensor()])
test_dataset = datasets.MNIST('./data', train=False, transform=transform, download=True)

# Subset casuale per velocizzare i test
indices = np.random.choice(len(test_dataset), n_samples, replace=False)
test_subset = Subset(test_dataset, indices)
test_loader = DataLoader(test_subset, batch_size=n_samples, shuffle=False)

x_test, y_test = next(iter(test_loader))
x_test = x_test.to(device)
y_test = y_test.to(device)

print(f"Test set shape: {x_test.shape}")
print(f"Labels shape: {y_test.shape}\n")

# ============================================
# CARICAMENTO MODELLI
# ============================================
def load_model(model_class, checkpoint_path):
    model = model_class().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model

models = {
    'CNNLight': load_model(CNNLight, 'checkpoints/cnn_light_best.pth'),
    'CNNDeep': load_model(CNNDeep, 'checkpoints/cnn_deep_best.pth'),
    'MLP': load_model(MLP, 'checkpoints/mlp_best.pth')
}

print(f"Loaded {len(models)} models\n")

# ============================================
# VALUTAZIONE ACCURACY
# ============================================
def evaluate_accuracy(model, x, y):
    """Calcola accuracy su un batch di dati"""
    with torch.no_grad():
        outputs = model(x)
        predictions = outputs.argmax(dim=1)
        accuracy = (predictions == y).float().mean().item() * 100
    return accuracy

# Accuracy sui dati puliti (clean accuracy)
print("="*70)
print("CLEAN ACCURACY (senza attacchi)")
print("="*70)
for name, model in models.items():
    clean_acc = evaluate_accuracy(model, x_test, y_test)
    print(f"{name:12}: {clean_acc:.2f}%")
print()

# ============================================
# GENERAZIONE ADVERSARIAL EXAMPLES CON AUTOATTACK
# ============================================
os.makedirs('adversarial_examples', exist_ok=True)
adversarial_examples = {}

print("="*70)
print("AUTOATTACK - GENERAZIONE ADVERSARIAL EXAMPLES")
print("="*70)

for model_name, model in models.items():
    print(f"\nGenerazione AutoAttack su {model_name}...")

    # Configura AutoAttack
    adversary = AutoAttack(
        model,
        norm='Linf',
        eps=epsilon,
        version='standard',  # Usa APGD-CE, APGD-DLR, FAB, Square Attack
        device=device,
        verbose=False
    )

    # Genera adversarial examples
    x_adv = adversary.run_standard_evaluation(x_test, y_test, bs=batch_size)
    adversarial_examples[model_name] = x_adv.cpu()

    # Salva su disco
    torch.save(x_adv.cpu(), f'adversarial_examples/autoattack_{model_name.lower()}.pt')

    # Valuta robust accuracy
    robust_acc = evaluate_accuracy(model, x_adv, y_test)
    print(f"Robust accuracy su {model_name}: {robust_acc:.2f}%")

print()

# ============================================
# TRANSFERABILITY TEST
# ============================================
print("="*70)
print("TRANSFERABILITY ANALYSIS")
print("="*70)
print("Test: adversarial examples generati su un modello vengono")
print("      testati su tutti gli altri modelli\n")

model_names = list(models.keys())
transfer_matrix = np.zeros((len(models), len(models)))

for i, attack_source in enumerate(model_names):
    x_adv = adversarial_examples[attack_source].to(device)
    print(f"Adversarial examples generati su {attack_source}:")

    for j, test_target in enumerate(model_names):
        model = models[test_target]
        accuracy = evaluate_accuracy(model, x_adv, y_test)
        transfer_matrix[i, j] = accuracy
        print(f"  Testato su {test_target:12}: {accuracy:6.2f}%")
    print()

# Salva matrice di transferability
np.save('adversarial_examples/transferability_matrix.npy', transfer_matrix)

# Stampa matrice formattata
print("="*70)
print("TRANSFERABILITY MATRIX")
print("Righe: adversarial generato su")
print("Colonne: testato su")
print("Valori: accuracy % (più basso = attacco più efficace)")
print("="*70)
header = "             | " + " | ".join(f"{n:10}" for n in model_names)
print(header)
print("-" * len(header))
for i, name in enumerate(model_names):
    row = " | ".join(f"{transfer_matrix[i, j]:>10.1f}" for j in range(len(model_names)))
    print(f"{name:12} | {row}")
print()

# ============================================
# VISUALIZZAZIONE ADVERSARIAL EXAMPLES
# ============================================
def save_adversarial_visualization(original, adversarial, model_name, n_images=10):
    """Salva visualizzazione confronto originale vs adversarial"""
    fig, axes = plt.subplots(3, n_images, figsize=(20, 6))

    for i in range(n_images):
        # Immagine originale
        axes[0, i].imshow(original[i].squeeze().cpu(), cmap='gray', vmin=0, vmax=1)
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_ylabel('Originale', fontsize=12)

        # Immagine adversarial
        axes[1, i].imshow(adversarial[i].squeeze().cpu(), cmap='gray', vmin=0, vmax=1)
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_ylabel('Adversarial', fontsize=12)

        # Differenza (amplificata per visibilità)
        diff = (adversarial[i] - original[i]).squeeze().cpu().numpy()
        im = axes[2, i].imshow(diff * 10, cmap='seismic', vmin=-1, vmax=1)
        axes[2, i].axis('off')
        if i == 0:
            axes[2, i].set_ylabel('Diff x10', fontsize=12)

    plt.suptitle(f'AutoAttack su {model_name} (epsilon={epsilon})', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'adversarial_examples/visualization_{model_name.lower()}.png',
                dpi=150, bbox_inches='tight')
    plt.close()

print("="*70)
print("SALVATAGGIO VISUALIZZAZIONI")
print("="*70)

for model_name in model_names:
    x_adv = adversarial_examples[model_name]
    save_adversarial_visualization(x_test[:10], x_adv[:10], model_name, n_images=10)
    print(f"Salvata visualizzazione: visualization_{model_name.lower()}.png")

# ============================================
# HEATMAP TRANSFERABILITY
# ============================================
fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(transfer_matrix, cmap='RdYlGn', vmin=0, vmax=100, aspect='auto')

# Colorbar
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Accuracy (%)', rotation=270, labelpad=20, fontsize=12)

# Etichette assi
ax.set_xticks(range(len(model_names)))
ax.set_yticks(range(len(model_names)))
ax.set_xticklabels(model_names, rotation=45, ha='right')
ax.set_yticklabels(model_names)
ax.set_xlabel('Testato su', fontsize=12)
ax.set_ylabel('Adversarial generato su', fontsize=12)
ax.set_title('Transferability Matrix - AutoAttack\n(valori bassi = attacco trasferibile)',
             fontsize=14, pad=20)

# Aggiungi valori numerici nelle celle
for i in range(len(model_names)):
    for j in range(len(model_names)):
        text_color = 'white' if transfer_matrix[i, j] < 50 else 'black'
        text = ax.text(j, i, f'{transfer_matrix[i, j]:.1f}',
                      ha="center", va="center", color=text_color, fontsize=11)

plt.tight_layout()
plt.savefig('adversarial_examples/transferability_heatmap.png',
            dpi=150, bbox_inches='tight')
print(f"Salvata heatmap: transferability_heatmap.png")

print("\n" + "="*70)
print("ESPERIMENTO COMPLETATO")
print("="*70)
print(f"Risultati salvati in: adversarial_examples/")
print(f"  - adversarial examples (.pt files)")
print(f"  - visualizzazioni (visualization_*.png)")
print(f"  - transferability matrix (.npy)")
print(f"  - transferability heatmap (.png)")
