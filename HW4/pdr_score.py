from datasets import load_dataset
import torch
import torch.nn as nn
import numpy as np

# --- Preparation: Make sure these variables and class definitions exist ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
colormap = list(load_dataset("lca0503/ml2025-hw4-colormap")["train"]["color"])

# Copy the PokemonClassifier class definition here to ensure this cell can run independently
class PokemonClassifier(nn.Module):
    def __init__(self):
        super(PokemonClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(32 * 5 * 5, 128), nn.ReLU(),
            nn.Linear(128, 1)
        )
    def forward(self, x):
        return self.network(x)

# --- Main scoring script ---

print("===== Start using local classifier to evaluate PDR score =====")

# 1. Load the trained classifier model
classifier_path = "/workspace/pokemon_classifier.pth"
local_pdr_classifier = PokemonClassifier().to(device)
try:
    local_pdr_classifier.load_state_dict(torch.load(classifier_path, map_location=device))


    local_pdr_classifier.eval()
    print(f"Successfully loaded classifier model: {classifier_path}")
except FileNotFoundError:
    print(f"Error: Unable to find classifier model file '{classifier_path}'. Please run the classifier training code first.")
    # If the file is not found, stop execution
    raise

# 2. Read the generated pixel data
results_path = "reconstructed_results.txt"
try:
    with open(results_path, 'r') as f:
        reconstructed_pixels = [list(map(int, line.strip().split())) for line in f.readlines()]
    print(f"Successfully read generated results: {results_path} (Total {len(reconstructed_pixels)} images)")
except FileNotFoundError:
    print(f"Error: Unable to find generated results file '{results_path}'. Please run the main model inference code first.")
    raise

# 3. Convert pixel data to image tensors
reconstructed_images = []
for p_seq in reconstructed_pixels:
    # Ensure the length is 400
    while len(p_seq) < 400:
        p_seq.append(0)
    p_seq = p_seq[:400]
    
    img_array = np.array([colormap[i] for i in p_seq], dtype=np.uint8).reshape(20, 20, 3)
    img_tensor = torch.tensor(img_array, dtype=torch.float32).permute(2, 0, 1) / 255.0
    reconstructed_images.append(img_tensor)

# 4. Run inference and calculate PDR
total_preds = 0
num_samples = len(reconstructed_images)

with torch.no_grad():
    # For efficiency, you can process in batches, but processing 80 images at once is also fine
    images_tensor = torch.stack(reconstructed_images).to(device)
    logits = local_pdr_classifier(images_tensor)
    preds = torch.sigmoid(logits) > 0.5
    
    # Count the number of images predicted as Pokémon (True)
    pokemon_count = preds.sum().item()

# 5. Calculate the final PDR score
local_pdr_score = pokemon_count / num_samples if num_samples > 0 else 0

print("\n===== PDR scoring completed =====")
print(f"Local PDR (Accuracy) score: {local_pdr_score:.4f}")
print(f"({pokemon_count} / {num_samples} images predicted as Pokémon)")