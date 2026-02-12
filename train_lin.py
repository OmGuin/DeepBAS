import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
from datetime import datetime
import os
import pickle

from models import BASLIN
from datasets import LinSet
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

os.makedirs('loss_curves', exist_ok=True)
os.makedirs('models', exist_ok=True)


parser = argparse.ArgumentParser(description="hyperparameters for training")
parser.add_argument('--lr',type=float, default=2e-5)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--num_epochs', type=int, default=250)
parser.add_argument('--output_bins', type=int, default=50)
parser.add_argument('--init_beta', type=float, default=0.5)
parser.add_argument('--save', action=argparse.BooleanOptionalAction, default=True)
parser.add_argument('--prog_bar', action=argparse.BooleanOptionalAction, default=True)
args = parser.parse_args()



lr = args.lr
BATCH_SIZE = args.batch_size
NUM_EPOCHS = args.num_epochs
model = BASLIN(output_dim=args.output_bins, init_beta = args.init_beta).to(device)

beta_param = []
other_params = []
for name, p in model.named_parameters():
    if not p.requires_grad:
        continue
    if name == "log_beta":
        beta_param.append(p)
    else:
        other_params.append(p)
optimizer = torch.optim.AdamW([
    {'params': other_params, 'lr':2e-5, 'weight_decay':1e-3},
    {'params': beta_param, 'lr':1e-3, 'weight_decay':0.0}
])
criterion = nn.PoissonNLLLoss(log_input=False)
dataset = LinSet(r"/content/train/pchdata", num_bins = args.output_bins)

train_ratio = 0.8
val_ratio = 0.2

val_size = int(val_ratio * len(dataset))
train_size = len(dataset) - val_size
train_set, val_set = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
best_val_loss = float('inf')
patience_counter = 0
early_stop_patience = 20

train_losses = []
val_losses = []
betas = []

for epoch in tqdm(range(NUM_EPOCHS), disable=not args.prog_bar):
    total_loss = 0.0
    val_loss = 0.0
    model.train()
    for i, (pch, tdist) in enumerate(train_loader):
        pch, tdist = pch.to(device), tdist.to(device)
        pred_dist = model(pch)
        loss = criterion(pred_dist, tdist)
        
        # if i == 200:
        #     true_pch = pch[0].cpu().numpy()
        #     pred_dist = pred_dist[0].cpu().detach().numpy()
        #     true_dist = tdist[0].cpu().numpy()    
        #     wass = wasserstein_distance(true_dist, pred_dist)
        #     fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        #     axes[0].plot(true_pch, marker='o', label='True PCH')
        #     axes[0].set_title("Photon Counting Histogram")
        #     axes[0].set_xlabel("Bin")
        #     axes[0].set_ylabel("Intensity Count")
        #     axes[0].legend()
        #     axes[0].grid(True)

        #     axes[1].plot(true_dist, marker='o', label=f'True Brightness Dist')
        #     axes[1].plot(pred_dist, marker='x', linestyle='--', label='Predicted Brightness Dist')
        #     axes[1].set_title(wass)
        #     axes[1].set_xlabel("Bin")
        #     axes[1].set_ylabel("Particle Count")
        #     axes[1].legend()
        #     axes[1].grid(True)

        #     plt.tight_layout()
        #     plt.show()


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    model.eval()
    with torch.no_grad():
        for pch, tdist in val_loader:
            pch, tdist = pch.to(device), tdist.to(device)
            pred_dist = model(pch)
            s_p = pred_dist.sum(axis=1)
            s_t = tdist.sum(axis=1)
            loss = criterion(pred_dist, tdist)
            
            val_loss += loss.item()

    current_val_loss = val_loss / len(val_loader)
    scheduler.step(current_val_loss)
    if current_val_loss < best_val_loss:
        best_val_loss = current_val_loss
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= early_stop_patience:
            print("Early stopping")
            break
    
    print(f"Epoch {epoch} | Loss: {total_loss / len(train_loader)} | Val: {val_loss / len(val_loader)} | Beta: {model.beta()}")
    
    train_losses.append(total_loss / len(train_loader))
    val_losses.append(val_loss / len(val_loader))
    betas.append(model.beta())


if args.save:
    time_now = datetime.now().strftime('%m-%d_%H-%M')
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Poisson Negative Log-Likelihood Loss vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Negative Log-Likelihood Loss')
    plt.legend()
    plt.savefig(f"loss_curves/loss_{time_now}.png", dpi=100, bbox_inches='tight')
    
    plt.clf()
    betas = [beta.item() for beta in betas]
    plt.plot(betas, label='beta')
    plt.legend()
    plt.savefig(f"loss_curves/beta_{time_now}.png", dpi=100, bbox_inches='tight')
    
    torch.save({
        "model_state_dict":model.state_dict(),
        "optim_state_dict":optimizer.state_dict()},
               f"models/bas_lin_{time_now}.pth")
    
    losses_and_beta = {"train_loss": train_losses, "val_loss": val_loss, "beta":betas}
    
    with open(f"loss_curves/bas_lin_{time_now}_losses_beta.pkl", 'wb') as f:
        pickle.dump({"train_loss": train_losses, "val_loss": val_loss, "beta":betas}, f)