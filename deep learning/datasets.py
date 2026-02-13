import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
import torch.nn.functional as F
import os
import json
import numpy as np

class LinSet(Dataset):
    def __init__(self, root, num_bins):
        self.root = root
        self.files = sorted([
            f for f in os.listdir(root)
            if os.path.isfile(os.path.join(root, f))
            and f.endswith(".json")
            and "metadata" not in f
        ])
        self.bins = num_bins
        
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        with open(os.path.join(self.root, self.files[idx]), "r") as f:
            sim_data = json.load(f)
            pch = np.array(sim_data['Data']['pch_bins'], dtype=np.float32)
            amps = np.array(sim_data['Data']['raw_particle_amps'], dtype=np.float32)
            pch /= pch.sum()
            
            lin_edges = np.linspace(0, 8000, self.bins+1)
            lin_bins, _ = np.histogram(amps, bins=lin_edges)
        return torch.tensor(pch, dtype=torch.float32), torch.tensor(lin_bins, dtype=torch.float32)

class LogSet(Dataset):
    def __init__(self, root, num_bins):
        self.root = root
        self.files = sorted([
            f for f in os.listdir(root)
            if os.path.isfile(os.path.join(root, f))
            and f.endswith(".json")
            and "metadata" not in f
        ])
        self.bins = num_bins

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        with open(os.path.join(self.root, self.files[idx]), "r") as f:
            sim_data = json.load(f)
            pch = np.array(sim_data['Data']['pch_bins'], dtype=np.float32)
            amps = np.array(sim_data['Data']['raw_particle_amps'], dtype=np.float32)
            pch /= pch.sum()
            
            log_edges = np.logspace(np.log10(50), np.log10(8000), self.bins+1)
            log_bins, _ = np.histogram(amps.astype(float), bins=log_edges)
            
        return torch.tensor(pch, dtype=torch.float32), torch.tensor(log_bins, dtype=torch.float32)