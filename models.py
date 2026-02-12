import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        if in_channels != out_channels:
            self.projection = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.projection = None
        self.gelu = nn.GELU()

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.gelu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.projection:
            identity = self.projection(identity)
        out += identity
        out = self.gelu(out)
        return out

class BASLIN(nn.Module):
    def __init__(self, input_length=24, output_dim=100, init_beta=3):
        super().__init__()
        self.resblock1 = ResidualBlock(1, 32)
        self.dropout1 = nn.Dropout(0.15)
        self.resblock2 = ResidualBlock(32, 64)
        self.dropout2 = nn.Dropout(0.15)
        self.resblock3 = ResidualBlock(64, 128)
        self.dropout3 = nn.Dropout(0.25)
        # self.resblock4 = ResidualBlock(128, 128)
        # self.dropout4 = nn.Dropout(0.25)

        # self.head_conv = nn.Conv1d(64, 1, kernel_size=1)
        # self.pool = nn.AdaptiveAvgPool1d(output_dim)

        self.log_beta = nn.Parameter(torch.log(torch.tensor(init_beta, dtype=torch.float32)))
        

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * input_length, 32 * input_length)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32 * input_length, output_dim)
        self.dropout = nn.Dropout(0.25)
        # self.softplus = nn.Softplus(beta=0.5)
        # self.softmax = nn.Softmax(dim = 1)
    def beta(self):
        return torch.exp(self.log_beta).clamp_min(1e-4)
    
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.dropout1(self.resblock1(x))
        x = self.dropout2(self.resblock2(x))
        x = self.dropout3(self.resblock3(x))

        x = self.flatten(x)
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        
        
        beta = self.beta()
        x = torch.nn.functional.softplus(x * beta)/beta
        return x


class BASLOG(nn.Module):
    def __init__(self, input_length=24, output_dim=50, init_beta=0.5):
        super().__init__()
        self.resblock1 = ResidualBlock(1, 32)
        self.dropout1 = nn.Dropout(0.15)
        self.resblock2 = ResidualBlock(32, 64)
        self.dropout2 = nn.Dropout(0.15)
        # self.resblock3 = ResidualBlock(64, 128)
        # self.dropout3 = nn.Dropout(0.25)
        # self.resblock4 = ResidualBlock(128, 128)
        # self.dropout4 = nn.Dropout(0.25)

        # self.head_conv = nn.Conv1d(64, 1, kernel_size=1)
        # self.pool = nn.AdaptiveAvgPool1d(output_dim)

        self.log_beta = nn.Parameter(torch.log(torch.tensor(init_beta, dtype=torch.float32)))
        

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * input_length, 32 * input_length)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32 * input_length, output_dim)
        self.dropout = nn.Dropout(0.25)
        # self.softplus = nn.Softplus(beta=0.5)
        # self.softmax = nn.Softmax(dim = 1)
    def beta(self):
        return torch.exp(self.log_beta).clamp_min(1e-4)
    
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.dropout1(self.resblock1(x))
        x = self.dropout2(self.resblock2(x))

        x = self.flatten(x)
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        
        
        beta = self.beta()
        x = torch.nn.functional.softplus(x * beta)/beta
        return x
