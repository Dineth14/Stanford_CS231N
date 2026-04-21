# Mamba (S4/SSM-based) Classifier for CIFAR-10 Dataset
# A pure-PyTorch implementation of the Mamba selective state-space model.
# Pure PyTorch sequential scan (no Triton/custom CUDA kernels needed).

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import time
import math


class SelectiveSSM(nn.Module):
    """Simplified selective state-space model (Mamba core) in pure PyTorch."""

    def __init__(self, d_model, d_state=8, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = int(d_model * expand)

        # Input projection: x -> (z, x_inner)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # 1-D depthwise conv over the sequence
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size=d_conv, padding=d_conv - 1,
            groups=self.d_inner, bias=True,
        )

        # SSM parameters projected from input
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + 1, bias=False)

        # Learnable SSM matrix A (log-parameterized for stability)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).expand(self.d_inner, -1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # dt (delta) projection with proper initialization
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)
        with torch.no_grad():
            dt_init = torch.exp(
                torch.rand(self.d_inner) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)
            )
            inv_dt = dt_init + torch.log(-torch.expm1(-dt_init))
            self.dt_proj.bias.copy_(inv_dt)
        nn.init.kaiming_uniform_(self.dt_proj.weight, a=math.sqrt(5))

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x):
        """x: (B, L, d_model) -> (B, L, d_model)"""
        B, L, _ = x.shape

        # Project and split
        xz = self.in_proj(x)
        x_inner, z = xz.chunk(2, dim=-1)

        # Causal 1-D conv
        x_inner = x_inner.transpose(1, 2)
        x_inner = self.conv1d(x_inner)[:, :, :L]
        x_inner = x_inner.transpose(1, 2)
        x_inner = F.silu(x_inner)

        # SSM parameters from input
        x_ssm = self.x_proj(x_inner)
        B_param = x_ssm[:, :, :self.d_state]
        C_param = x_ssm[:, :, self.d_state:self.d_state * 2]
        dt_raw = x_ssm[:, :, -1:]

        dt = F.softplus(self.dt_proj(dt_raw))
        A = -torch.exp(self.A_log)

        y = self._selective_scan(x_inner, dt, A, B_param, C_param)
        y = y + x_inner * self.D.unsqueeze(0).unsqueeze(0)
        y = y * F.silu(z)
        return self.out_proj(y)

    def _selective_scan(self, x, dt, A, B_param, C_param):
        """Memory-efficient sequential scan."""
        B_batch, L, d_inner = x.shape

        ys = torch.empty(B_batch, L, d_inner, device=x.device, dtype=x.dtype)
        h = torch.zeros(B_batch, d_inner, self.d_state, device=x.device, dtype=x.dtype)
        for t in range(L):
            dt_t = dt[:, t]
            dA_t = torch.exp(A.unsqueeze(0) * dt_t.unsqueeze(-1))
            dB_t = dt_t.unsqueeze(-1) * B_param[:, t].unsqueeze(1)
            h = dA_t * h + dB_t * x[:, t].unsqueeze(-1)
            ys[:, t] = (h * C_param[:, t].unsqueeze(1)).sum(-1)
        return ys


class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=8, d_conv=4, expand=2, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ssm = SelectiveSSM(d_model, d_state, d_conv, expand)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return x + self.dropout(self.ssm(self.norm(x)))


class VisionMamba(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_channels=3, num_classes=10,
                 d_model=128, depth=4, d_state=8, d_conv=4, expand=2, dropout=0.1):
        super().__init__()
        num_patches = (img_size // patch_size) ** 2

        self.patch_embed = nn.Conv2d(in_channels, d_model, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, d_model))
        self.pos_drop = nn.Dropout(dropout)

        self.blocks = nn.Sequential(*[
            MambaBlock(d_model, d_state, d_conv, expand, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        x = self.blocks(x)
        x = self.norm(x)
        return self.head(x.mean(dim=1))


def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f'Using GPU: {torch.cuda.get_device_name(0)}')
        print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
    else:
        device = torch.device('cpu')
        print('Using CPU')
    return device


def get_dataloaders(batch_size=128):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return trainloader, testloader


def train_one_epoch(model, trainloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    num_batches = len(trainloader)

    for batch_idx, (inputs, labels) in enumerate(trainloader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        if (batch_idx + 1) % 100 == 0:
            print(f'  Batch [{batch_idx+1}/{num_batches}] Loss: {loss.item():.4f}', flush=True)

    return running_loss / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(model, testloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in testloader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return running_loss / total, 100.0 * correct / total


def main():
    device = get_device()
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')

    num_epochs = 30
    batch_size = 128
    learning_rate = 1e-3

    trainloader, testloader = get_dataloaders(batch_size)

    model = VisionMamba(
        img_size=32, patch_size=4, in_channels=3, num_classes=10,
        d_model=128, depth=4, d_state=8, d_conv=4, expand=2, dropout=0.1,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f'Model parameters: {total_params:,}')

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.05)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    best_acc = 0.0
    start_time = time.time()
    print('Starting training...', flush=True)

    for epoch in range(num_epochs):
        epoch_start = time.time()
        train_loss, train_acc = train_one_epoch(model, trainloader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, testloader, criterion, device)
        scheduler.step()
        epoch_time = time.time() - epoch_start

        print(f'Epoch [{epoch+1}/{num_epochs}] '
              f'Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | '
              f'Test Loss: {test_loss:.4f} Acc: {test_acc:.2f}% | '
              f'Time: {epoch_time:.1f}s')

        if test_acc > best_acc:
            best_acc = test_acc
            save_path = os.path.join(os.path.dirname(__file__), 'best_mamba_cifar10.pth')
            torch.save(model.state_dict(), save_path)
            print(f'  -> Saved best model ({test_acc:.2f}%)')

    elapsed = time.time() - start_time
    print(f'\nTraining complete in {elapsed/60:.1f} minutes')
    print(f'Best Test Accuracy: {best_acc:.2f}%')
    print(f'Model saved to: {os.path.join(os.path.dirname(__file__), "best_mamba_cifar10.pth")}')


if __name__ == '__main__':
    main()
