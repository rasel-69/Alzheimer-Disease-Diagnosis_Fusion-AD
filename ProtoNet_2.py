# If timm isn't installed in your Colab:
# !pip -q install timm

import os
import math
import random
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import timm

# ------------------ CONFIG ------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 224
N_WAY = 4
K_SHOT = 5
Q_QUERY = 5
EPOCHS = 60


train_dir = "/content/drive/MyDrive/Alzheimer_Disease/Split/train"
val_dir = "/content/drive/MyDrive/Alzheimer_Disease/Split/val"
test_dir = "/content/drive/MyDrive/Alzheimer_Disease/Split/test"

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)



class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.ln = nn.LayerNorm(channels, eps=eps)

    def forward(self, x):
        # x: [B, C, H, W]
        return self.ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

# ----------------------------
# Efficient Channel Attention (ECA)
# ----------------------------
class ECA(nn.Module):
    def __init__(self, channels, k_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size,
                              padding=(k_size - 1) // 2,
                              bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)                     # [B, C, 1, 1]
        y = self.conv(y.squeeze(-1).transpose(1, 2))
        y = self.sigmoid(y).transpose(1, 2).unsqueeze(-1)
        return x * y.expand_as(x)

# ============================================================
# MCAU: Multi-Scale Contextual Attention Unit
# ============================================================
class MCAU(nn.Module):
    def __init__(self, channels):
        super().__init__()

        # Multi-scale depthwise convolutions
        self.dw3 = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.dw7 = nn.Conv2d(channels, channels, 7, padding=3, groups=channels)
        self.dw11 = nn.Conv2d(channels, channels, 11, padding=5, groups=channels)

        self.ln3 = LayerNorm2d(channels)
        self.ln7 = LayerNorm2d(channels)
        self.ln11 = LayerNorm2d(channels)

        # Pointwise convs
        self.pconv1 = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.ln_p1 = LayerNorm2d(channels)

        self.pconv2 = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.ln_p2 = LayerNorm2d(channels)

        self.eca = ECA(channels)

    def forward(self, x):
        x3 = self.ln3(self.dw3(x))
        x7 = self.ln7(self.dw7(x))
        x11 = self.ln11(self.dw11(x))

        x_sum = x3 + x7 + x11                     # ⊕ element-wise add

        x = self.pconv1(x_sum)
        x = self.ln_p1(x)

        x = self.pconv2(x)
        x = self.ln_p2(x)

        x = self.eca(x)
        return x * x_sum                          # ⊗ element-wise multiply

# ============================================================
# FFN Block (as shown in FMSAM diagram)
# ============================================================
class FFNBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.pconv1 = nn.Conv2d(channels, channels, 1, bias=False)
        self.ln1 = LayerNorm2d(channels)

        self.pconv2 = nn.Conv2d(channels, channels, 1, bias=False)
        self.ln2 = LayerNorm2d(channels)

        self.act = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        residual = x
        x = self.ln1(self.pconv1(x))
        x = self.ln2(self.pconv2(x))
        x = self.act(x)
        x = self.conv3(x)
        return x + residual                       # ⊕ residual

# ============================================================
# FLM + FMF (Fuzzy Logic Modulation)
# ============================================================
class FuzzyLogicModulation(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.fmf = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.Softmax(dim=1)
        )

        self.flm = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        membership = self.fmf(x)
        x = x * membership
        gate = self.flm(x)
        return x * gate

# ============================================================
# FMSAM: Full Module (Final)
# ============================================================
class FMSAM(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.ffn = FFNBlock(channels)
        self.mcau = MCAU(channels)
        self.ln_after_mcau = LayerNorm2d(channels)

        self.flm = FuzzyLogicModulation(channels)
        self.final_conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        # FFN branch
        x_ffn = self.ffn(x)

        # MCAU attention
        x_att = self.mcau(x_ffn)
        x_att = self.ln_after_mcau(x_att)

        # Fuzzy modulation
        x_fuzzy = self.flm(x_att)

        # Final residual + conv
        out = self.final_conv(x_fuzzy + x_ffn)
        return out


# ------------------ Encoder: FMSAM(fuzzy logic) + ConvNeXtSmall ------------------
class FuzzyConvNeXtSmallEncoder(nn.Module):
    def __init__(self, model_name='convnext_small', output_dim=256):
        super().__init__()

        self.backbone = timm.create_model(
            model_name, pretrained=True, features_only=True
        )

        in_channels = self.backbone.feature_info[-1]['num_chs']  

        # Fuzzy Multi-Scale Attention
        self.fuzzy_att = FuzzyMultiScaleAttention(in_channels)

        # FMSAM module (JUST CALLED, no structure changes)
        self.fmsam = FMSAM(in_channels)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),

            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256)
        )

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, output_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

    def forward(self, x):
        features = self.backbone(x)[-1]    # [B, C, H, W]

        x = self.fuzzy_att(features)       # Fuzzy attention
        x = self.fmsam(x)                  # 🔥 FMSAM added here

        x = self.conv(x)
        x = self.pool(x)

        return self.fc(x)


# Separate transforms for train/eval (optional augmentation for train)
transform_train = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    # You can enable light augmentation if you want:
    # transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

transform_eval = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

def load_dataset(data_dir, min_images_per_class=K_SHOT + Q_QUERY):
    data = {}
    class_names = sorted(d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d)))
    for cls in class_names:
        folder = os.path.join(data_dir, cls)
        images = [os.path.join(folder, f) for f in os.listdir(folder)
                  if f.lower().endswith(('png', 'jpg', 'jpeg'))]
        if len(images) >= min_images_per_class:
            data[cls] = images
    if len(data) < N_WAY:
        raise ValueError(f"Not enough classes with at least {min_images_per_class} images. "
                         f"Found only {len(data)} classes in {data_dir}.")
    return data

def create_episode(data, transform, n_way=N_WAY, k_shot=K_SHOT, q_query=Q_QUERY):
    # Sample classes
    selected_classes = random.sample(list(data.keys()), n_way)
    support_imgs, support_lbls, query_imgs, query_lbls = [], [], [], []

    for idx, cls in enumerate(selected_classes):
        # sample k+q images from this class
        pool = data[cls]
        if len(pool) < k_shot + q_query:
            raise ValueError(f"Class {cls} has only {len(pool)} images, needs {k_shot + q_query}.")
        selected = random.sample(pool, k_shot + q_query)

        for img_path in selected[:k_shot]:
            img = Image.open(img_path).convert('RGB')
            support_imgs.append(transform(img))
            support_lbls.append(idx)
        for img_path in selected[k_shot:]:
            img = Image.open(img_path).convert('RGB')
            query_imgs.append(transform(img))
            query_lbls.append(idx)

    return (
        torch.stack(support_imgs).to(DEVICE),
        torch.tensor(support_lbls, device=DEVICE),
        torch.stack(query_imgs).to(DEVICE),
        torch.tensor(query_lbls, device=DEVICE)
    )

# ------------------ Prototypical Loss Functions ------------------
def compute_prototypes(support_embeddings, support_labels, n_way):
    return torch.stack([
        support_embeddings[support_labels == i].mean(dim=0) for i in range(n_way)
    ])

def euclidean_distance(a, b):
    return torch.cdist(a, b)

def compute_prototypical_loss(encoder, sx, sy, qx, qy, n_way):
    support_embeddings = encoder(sx)  # [K*N, D]
    query_embeddings = encoder(qx)    # [Q*N, D]
    prototypes = compute_prototypes(support_embeddings, sy, n_way)  # [N, D]
    dists = euclidean_distance(query_embeddings, prototypes)        # [Q*N, N]
    logits = -dists
    loss = F.cross_entropy(logits, qy)
    preds = torch.argmax(logits, dim=1)
    acc = (preds == qy).float().mean()
    return loss, acc

@torch.no_grad()
def evaluate_on_episodes(encoder, data, transform, n_way, k_shot, q_query, episodes=5):
    encoder.eval()
    losses, accs = [], []
    for _ in range(episodes):
        sx, sy, qx, qy = create_episode(data, transform, n_way, k_shot, q_query)
        loss, acc = compute_prototypical_loss(encoder, sx, sy, qx, qy, n_way)
        losses.append(loss.item())
        accs.append(acc.item())
    return float(np.mean(losses)), float(np.mean(accs))

# ------------------ Training  ------------------
encoder = FuzzyConvNeXtSmallEncoder().to(DEVICE)
optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-4)

train_data = load_dataset(train_dir, min_images_per_class=K_SHOT + Q_QUERY)
val_data = load_dataset(val_dir, min_images_per_class=K_SHOT + Q_QUERY)

train_losses, train_accs = [], []
val_losses, val_accs = [], []

best_val_acc = 0.0
best_state = None

for epoch in range(1, EPOCHS + 1):
    # Train on one episode
    encoder.train()
    sx, sy, qx, qy = create_episode(train_data, transform_train, n_way=N_WAY, k_shot=K_SHOT, q_query=Q_QUERY)
    loss, acc = compute_prototypical_loss(encoder, sx, sy, qx, qy, N_WAY)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    train_losses.append(loss.item())
    train_accs.append(acc.item())


    val_loss, val_acc = evaluate_on_episodes(
        encoder, val_data, transform_eval, n_way=N_WAY, k_shot=K_SHOT, q_query=Q_QUERY, episodes=EVAL_EPISODES
    )
    val_losses.append(val_loss)
    val_accs.append(val_acc)

    # Track best
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_state = {k: v.detach().cpu().clone() for k, v in encoder.state_dict().items()}

    print(f"[Epoch {epoch:03d}] "
          f"train_loss={loss.item():.4f} train_acc={acc.item()*100:.2f}% | "
          f"val_loss={val_loss:.4f} val_acc={val_acc*100:.2f}%")

# Restore best checkpoint and save it
if best_state is not None:
    encoder.load_state_dict(best_state)
    torch.save(encoder.state_dict(), "/content/best_encoder.pth")
    print(f"Loaded and saved best model (val_acc={best_val_acc*100:.2f}%) to /content/best_encoder.pth")

# Keep history for plotting
history = {
    "train_losses": train_losses,
    "train_accs": train_accs,
    "val_losses": val_losses,
    "val_accs": val_accs,
}