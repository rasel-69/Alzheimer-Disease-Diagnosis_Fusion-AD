

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
from PIL import Image
import os
import random
import numpy as np

# =========================================================
# CONFIGURATION
# =========================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMG_SIZE = 224
N_WAY = 4
K_SHOT = 5
Q_QUERY = 5
EPOCHS = 60


train_dir = "/content/drive/MyDrive/Alzheimer_Disease/Split/train"
val_dir   = "/content/drive/MyDrive/Alzheimer_Disease/Split/val"
test_dir  = "/content/drive/MyDrive/Alzheimer_Disease/Split/test"

# =========================================================
# CBAM MODULE
# =========================================================
class CBAM(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        reduced = max(1, channels // reduction)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.channel_att = nn.Sequential(
            nn.Conv2d(channels, reduced, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced, channels, 1, bias=False)
        )

        self.spatial_att = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, x):
        # Channel Attention
        avg_out = self.channel_att(self.avg_pool(x))
        max_out = self.channel_att(self.max_pool(x))
        x = x * torch.sigmoid(avg_out + max_out)

        # Spatial Attention
        avg = torch.mean(x, dim=1, keepdim=True)
        max_, _ = torch.max(x, dim=1, keepdim=True)
        spatial = torch.cat([avg, max_], dim=1)
        x = x * torch.sigmoid(self.spatial_att(spatial))

        return x

# =========================================================
# CBAM + MODIFIED MOBILENETV3 ENCODER
# =========================================================
class CBAMMobileNetV3Encoder(nn.Module):
    def __init__(self, output_dim=EMBEDDING_DIM):
        super().__init__()

        backbone = mobilenet_v3_large(
            weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1
        )

        self.features = backbone.features      # Output: 960 channels
        self.cbam = CBAM(channels=960)
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(960, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, output_dim)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.cbam(x)
        x = self.pool(x)
        x = self.projection(x)
        return x

# =========================================================
# IMAGE TRANSFORMS
# =========================================================
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# =========================================================
# DATASET LOADING
# =========================================================
def load_dataset(root):
    dataset = {}
    classes = sorted(os.listdir(root))
    for cls in classes:
        cls_path = os.path.join(root, cls)
        if os.path.isdir(cls_path):
            images = [
                os.path.join(cls_path, f)
                for f in os.listdir(cls_path)
                if f.lower().endswith((".jpg", ".png", ".jpeg"))
            ]
            dataset[cls] = images
    return dataset

# =========================================================
# EPISODIC SAMPLING
# =========================================================
def create_episode(data, n_way, k_shot, q_query):
    selected_classes = random.sample(list(data.keys()), n_way)

    support_x, support_y = [], []
    query_x, query_y = [], []

    for label, cls in enumerate(selected_classes):
        samples = random.sample(data[cls], k_shot + q_query)

        for img in samples[:k_shot]:
            support_x.append(transform(Image.open(img).convert("RGB")))
            support_y.append(label)

        for img in samples[k_shot:]:
            query_x.append(transform(Image.open(img).convert("RGB")))
            query_y.append(label)

    return (
        torch.stack(support_x).to(DEVICE),
        torch.tensor(support_y).to(DEVICE),
        torch.stack(query_x).to(DEVICE),
        torch.tensor(query_y).to(DEVICE),
    )

# =========================================================
# PROTOTYPICAL NETWORK FUNCTIONS
# =========================================================
def compute_prototypes(embeddings, labels, n_way):
    return torch.stack([
        embeddings[labels == i].mean(0) for i in range(n_way)
    ])

def prototypical_loss(encoder, sx, sy, qx, qy, n_way):
    support_emb = encoder(sx)
    query_emb   = encoder(qx)

    prototypes = compute_prototypes(support_emb, sy, n_way)
    distances  = torch.cdist(query_emb, prototypes)

    logits = -distances
    loss = F.cross_entropy(logits, qy)

    preds = torch.argmax(logits, dim=1)
    acc = (preds == qy).float().mean()

    return loss, acc

# =========================================================
# TRAINING LOOP
# =========================================================
encoder = CBAMMobileNetV3Encoder().to(DEVICE)
optimizer = torch.optim.Adam(encoder.parameters(), lr=LR)

train_data = load_dataset(train_dir)
val_data   = load_dataset(val_dir)

print(" Training Started...\n")

for epoch in range(1, EPOCHS + 1):
    encoder.train()

    sx, sy, qx, qy = create_episode(
        train_data, N_WAY, K_SHOT, Q_QUERY
    )

    loss, acc = prototypical_loss(
        encoder, sx, sy, qx, qy, N_WAY
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(
        f"Epoch [{epoch:03d}/{EPOCHS}] "
        f"Loss: {loss.item():.4f} | "
        f"Accuracy: {acc.item()*100:.2f}%"
    )

print("\n Training Completed Successfully!")
