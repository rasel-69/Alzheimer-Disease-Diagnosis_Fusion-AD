# ===========================
# Using CBAM + ResNet50 Encoder + Prototypical Network
# ===========================

import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np
from sklearn.metrics import classification_report
import time

# ===========================
#  CONFIGURATION
# ===========================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 224
N_WAY = 4
K_SHOT = 5
Q_QUERY = 5
EPOCHS = 60
LR = 1e-4

train_dir = "/content/drive/MyDrive/Alzheimer_Disease/Split/train"
val_dir = "/content/drive/MyDrive/Alzheimer_Disease/Split/val"

# ===========================
#  CBAM MODULE
# ===========================
class CBAM(nn.Module):
    def __init__(self, channels, reduction=8):
        super(CBAM, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.spatial = nn.Conv2d(2, 1, kernel_size=7, padding=3)

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        att = torch.sigmoid(avg_out + max_out)
        x = x * att

        avg = torch.mean(x, dim=1, keepdim=True)
        max_ = torch.max(x, dim=1, keepdim=True)[0]
        concat = torch.cat((avg, max_), dim=1)
        spatial_att = torch.sigmoid(self.spatial(concat))
        return x * spatial_att

# ===========================
# ENCODER (ResNet50 + CBAM)
# ===========================
class CBAMResNetEncoder(nn.Module):
    def __init__(self, output_dim=256):
        super().__init__()
        base = models.resnet50(pretrained=True)
        self.features = nn.Sequential(*list(base.children())[:-2])  # remove avgpool and fc
        self.cbam = CBAM(2048)
        self.conv = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=3, padding=1),
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
        x = self.features(x)
        x = self.cbam(x)
        x = self.conv(x)
        x = self.pool(x)
        x = self.fc(x)
        return x

# ===========================
#  DATASET LOADING
# ===========================
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

def load_dataset(data_dir):
    data = {}
    class_names = sorted(os.listdir(data_dir))
    for cls in class_names:
        folder = os.path.join(data_dir, cls)
        if os.path.isdir(folder):
            images = [os.path.join(folder, f) for f in os.listdir(folder)
                      if f.lower().endswith(('png', 'jpg', 'jpeg'))]
            data[cls] = images
    return data

def create_episode(data, n_way=N_WAY, k_shot=K_SHOT, q_query=Q_QUERY):
    selected_classes = random.sample(list(data.keys()), n_way)
    support_imgs, support_lbls, query_imgs, query_lbls = [], [], [], []

    for idx, cls in enumerate(selected_classes):
        selected = random.sample(data[cls], k_shot + q_query)
        for img_path in selected[:k_shot]:
            support_imgs.append(transform(Image.open(img_path).convert('RGB')))
            support_lbls.append(idx)
        for img_path in selected[k_shot:]:
            query_imgs.append(transform(Image.open(img_path).convert('RGB')))
            query_lbls.append(idx)

    return (
        torch.stack(support_imgs).to(DEVICE),
        torch.tensor(support_lbls).to(DEVICE),
        torch.stack(query_imgs).to(DEVICE),
        torch.tensor(query_lbls).to(DEVICE),
        selected_classes
    )

# ===========================
#  PROTOTYPICAL NETWORK UTILITIES
# ===========================
def compute_prototypes(support_embeddings, support_labels, n_way):
    prototypes = []
    for i in range(n_way):
        cls_embeddings = support_embeddings[support_labels == i]
        prototypes.append(torch.mean(cls_embeddings, dim=0))
    return torch.stack(prototypes)

def euclidean_distance(a, b):
    return torch.cdist(a, b)

def compute_prototypical_loss(model, support_x, support_y, query_x, query_y, n_way):
    support_embeddings = model(support_x)
    query_embeddings = model(query_x)
    prototypes = compute_prototypes(support_embeddings, support_y, n_way)
    dists = euclidean_distance(query_embeddings, prototypes)
    log_p_y = F.log_softmax(-dists, dim=1)
    loss = F.nll_loss(log_p_y, query_y)
    acc = (log_p_y.argmax(dim=1) == query_y).float().mean()
    return loss, acc.item(), log_p_y.argmax(dim=1)

# ===========================
#  TRAINING LOOP
# ===========================
encoder = CBAMResNetEncoder().to(DEVICE)
optimizer = torch.optim.Adam(encoder.parameters(), lr=LR)

train_data = load_dataset(train_dir)
val_data = load_dataset(val_dir)

best_val_acc = 0.0

for epoch in range(1, EPOCHS + 1):
    encoder.train()
    support_x, support_y, query_x, query_y, _ = create_episode(train_data)
    loss, acc, _ = compute_prototypical_loss(encoder, support_x, support_y, query_x, query_y, N_WAY)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # --- Validation ---
    encoder.eval()
    with torch.no_grad():
        support_x, support_y, query_x, query_y, _ = create_episode(val_data)
        val_loss, val_acc, _ = compute_prototypical_loss(encoder, support_x, support_y, query_x, query_y, N_WAY)

    print(f"[Epoch {epoch}/{EPOCHS}] Train Loss: {loss:.4f} | Train Acc: {acc*100:.2f}% | Val Acc: {val_acc*100:.2f}%")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(encoder.state_dict(), "best.pt")
        print(f" Saved best model at epoch {epoch} with Val Acc = {val_acc*100:.2f}%")

print("\nTraining complete. Best Validation Accuracy:", best_val_acc*100)

# ===========================
#  CLASSIFICATION REPORT 
# ===========================
encoder.load_state_dict(torch.load("best.pt"))
encoder.eval()

y_true, y_pred, all_labels = [], [], []

with torch.no_grad():
    for _ in range(10):  # test on multiple random episodes
        support_x, support_y, query_x, query_y, cls_names = create_episode(val_data)
        _, _, preds = compute_prototypical_loss(encoder, support_x, support_y, query_x, query_y, N_WAY)
        y_true.extend(query_y.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())
        all_labels = cls_names

print("\n=== CLASSIFICATION REPORT ===")
print(classification_report(y_true, y_pred, target_names=[str(n) for n in all_labels]))
