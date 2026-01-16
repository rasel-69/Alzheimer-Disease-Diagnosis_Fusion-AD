import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os
import random
import timm

# ---------------- CONFIG ----------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 128        # smaller image size
N_WAY = 3             # smaller N-way
K_SHOT = 3            # smaller K-shot
Q_QUERY = 3           # smaller Q-query
EPOCHS = 60


train_dir = "/content/drive/MyDrive/Alzheimer_Disease/Split/train"
val_dir = "/content/drive/MyDrive/Alzheimer_Disease/Split/val"

# ---------------- CBAM Block ----------------
class CBAM(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        reduced_channels = max(1, channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, reduced_channels, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, channels, 1, bias=False)
        )
        self.spatial = nn.Conv2d(2, 1, kernel_size=7, padding=3)

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        att = torch.sigmoid(avg_out + max_out)
        x = x * att

        avg = torch.mean(x, dim=1, keepdim=True)
        max_ = torch.max(x, dim=1, keepdim=True)[0]
        spatial_att = torch.sigmoid(self.spatial(torch.cat((avg, max_), dim=1)))
        return x * spatial_att

# ---------------- CBAM + SwinViT Encoder ----------------
class CBAMSwinViTEncoder(nn.Module):
    def __init__(self, model_name='swin_tiny_patch4_window7_224', output_dim=OUTPUT_DIM):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=True, features_only=True)
        last_ch = self.backbone.feature_info[-1]['num_chs']
        print(f"SwinViT backbone last stage channels: {last_ch}")

        self.cbam = CBAM(last_ch)

        # Smaller conv head
        self.conv = nn.Sequential(
            nn.Conv2d(last_ch, 256, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )

    def forward(self, x):
        feats = self.backbone(x)[-1]  # last stage
        if feats.shape[1] not in (self.backbone.feature_info[-1]['num_chs'], 1, 3):
            feats = feats.permute(0, 3, 1, 2).contiguous()  # NHWC -> NCHW
        x = self.cbam(feats)
        x = self.conv(x)
        x = self.pool(x)
        x = self.head(x)
        return x


transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# ---------------- Dataset Loader ----------------
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

# ---------------- Episode Creation ----------------
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

    return (torch.stack(support_imgs).to(DEVICE),
            torch.tensor(support_lbls).to(DEVICE),
            torch.stack(query_imgs).to(DEVICE),
            torch.tensor(query_lbls).to(DEVICE))

# ---------------- Prototypical Functions ----------------
def compute_prototypes(support_embeddings, support_labels, n_way):
    return torch.stack([support_embeddings[support_labels == i].mean(dim=0) for i in range(n_way)])

def euclidean_distance(a, b):
    return torch.cdist(a, b)

def compute_prototypical_loss(encoder, sx, sy, qx, qy, n_way):
    encoder.train()
    support_embeddings = encoder(sx)
    query_embeddings = encoder(qx)
    prototypes = compute_prototypes(support_embeddings, sy, n_way)
    dists = euclidean_distance(query_embeddings, prototypes)
    logits = -dists
    loss = F.cross_entropy(logits, qy)
    preds = torch.argmax(logits, dim=1)
    acc = (preds == qy).float().mean()
    return loss, acc

# ---------------- Train Loop with Mixed Precision ----------------
encoder = CBAMSwinViTEncoder().to(DEVICE)
optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-4)
scaler = torch.cuda.amp.GradScaler()  # mixed precision

train_data = load_dataset(train_dir)
val_data = load_dataset(val_dir)

for epoch in range(1, EPOCHS+1):
    support_x, support_y, query_x, query_y = create_episode(train_data)
    optimizer.zero_grad()

    with torch.cuda.amp.autocast():
        loss, acc = compute_prototypical_loss(encoder, support_x, support_y, query_x, query_y, N_WAY)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    print(f"[Epoch {epoch}] Loss: {loss:.4f} | Accuracy: {acc*100:.2f}%")
