import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os
import random
import numpy as np
import timm  # EfficientNetV2 is supported here

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

# ----------------------------
# LayerNorm for CNN feature maps
# ----------------------------
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
# (From your first image – unchanged logic)
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




# --------------------------------------------------
# Encoder: ResNet50 + FMSAM
# --------------------------------------------------
class FuzzyResNet50Encoder(nn.Module):
    def __init__(self, output_dim=256, pretrained=True):
        super().__init__()

         # -------- FMSAM on top-level features --------
        self.fmsam = FMSAM(channels=2048)

        # -------- Load ResNet50 Backbone --------
        resnet = models.resnet50(pretrained=pretrained)

        # Keep layers up to layer4
        self.stem = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool
        )

        self.layer1 = resnet.layer1   # 256 channels
        self.layer2 = resnet.layer2   # 512 channels
        self.layer3 = resnet.layer3   # 1024 channels
        self.layer4 = resnet.layer4   # 2048 channels

       

        # -------- Projection Head --------
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Linear(512, output_dim)
        )

    def forward(self, x):
        # ResNet feature extraction
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)            # [B, 2048, H/32, W/32]

        # Fuzzy Multi-Scale Attention
        x = self.fmsam(x)

        # Global embedding
        x = self.head(x)
        return x


# ------------------ Data Preprocessing ------------------
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
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
        torch.tensor(query_lbls).to(DEVICE)
    )

# ------------------ Prototypical Loss Functions ------------------
def compute_prototypes(support_embeddings, support_labels, n_way):
    return torch.stack([
        support_embeddings[support_labels == i].mean(dim=0) for i in range(n_way)
    ])
# ------------------ using Euclidean Distance ------------------
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

# ------------------ Training Loop ------------------
encoder = FuzzyResNet50Encoder().to(DEVICE)
optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-4)

train_data = load_dataset(train_dir)
val_data = load_dataset(val_dir)

for epoch in range(1, EPOCHS + 1):
    support_x, support_y, query_x, query_y = create_episode(train_data)
    loss, acc = compute_prototypical_loss(encoder, support_x, support_y, query_x, query_y, N_WAY)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"[Epoch {epoch}] Loss: {loss:.4f} | Accuracy: {acc * 100:.2f}%")



# Save the trained model weights in .pt format
torch.save(encoder.state_dict(), "fuzzy_resnet50_encoder.pt")
print("Model saved as fuzzy_resnet50_encoder.pt")