import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os
import random
import numpy as np

# ------------------ CONFIG ------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 224
N_WAY = 4
K_SHOT = 5
Q_QUERY = 5
EPOCHS = 60


train_dir = "/content/drive/MyDrive/Alzheimer_Disease/Split/train"
val_dir   = "/content/drive/MyDrive/Alzheimer_Disease/Split/val"
test_dir  = "/content/drive/MyDrive/Alzheimer_Disease/Split/test"

# ------------------ REPRODUCIBILITY ------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ------------------ MISH ACTIVATION ------------------
class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

# ------------------ CBAM MODULE ------------------
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
        avg_out = self.channel_att(self.avg_pool(x))
        max_out = self.channel_att(self.max_pool(x))
        x = x * torch.sigmoid(avg_out + max_out)
        avg = torch.mean(x, dim=1, keepdim=True)
        max_, _ = torch.max(x, dim=1, keepdim=True)
        x = x * torch.sigmoid(self.spatial_att(torch.cat([avg, max_], dim=1)))
        return x

# ------------------ PATCH EMBEDDING ------------------
class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.num_patches = (img_size // patch_size) ** 2

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x

# ------------------ CBAM ATTENTION FOR SWIN-ViT ------------------
class CBAMAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.cbam = CBAM(dim)

    def forward(self, x):
        B, N, C = x.shape
        H = W = int(N ** 0.5)
        assert H * W == N, "Number of patches must be perfect square"
        x = x.transpose(1,2).reshape(B, C, H, W)
        x = self.cbam(x)
        x = x.flatten(2).transpose(1,2)
        return x

# ------------------ MLP WITH MISH ------------------
class MlpMish(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = Mish()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

# ------------------ SWIN BLOCK WITH CBAM ------------------
class SwinBlockCBAM(nn.Module):
    def __init__(self, dim, mlp_ratio=4., drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = CBAMAttention(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MlpMish(dim, int(dim*mlp_ratio), drop=drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

# ------------------ SWIN-ViT ENCODER WITH CBAM ------------------
class CBAMSwinViTEncoder(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, depths=2, output_dim=OUTPUT_DIM):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.pos_drop = nn.Dropout(0.)
        self.blocks = nn.Sequential(*[SwinBlockCBAM(embed_dim) for _ in range(depths)])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, output_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
    def forward(self, x):
        x = self.patch_embed(x)
        x = self.pos_drop(x)
        x = self.blocks(x)
        x = self.norm(x)
        x = x.mean(dim=1)
        x = self.head(x)
        return x

# ------------------ TRANSFORMS ------------------
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# ------------------ DATASET ------------------
def load_dataset(data_dir):
    data = {}
    class_names = sorted(os.listdir(data_dir))
    for cls in class_names:
        folder = os.path.join(data_dir, cls)
        if os.path.isdir(folder):
            images = [os.path.join(folder, f) for f in os.listdir(folder)
                      if f.lower().endswith(('png','jpg','jpeg'))]
            if images:
                data[cls] = images
    return data

def get_allowed_classes(data, needed):
    return [c for c, imgs in data.items() if len(imgs) >= needed]

def create_episode(data, allowed_classes, n_way=N_WAY, k_shot=K_SHOT, q_query=Q_QUERY):
    selected_classes = random.sample(allowed_classes, n_way)
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

# ------------------ PROTOTYPICAL LOSS ------------------
def compute_prototypes(support_embeddings, support_labels, n_way):
    return torch.stack([support_embeddings[support_labels==i].mean(0) for i in range(n_way)])

def euclidean_distance(a, b):
    return torch.cdist(a, b)

def prototypical_loss(encoder, sx, sy, qx, qy, n_way):
    encoder.train()
    support_emb = encoder(sx)
    query_emb   = encoder(qx)
    prototypes  = compute_prototypes(support_emb, sy, n_way)
    dists       = euclidean_distance(query_emb, prototypes)
    logits      = -dists
    loss        = F.cross_entropy(logits, qy)
    acc         = (logits.argmax(1) == qy).float().mean()
    return loss, acc

@torch.no_grad()
def prototypical_eval(encoder, sx, sy, qx, qy, n_way):
    encoder.eval()
    support_emb = encoder(sx)
    query_emb   = encoder(qx)
    prototypes  = compute_prototypes(support_emb, sy, n_way)
    dists       = euclidean_distance(query_emb, prototypes)
    logits      = -dists
    loss        = F.cross_entropy(logits, qy)
    acc         = (logits.argmax(1) == qy).float().mean()
    return loss, acc

# ------------------ TRAINING ------------------
encoder = CBAMSwinViTEncoder().to(DEVICE)
optimizer = torch.optim.Adam(encoder.parameters(), lr=LR)

train_data = load_dataset(train_dir)
val_data   = load_dataset(val_dir)

allowed_train = get_allowed_classes(train_data, K_SHOT + Q_QUERY)
allowed_val   = get_allowed_classes(val_data, K_SHOT + Q_QUERY)

for epoch in range(1, EPOCHS+1):
    sx, sy, qx, qy = create_episode(train_data, allowed_train)
    loss, acc = prototypical_loss(encoder, sx, sy, qx, qy, N_WAY)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # validation
    val_losses, val_accs = [], []
    for _ in range(EVAL_EPISODES):
        sx, sy, qx, qy = create_episode(val_data, allowed_val)
        v_loss, v_acc = prototypical_eval(encoder, sx, sy, qx, qy, N_WAY)
        val_losses.append(v_loss.item())
        val_accs.append(v_acc.item())

    print(f"[Epoch {epoch:03d}] Train Loss: {loss.item():.4f} | "
          f"Train Acc: {acc.item()*100:.2f}% | "
          f"Val Loss: {np.mean(val_losses):.4f} | "
          f"Val Acc: {np.mean(val_accs)*100:.2f}%")

