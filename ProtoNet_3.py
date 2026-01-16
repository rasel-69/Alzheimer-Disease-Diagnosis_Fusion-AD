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
val_dir = "/content/drive/MyDrive/Alzheimer_Disease/Split/val"
test_dir = "/content/drive/MyDrive/Alzheimer_Disease/Split/test"

# Reproducibility
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ------------------ LayerNorm2d ------------------
class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.ln = nn.LayerNorm(channels, eps=eps)
    def forward(self, x):
        return self.ln(x.permute(0,2,3,1)).permute(0,3,1,2)

# ------------------ Patch Embedding ------------------
class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.num_patches = (img_size // patch_size) ** 2
    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1,2)  # [B, N, C]
        return x

# ------------------ MLP with GELU ------------------
class MlpGELU(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

# ------------------ Fuzzy Gated Attention ------------------
class FuzzyGatedAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # Using identity attention since we remove MCAU/FMSAM
    def forward(self, x):
        return x

# ------------------ Swin Transformer Block (default) ------------------
class SwinBlockDefault(nn.Module):
    def __init__(self, dim, mlp_ratio=4., drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = FuzzyGatedAttention(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MlpGELU(dim, int(dim*mlp_ratio), drop=drop)
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

# ------------------ Default SwinViT Encoder ------------------
class DefaultSwinViTEncoder(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, depths=2, output_dim=256):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.pos_drop = nn.Dropout(0.)
        self.blocks = nn.Sequential(*[SwinBlockDefault(embed_dim) for _ in range(depths)])
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
        x = x.mean(dim=1)  # Global average
        x = self.head(x)
        return x

# ------------------ Dataset and transforms ------------------
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
            images = [os.path.join(folder,f) for f in os.listdir(folder)
                      if f.lower().endswith(('png','jpg','jpeg'))]
            if images:
                data[cls] = images
    return data

def get_allowed_classes(data, needed):
    return [c for c, paths in data.items() if len(paths) >= needed]

def create_episode_from_allowed(data, allowed_classes, n_way=N_WAY, k_shot=K_SHOT, q_query=Q_QUERY):
    selected_classes = random.sample(allowed_classes, n_way)
    support_imgs, support_lbls, query_imgs, query_lbls = [], [], [], []
    for idx, cls in enumerate(selected_classes):
        selected = random.sample(data[cls], k_shot+q_query)
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

# ------------------ Prototypical Loss ------------------
def compute_prototypes(support_embeddings, support_labels, n_way):
    return torch.stack([support_embeddings[support_labels==i].mean(dim=0) for i in range(n_way)])

def euclidean_distance(a,b):
    return torch.cdist(a,b)

def compute_prototypical_loss(encoder, sx, sy, qx, qy, n_way):
    encoder.train()
    support_embeddings = encoder(sx)
    query_embeddings = encoder(qx)
    prototypes = compute_prototypes(support_embeddings, sy, n_way)
    dists = euclidean_distance(query_embeddings, prototypes)
    logits = -dists
    loss = F.cross_entropy(logits, qy)
    preds = torch.argmax(logits, dim=1)
    acc = (preds==qy).float().mean()
    return loss, acc

@torch.no_grad()
def compute_prototypical_loss_eval(encoder, sx, sy, qx, qy, n_way):
    encoder.eval()
    support_embeddings = encoder(sx)
    query_embeddings = encoder(qx)
    prototypes = compute_prototypes(support_embeddings, sy, n_way)
    dists = euclidean_distance(query_embeddings, prototypes)
    logits = -dists
    loss = F.cross_entropy(logits, qy)
    preds = torch.argmax(logits, dim=1)
    acc = (preds==qy).float().mean()
    return loss, acc

@torch.no_grad()
def evaluate_on_val(encoder, val_data, n_way, k_shot, q_query, episodes=5):
    allowed = get_allowed_classes(val_data, k_shot+q_query)
    if len(allowed) < n_way:
        raise ValueError(f"Validation has only {len(allowed)} classes; need >= {n_way}")
    losses, accs = [], []
    for _ in range(episodes):
        sx, sy, qx, qy = create_episode_from_allowed(val_data, allowed, n_way, k_shot, q_query)
        loss, acc = compute_prototypical_loss_eval(encoder, sx, sy, qx, qy, n_way)
        losses.append(loss.item()); accs.append(acc.item())
    return float(np.mean(losses)), float(np.mean(accs))

# ------------------ Training ------------------
encoder = DefaultSwinViTEncoder().to(DEVICE)
optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-4)

train_data = load_dataset(train_dir)
val_data = load_dataset(val_dir)

allowed_train = get_allowed_classes(train_data, K_SHOT+Q_QUERY)
if len(allowed_train) < N_WAY:
    raise ValueError(f"Train has only {len(allowed_train)} classes; need >= {N_WAY}")

train_losses, train_accs, val_losses, val_accs = [],[],[],[]
scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

for epoch in range(1, EPOCHS+1):
    sx, sy, qx, qy = create_episode_from_allowed(train_data, allowed_train, n_way=N_WAY, k_shot=K_SHOT, q_query=Q_QUERY)
    with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
        loss, acc = compute_prototypical_loss(encoder, sx, sy, qx, qy, N_WAY)
    optimizer.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
    scaler.step(optimizer)
    scaler.update()
    train_losses.append(loss.item())
    train_accs.append(acc.item())
    val_loss, val_acc = evaluate_on_val(encoder, val_data, n_way=N_WAY, k_shot=K_SHOT, q_query=Q_QUERY, episodes=EVAL_EPISODES)
    val_losses.append(val_loss)
    val_accs.append(val_acc)
    print(f"[Epoch {epoch:03d}] train_loss={loss.item():.4f} train_acc={acc.item()*100:.2f}% | val_loss={val_loss:.4f} val_acc={val_acc*100:.2f}%")

# Save history
np.savez("/content/train_history.npz",
         train_losses=np.array(train_losses),
         train_accs=np.array(train_accs),
         val_losses=np.array(val_losses),
         val_accs=np.array(val_accs))
print("Saved training history to /content/train_history.npz")

# Save model
state_path = "/content/default_swinvit_encoder_state.pt"
torch.save(encoder.state_dict(), state_path)
print(f"Saved state_dict to: {state_path}")

encoder_cpu = DefaultSwinViTEncoder().to("cpu")
encoder_cpu.load_state_dict(torch.load(state_path,map_location="cpu"))
encoder_cpu.eval()
script_path = "/content/default_swinvit_encoder_scripted.pt"
try:
    scripted = torch.jit.script(encoder_cpu)
    scripted.save(script_path)
    print(f"Saved TorchScript (script) to: {script_path}")
except Exception as e:
    print(f"Scripting failed ({e}); falling back to trace.")
    example = torch.randn(1,3,IMG_SIZE,IMG_SIZE)
    traced = torch.jit.trace(encoder_cpu, example)
    traced.save(script_path)
    print(f"Saved TorchScript (trace) to: {script_path}")

# Save to Drive
drive_dir = "/content/drive/MyDrive/Alzheimer_Disease/Models"
os.makedirs(drive_dir, exist_ok=True)
drive_state_path = os.path.join(drive_dir, "default_swinvit_encoder_state.pt")
torch.save(encoder.state_dict(), drive_state_path)
print(f"Copied weights to Drive: {drive_state_path}")
