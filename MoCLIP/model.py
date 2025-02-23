import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
from transformers import CLIPModel, CLIPTokenizer
from motion_loader import get_dataset_loader  
from tqdm import tqdm
import random

def evaluate_model(model, test_loader, clip_tokenizer, opt, desc="Test", sample_size=32):
    model.eval()
    total_loss = 0.0
    count = 0

    all_motion_embs = []
    all_text_embs = []

    with torch.no_grad():
        for batch_data in tqdm(test_loader, desc=desc):
            caption, motion, m_length = batch_data

            # 文本 tokenization
            caption = [c.lower() for c in caption]
            text_enc = clip_tokenizer(
                caption,
                padding=True,
                truncation=True,
                max_length=opt.max_length,
                return_tensors="pt"
            )
            input_ids = text_enc["input_ids"].to(opt.device)
            attention_mask = text_enc["attention_mask"].to(opt.device)

            # motion 数据处理
            if isinstance(motion, list):
                motion = torch.stack([torch.tensor(m, dtype=torch.float32) for m in motion], dim=0)
            else:
                motion = motion.float()
            motion = motion.to(opt.device)
            m_length = m_length.to(opt.device)

            # 获取 motion 与 text 的 embedding
            motion_emb, text_emb = model(motion, m_length, input_ids, attention_mask)
            loss = clip_contrastive_loss(motion_emb, text_emb, model.logit_scale)
            total_loss += loss.item()
            count += 1

            all_motion_embs.append(motion_emb.cpu())
            all_text_embs.append(text_emb.cpu())

    avg_loss = total_loss / max(count, 1)
    print(f"{desc} Average Contrastive Loss: {avg_loss:.4f}")

    # 拼接所有样本的 embedding，并归一化
    motion_embs = F.normalize(torch.cat(all_motion_embs, dim=0), dim=-1)
    text_embs   = F.normalize(torch.cat(all_text_embs, dim=0), dim=-1)

    # 按 sample_size 划分批次计算检索指标
    num_samples = motion_embs.shape[0]
    if num_samples >= sample_size:
        num_full_samples = (num_samples // sample_size) * sample_size
        motion_embs = motion_embs[:num_full_samples]
        text_embs   = text_embs[:num_full_samples]
        num_batches = num_full_samples // sample_size
    else:
        num_batches = 1

    m2t_r1_list, m2t_r2_list, m2t_r3_list = [], [], []
    t2m_r1_list, t2m_r2_list, t2m_r3_list = [], [], []

    for i in range(num_batches):
        start_idx = i * sample_size
        end_idx = (i + 1) * sample_size
        batch_motion = motion_embs[start_idx:end_idx]
        batch_text = text_embs[start_idx:end_idx]

        sim_matrix = batch_motion @ batch_text.t()
        N = sim_matrix.size(0)

        # Motion -> Text 检索
        ranks = []
        for j in range(N):
            sim_row = sim_matrix[j]
            sorted_idx = torch.argsort(sim_row, descending=True)
            rank = (sorted_idx == j).nonzero(as_tuple=True)[0].item()
            ranks.append(rank)
        ranks = torch.tensor(ranks)
        m2t_r1 = (ranks < 1).float().mean().item()
        m2t_r2 = (ranks < 2).float().mean().item()
        m2t_r3 = (ranks < 3).float().mean().item()
        m2t_r1_list.append(m2t_r1)
        m2t_r2_list.append(m2t_r2)
        m2t_r3_list.append(m2t_r3)

        # Text -> Motion 检索
        ranks_t2m = []
        for j in range(N):
            sim_col = sim_matrix[:, j]
            sorted_idx = torch.argsort(sim_col, descending=True)
            rank = (sorted_idx == j).nonzero(as_tuple=True)[0].item()
            ranks_t2m.append(rank)
        ranks_t2m = torch.tensor(ranks_t2m)
        t2m_r1 = (ranks_t2m < 1).float().mean().item()
        t2m_r2 = (ranks_t2m < 2).float().mean().item()
        t2m_r3 = (ranks_t2m < 3).float().mean().item()
        t2m_r1_list.append(t2m_r1)
        t2m_r2_list.append(t2m_r2)
        t2m_r3_list.append(t2m_r3)

    avg_m2t_r1 = sum(m2t_r1_list) / len(m2t_r1_list)
    avg_m2t_r2 = sum(m2t_r2_list) / len(m2t_r2_list)
    avg_m2t_r3 = sum(m2t_r3_list) / len(m2t_r3_list)
    avg_t2m_r1 = sum(t2m_r1_list) / len(t2m_r1_list)
    avg_t2m_r2 = sum(t2m_r2_list) / len(t2m_r2_list)
    avg_t2m_r3 = sum(t2m_r3_list) / len(t2m_r3_list)

    print(f"{desc} M->T Retrieval (per {sample_size} samples): R@1={avg_m2t_r1:.3f}, R@2={avg_m2t_r2:.3f}, R@3={avg_m2t_r3:.3f}")
    print(f"{desc} T->M Retrieval (per {sample_size} samples): R@1={avg_t2m_r1:.3f}, R@2={avg_t2m_r2:.3f}, R@3={avg_t2m_r3:.3f}")

    model.train()
    return avg_loss

# --------------------------------------------------------
# PositionalEncoding
# --------------------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.2):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  
        pe[:, 1::2] = torch.cos(position * div_term)  
        pe = pe.unsqueeze(1)  # (max_len, 1, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: (T, B, D)
        """
        seq_len = x.size(0)
        x = x + self.pe[:seq_len, :]
        return self.dropout(x)

# --------------------------------------------------------
# MotionEncoder (4层+较大 dropout + mask)
# --------------------------------------------------------
class MotionEncoder(nn.Module):
    def __init__(self, input_dim=263, embed_dim=512, num_heads=8, num_layers=4,
                 dim_feedforward=2048, dropout=0.2, max_seq_length=196):
        super(MotionEncoder, self).__init__()
        self.embed_dim = embed_dim
        self.input_proj = nn.Linear(input_dim, embed_dim)
        self.pos_encoder = PositionalEncoding(d_model=embed_dim, max_len=max_seq_length, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, motion, lengths):
        """
        motion: (B, T, D)
        lengths: (B,)
        """
        x = self.input_proj(motion).transpose(0, 1)  # (T, B, embed_dim)

        x = self.pos_encoder(x)                      # (T, B, embed_dim)

        # 构造 padding mask
        B, T = motion.size(0), motion.size(1)
        device = motion.device
        pad_mask = torch.zeros((B, T), dtype=torch.bool, device=device)
        for i, length in enumerate(lengths):
            if length < T:
                pad_mask[i, length:] = True

        x = self.transformer_encoder(x, src_key_padding_mask=pad_mask)  # (T, B, embed_dim)
        x = x.transpose(0, 1)  # (B, T, embed_dim)

        pooled_list = []
        for i in range(B):
            valid_len = lengths[i]
            if valid_len > 0:
                pooled_list.append(x[i, :valid_len].mean(dim=0))
            else:
                pooled_list.append(torch.zeros(self.embed_dim, device=device))
        pooled = torch.stack(pooled_list, dim=0)  # (B, embed_dim)
        pooled = self.dropout(pooled)
        pooled = self.fc(pooled)  # (B, embed_dim)
        return pooled

# --------------------------------------------------------
# ClipMotionAlignModel
# --------------------------------------------------------
class ClipMotionAlignModel(nn.Module):
    def __init__(self, clip_model: CLIPModel, motion_encoder: nn.Module, temperature=0.07):
        super().__init__()
        self.clip_model = clip_model
        self.motion_encoder = motion_encoder
        # 初始化 logit_scale = log(1/temperature)
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1.0 / temperature)))

    def forward(self, motion, lengths, input_ids, attention_mask):
        motion_emb = self.motion_encoder(motion, lengths)
        text_emb = self.clip_model.get_text_features(input_ids=input_ids, attention_mask=attention_mask)
        return motion_emb, text_emb

# --------------------------------------------------------
# clip_contrastive_loss
# --------------------------------------------------------
def clip_contrastive_loss(motion_emb, text_emb, logit_scale):
    logit_scale = logit_scale.exp().clamp(max=20)
    motion_emb = F.normalize(motion_emb, dim=-1)
    text_emb   = F.normalize(text_emb, dim=-1)
    logits_per_motion = motion_emb @ text_emb.t() * logit_scale
    logits_per_text   = text_emb @ motion_emb.t() * logit_scale
    B = motion_emb.size(0)
    ground_truth = torch.arange(B, device=motion_emb.device)
    loss_m2t = F.cross_entropy(logits_per_motion, ground_truth)
    loss_t2m = F.cross_entropy(logits_per_text,   ground_truth)
    return (loss_m2t + loss_t2m) * 0.5

