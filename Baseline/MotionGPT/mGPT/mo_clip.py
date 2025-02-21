import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, CLIPTokenizer
from collections import OrderedDict
from datetime import datetime

# ---------------------------
# 全局缓存：只在第一次时加载模型，后续重复使用
# ---------------------------
GLOBAL_CACHE = {
    "clip_model": None,
    "clip_tokenizer": None,
    "motion_encoder": None,
    "clip_motion_align_model": None,
    "device": None
}

# ---------------------------
# PositionalEncoding 定义
# ---------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.2):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数下标
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数下标
        pe = pe.unsqueeze(1)  # (max_len, 1, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: (T, B, D)
        """
        seq_len = x.size(0)
        x = x + self.pe[:seq_len, :]
        return self.dropout(x)


# ---------------------------
# MotionEncoder 定义
# ---------------------------
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


# ---------------------------
# ClipMotionAlignModel 定义
# ---------------------------
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


def _init_clip_motion_model(model_path):
    """
    仅在全局缓存为空时，初始化 CLIP模型、tokenizer、MotionEncoder 并加载预训练权重。
    之后存入 GLOBAL_CACHE，供后续重复使用。
    """
    if GLOBAL_CACHE["clip_motion_align_model"] is not None:
        # 已经加载过，直接返回
        return

    # 根据原始训练的配置定义
    class OPT:
        embed_dim = 768
        # 此处从全局获取 device
        device = 'cuda'
        clip_model_name = "openai/clip-vit-large-patch14"
        max_length = 77
        input_dim = 263
        max_seq_length = 196

    # 加载 CLIP 模型和 tokenizer
    clip_model = CLIPModel.from_pretrained(OPT.clip_model_name)
    clip_tokenizer = CLIPTokenizer.from_pretrained(OPT.clip_model_name)
    clip_model.to(OPT.device)

    # 构建 MotionEncoder 与整体模型
    motion_encoder = MotionEncoder(
        input_dim=OPT.input_dim,
        embed_dim=OPT.embed_dim,
        num_heads=8,
        num_layers=4,
        dim_feedforward=2048,
        dropout=0.2,
        max_seq_length=OPT.max_seq_length
    )

    model = ClipMotionAlignModel(
        clip_model=clip_model,
        motion_encoder=motion_encoder,
        temperature=0.07
    ).to(OPT.device)

    # 加载预训练权重
    state_dict = torch.load(model_path, map_location=OPT.device)
    model.load_state_dict(state_dict)
    model.eval()

    # 缓存到全局变量
    GLOBAL_CACHE["clip_model"] = clip_model
    GLOBAL_CACHE["clip_tokenizer"] = clip_tokenizer
    GLOBAL_CACHE["motion_encoder"] = motion_encoder
    GLOBAL_CACHE["clip_motion_align_model"] = model


# ---------------------------
# 定义获取文本与动作编码的函数（保持原接口）
# ---------------------------
def get_co_embeddings_2(captions, motions, model_path="cma/clip_motion_align_epoch_21.pt"):
    """
    参数：
        captions: List[str]，文本描述列表
        motions: list 或 tensor，动作数据，形状应为 (B, T, input_dim) 或者 list，每个元素为 (T, input_dim) 的数组
        model_path: 模型权重文件路径
    返回：
        text_embeddings, motion_embeddings
    """
    # 如果全局模型还未初始化，则进行初始化（只执行一次）
    _init_clip_motion_model(model_path)

    # 取出已缓存的模型、tokenizer、device
    clip_motion_model = GLOBAL_CACHE["clip_motion_align_model"]
    clip_tokenizer = GLOBAL_CACHE["clip_tokenizer"]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ---------------------------
    # 文本处理
    # ---------------------------



    captions_lower = [caption[0].lower() for caption in captions]

    text_encodings = clip_tokenizer(
        captions_lower,
        padding=True,
        truncation=True,
        max_length=77,  # 与原 OPT.max_length 保持一致
        return_tensors="pt"
    )
    input_ids = text_encodings["input_ids"].to(device)
    attention_mask = text_encodings["attention_mask"].to(device)

    # ---------------------------
    # 动作数据处理
    # ---------------------------
    if isinstance(motions, list):
        motion_tensors = []
        lengths = []
        for m in motions:
            m_tensor = torch.tensor(m, dtype=torch.float32)
            motion_tensors.append(m_tensor)
            lengths.append(m_tensor.shape[0])

        max_T = max(lengths)
        padded_motions = []
        for m_tensor in motion_tensors:
            T = m_tensor.shape[0]
            if T < max_T:
                pad = torch.zeros((max_T - T, m_tensor.shape[1]), dtype=torch.float32)
                m_tensor = torch.cat([m_tensor, pad], dim=0)
            padded_motions.append(m_tensor)
        motions_tensor = torch.stack(padded_motions, dim=0)
        lengths_tensor = torch.tensor(lengths, dtype=torch.long, device=device)
    else:
        motions_tensor = motions.float().to(device)
        B, T, _ = motions_tensor.shape
        lengths_tensor = torch.tensor([T] * B, dtype=torch.long, device=device)

    # ---------------------------
    # 模型前向传播，获得编码
    # ---------------------------
    with torch.no_grad():
        motion_emb, text_emb = clip_motion_model(motions_tensor, lengths_tensor, input_ids, attention_mask)

    # 对编码进行归一化（可选）
    motion_embeddings = F.normalize(motion_emb, dim=-1).cpu()
    text_embeddings = F.normalize(text_emb, dim=-1).cpu()

    return text_embeddings, motion_embeddings

def compute_mmd(x, y, sigma=10, scale=1000):
    """
    计算 Gaussian RBF 核下的 MMD 值（针对 motion embedding）

    Args:
        x: numpy 数组，形状为 (n, d)
        y: numpy 数组，形状为 (m, d)
        sigma: Gaussian 核的带宽参数（默认 10）
        scale: 缩放因子（默认 1000），使结果更直观
    Returns:
        MMD 值（float）
    """
    gamma = 1.0 / (2 * sigma**2)
    x_norm = np.sum(x**2, axis=1, keepdims=True)  # (n, 1)
    y_norm = np.sum(y**2, axis=1, keepdims=True)  # (m, 1)
    K_xx = np.exp(-gamma * (x_norm + x_norm.T - 2 * np.dot(x, x.T)))
    K_yy = np.exp(-gamma * (y_norm + y_norm.T - 2 * np.dot(y, y.T)))
    K_xy = np.exp(-gamma * (x_norm + y_norm.T - 2 * np.dot(x, y.T)))
    mmd_val = scale * (np.mean(K_xx) + np.mean(K_yy) - 2 * np.mean(K_xy))
    return mmd_val