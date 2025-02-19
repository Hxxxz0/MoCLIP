import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, CLIPTokenizer


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.2):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数下标
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数下标
        pe = pe.unsqueeze(1)  # (max_len, 1, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: (T, B, D)
        Returns:
            加上位置编码后的张量
        """
        seq_len = x.size(0)
        x = x + self.pe[:seq_len, :]
        return self.dropout(x)


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
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, motion, lengths):
        """
        Args:
            motion: (B, T, D)
            lengths: (B,)
        Returns:
            池化后的 motion embedding (B, embed_dim)
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


class EvalWarperMoClip:
    """
    评价动作和文本对齐的工具类

    初始化时可配置 OPT 中的一些参数，如 embed_dim、clip_model_name、max_length、input_dim 等。
    """
    def __init__(self,
                 model_path: str,
                 embed_dim: int = 768,
                 clip_model_name: str = "openai/clip-vit-large-patch14",
                 max_length: int = 77,
                 input_dim: int = 263,
                 max_seq_length: int = 196,
                 temperature: float = 0.07,
                 device: torch.device = None):
        """
        Args:
            model_path: 模型权重文件路径
            embed_dim: embedding 维度 (default: 768)
            clip_model_name: CLIP 模型的名称 (default: "openai/clip-vit-large-patch14")
            max_length: 文本最大 token 长度 (default: 77)
            input_dim: 动作输入维度 (default: 263)
            max_seq_length: 动作序列最大长度 (default: 196)
            temperature: 温度参数 (default: 0.07)
            device: 运算设备，若为 None 则自动选择 ("cuda" if available else "cpu")
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        # 保存相关参数
        self.embed_dim = embed_dim
        self.clip_model_name = clip_model_name
        self.max_length = max_length
        self.input_dim = input_dim
        self.max_seq_length = max_seq_length
        self.temperature = temperature
        self.model_path = model_path

        # 初始化 CLIP 模型和 tokenizer
        self.clip_model = CLIPModel.from_pretrained(self.clip_model_name)
        self.clip_tokenizer = CLIPTokenizer.from_pretrained(self.clip_model_name)
        self.clip_model.to(self.device)

        # 构建 MotionEncoder 并构造整体模型
        self.motion_encoder = MotionEncoder(
            input_dim=self.input_dim,
            embed_dim=self.embed_dim,
            num_heads=8,
            num_layers=4,
            dim_feedforward=2048,
            dropout=0.2,
            max_seq_length=self.max_seq_length
        )
        self.model = ClipMotionAlignModel(
            clip_model=self.clip_model,
            motion_encoder=self.motion_encoder,
            temperature=self.temperature
        ).to(self.device)

        # 加载预训练权重
        state_dict = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()

    def get_co_embeddings(self, captions, motions):
        """
        获取文本和动作编码

        Args:
            captions: List[str]，文本描述列表
            motions: list 或 tensor，动作数据，形状应为 (B, T, input_dim) 或者 list，
                     每个元素为 (T, input_dim) 的数组

        Returns:
            text_embeddings, motion_embeddings
        """
        # 文本预处理：转为小写
        captions_lower = [caption.lower() for caption in captions]
        text_encodings = self.clip_tokenizer(
            captions_lower,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        input_ids = text_encodings["input_ids"].to(self.device)
        attention_mask = text_encodings["attention_mask"].to(self.device)

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
            lengths_tensor = torch.tensor(lengths, dtype=torch.long, device=self.device)
        else:
            motions_tensor = motions.float().to(self.device)
            B, T, _ = motions_tensor.shape
            lengths_tensor = torch.tensor([T] * B, dtype=torch.long, device=self.device)

        # ---------------------------
        # 模型前向传播获得编码
        # ---------------------------
        with torch.no_grad():
            motion_emb, text_emb = self.model(motions_tensor, lengths_tensor, input_ids, attention_mask)

        # 对编码进行归一化（可选）
        motion_embeddings = F.normalize(motion_emb, dim=-1).cpu()
        text_embeddings = F.normalize(text_emb, dim=-1).cpu()

        return text_embeddings, motion_embeddings

    @staticmethod
    def compute_mmd(x, y, sigma=10, scale=1000):
        """
        计算 Gaussian RBF 核下的 MMD 值

        Args:
            x: numpy 数组，形状为 (n, d)
            y: numpy 数组，形状为 (m, d)
            sigma: Gaussian 核的带宽参数 (default: 10)
            scale: 缩放因子 (default: 1000)

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


# ---------------------------
# 使用示例
# ---------------------------
if __name__ == '__main__':
    # 假设模型权重文件路径为 "path_to_your_model.pt"
    warper = EvalWarperMoClip(model_path="path_to_your_model.pt")

    # 示例文本和动作数据（请根据实际数据调整）
    captions = ["A person is walking", "A person is running"]
    # 示例：每个动作数据为一个 (T, input_dim) 的列表，其中 T 可以不同
    motions = [
        [[0.1] * warper.input_dim for _ in range(50)],
        [[0.2] * warper.input_dim for _ in range(60)]
    ]

    text_embs, motion_embs = warper.get_co_embeddings(captions, motions)
    print("Text Embeddings shape:", text_embs.shape)
    print("Motion Embeddings shape:", motion_embs.shape)

    # 比如计算两个 motion embedding 间的 MMD
    mmd_value = EvalWarperMoClip.compute_mmd(motion_embs.numpy(), motion_embs.numpy())
    print("Motion MMD:", mmd_value)
