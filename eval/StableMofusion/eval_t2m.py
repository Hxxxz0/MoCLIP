# -*- coding: utf-8 -*-
"""
该文件源自 T2M(https://github.com/EricGuo5513/text-to-motion),
依据其 LICENSE (https://github.com/EricGuo5513/text-to-motion/blob/main/LICENSE) 进行分享和修改。
Copyright (c) 2022 Chuan Guo
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, CLIPTokenizer
from collections import OrderedDict
from datetime import datetime
from utils.metrics import *

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
# 设置/获取 全局 device
# ---------------------------
def set_global_device(dev):
    """
    设置全局 device（例如 'cuda:0' 或 'cpu'）
    """
    GLOBAL_CACHE["device"] = dev


def get_global_device():
    """
    获取全局 device，如未设置则默认使用 'cuda' or 'cpu'
    """
    if GLOBAL_CACHE["device"] is None:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        GLOBAL_CACHE["device"] = dev
    return GLOBAL_CACHE["device"]


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
        device = get_global_device()
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
def get_co_embeddings_2(captions, motions, model_path="./moCLIP/clip_motion_align_epoch_16.pt"):
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
    device = get_global_device()

    # ---------------------------
    # 文本处理
    # ---------------------------
    captions_lower = [caption.lower() for caption in captions]
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


# ---------------------------
# 评价指标相关函数（原封不动）
# ---------------------------
def get_metric_statistics(values, replication_times):
    mean = np.mean(values, axis=0)
    std = np.std(values, axis=0)
    conf_interval = 1.96 * std / np.sqrt(replication_times)
    return mean, conf_interval


def evaluate_matching_score(eval_wrapper, motion_loaders, file):
    
    match_score_dict = OrderedDict({})
    R_precision_dict = OrderedDict({})
    activation_dict = OrderedDict({})

    print('========== Evaluating Matching Score ==========')
    for motion_loader_name, motion_loader in motion_loaders.items():
        all_motion_embeddings = []
        all_size = 0
        matching_score_sum = 0
        top_k_count = 0
        with torch.no_grad():
            for idx, batch in enumerate(motion_loader):
                try:
                    word_embeddings, pos_one_hots, captions, sent_lens, motions, m_lens, _ = batch
                except:
                    captions, motions, m_lens = batch

                # 调用 get_co_embeddings_2 使用 clip 模型进行编码
                text_embeddings, motion_embeddings = get_co_embeddings_2(
                    captions,
                    motions,
                )
                dist_mat = euclidean_distance_matrix(text_embeddings.cpu().numpy(),
                                                     motion_embeddings.cpu().numpy())
                matching_score_sum += dist_mat.trace()

                argsmax = np.argsort(dist_mat, axis=1)
                top_k_mat = calculate_top_k(argsmax, top_k=3)
                top_k_count += top_k_mat.sum(axis=0)

                all_size += text_embeddings.shape[0]
                all_motion_embeddings.append(motion_embeddings.cpu().numpy())

        all_motion_embeddings = np.concatenate(all_motion_embeddings, axis=0)
        matching_score = matching_score_sum / all_size
        R_precision = top_k_count / all_size
        match_score_dict[motion_loader_name] = matching_score
        R_precision_dict[motion_loader_name] = R_precision
        activation_dict[motion_loader_name] = all_motion_embeddings

        print(f'---> [{motion_loader_name}] Matching Score: {matching_score:.4f}')
        print(f'---> [{motion_loader_name}] Matching Score: {matching_score:.4f}', file=file, flush=True)

        line = f'---> [{motion_loader_name}] R_precision: '
        for i in range(len(R_precision)):
            line += '(top %d): %.4f ' % (i+1, R_precision[i])
        print(line)
        print(line, file=file, flush=True)

    return match_score_dict, R_precision_dict, activation_dict


def evaluate_fid(eval_wrapper, groundtruth_loader, activation_dict, file):
    
    eval_dict = OrderedDict({})
    print('========== Evaluating FID ==========')
    gt_motion_embeddings = []
    with torch.no_grad():
        for idx, batch in enumerate(groundtruth_loader):
            _, _, _, sent_lens, motions, m_lens, _ = batch
            # 这里使用 get_co_embeddings_2 提取 ground truth motion embedding（确保使用 clip）
            text_embeddings, motion_embeddings = get_co_embeddings_2(
                captions=["dummy"] * motions.shape[0],  # 如果没有真实文本，可传入dummy
                motions=motions
            )
            gt_motion_embeddings.append(motion_embeddings.cpu().numpy())
    gt_motion_embeddings = np.concatenate(gt_motion_embeddings, axis=0)
    gt_mu, gt_cov = calculate_activation_statistics(gt_motion_embeddings)

    for model_name, motion_embeddings in activation_dict.items():
        mu, cov = calculate_activation_statistics(motion_embeddings)
        fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)
        print(f'---> [{model_name}] FID: {fid:.4f}')
        print(f'---> [{model_name}] FID: {fid:.4f}', file=file, flush=True)
        eval_dict[model_name] = fid
    return eval_dict


def evaluate_diversity(activation_dict, file, diversity_times):
    
    eval_dict = OrderedDict({})
    print('========== Evaluating Diversity ==========')
    for model_name, motion_embeddings in activation_dict.items():
        diversity = calculate_diversity(motion_embeddings, diversity_times)
        eval_dict[model_name] = diversity
        print(f'---> [{model_name}] Diversity: {diversity:.4f}')
        print(f'---> [{model_name}] Diversity: {diversity:.4f}', file=file, flush=True)
    return eval_dict


def evaluate_multimodality(eval_wrapper, mm_motion_loaders, file, mm_num_times):
    
    eval_dict = OrderedDict({})
    print('========== Evaluating MultiModality ==========')
    for model_name, mm_motion_loader in mm_motion_loaders.items():
        mm_motion_embeddings = []
        with torch.no_grad():
            for idx, batch in enumerate(mm_motion_loader):
                # (1, mm_replications, dim_pos)
                motions, m_lens = batch
                motion_embedings = eval_wrapper.get_motion_embeddings(motions[0], m_lens[0])
                mm_motion_embeddings.append(motion_embedings.unsqueeze(0))
        if len(mm_motion_embeddings) == 0:
            multimodality = 0
        else:
            mm_motion_embeddings = torch.cat(mm_motion_embeddings, dim=0).cpu().numpy()
            multimodality = calculate_multimodality(mm_motion_embeddings, mm_num_times)
        print(f'---> [{model_name}] Multimodality: {multimodality:.4f}')
        print(f'---> [{model_name}] Multimodality: {multimodality:.4f}', file=file, flush=True)
        eval_dict[model_name] = multimodality
    return eval_dict


# ---------------------------
# 新增：计算 MMD 部分（使用 clip 模型提取 motion embedding）
# ---------------------------
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


def evaluate_mmd(groundtruth_loader, activation_dict, eval_wrapper, file):
    """
    计算 ground truth motion 与各模型生成 motion 之间的 MMD 值，均使用 clip 模型提取 embedding

    Args:
         groundtruth_loader: ground truth 数据加载器
         activation_dict: dict，键为模型名称，值为该模型生成的 motion embedding（numpy 数组）
         eval_wrapper: 用于获得 motion embedding 的评估封装器（这里为了保持 clip 一致，我们通过 get_co_embeddings_2 提取）
         file: 日志文件句柄，用于写日志
    Returns:
         OrderedDict，记录各模型的 motion MMD 值
    """
    print('========== Evaluating Motion MMD ==========')
    gt_motion_embeddings = []
    with torch.no_grad():
        for idx, batch in enumerate(groundtruth_loader):
            try:
                # 如果有 captions，取出 captions
                _, _, captions, sent_lens, motions, m_lens, _ = batch
            except:
                # 否则假设 batch 直接返回 (captions, motions, m_lens)
                captions, motions, m_lens = batch
            # 使用 clip 模型提取 ground truth motion embedding
            _, motion_embeddings = get_co_embeddings_2(captions, motions)
            gt_motion_embeddings.append(motion_embeddings.cpu().numpy())
    gt_motion_embeddings = np.concatenate(gt_motion_embeddings, axis=0)
    mmd_dict = OrderedDict({})
    for model_name, motion_embeddings in activation_dict.items():
        mmd_val = compute_mmd(gt_motion_embeddings, motion_embeddings, sigma=10, scale=1000)
        mmd_dict[model_name] = mmd_val
        print(f'---> [{model_name}] Motion MMD: {mmd_val:.4f}')
        print(f'---> [{model_name}] Motion MMD: {mmd_val:.4f}', file=file, flush=True)
    return mmd_dict


# ---------------------------
# 下面仅保留示例的 evaluation 函数（评估 R_precision 和 Motion MMD）
# ---------------------------
def evaluation(eval_wrapper, gt_loader, eval_motion_loaders, log_file, replication_times,
               diversity_times, mm_num_times, device, run_mm=False):
    """
    评估函数：示例评估 R_precision 和 Motion MMD 指标

    参数说明：
      - eval_wrapper: 评估时所用的封装器/模型
      - gt_loader: ground truth 数据加载器
      - eval_motion_loaders: 一个字典，其键为模型名称，值为一个函数，调用后返回 (motion_loader, mm_motion_loader, gen_time)
      - log_file: 日志文件路径
      - replication_times: 复制实验次数
      - diversity_times, mm_num_times, run_mm: 兼容原接口，但本示例函数中不使用
      - device: 传入的 GPU/CPU device (如 'cuda:0')
    """
    # 设置全局device
    set_global_device(device)

    with open(log_file, 'a') as f:
        all_metrics = OrderedDict({
            'R_precision': OrderedDict(),
            'Motion MMD': OrderedDict()
        })

        for replication in range(replication_times):
            print(f'\nTime: {datetime.now()}')
            print(f'\nTime: {datetime.now()}', file=f, flush=True)
            
            # 初始化 loader，ground truth loader 放入 motion_loaders 字典
            motion_loaders = {'ground truth': gt_loader}
            
            # 加载生成的动作数据
            for name, getter in eval_motion_loaders.items():
                motion_loader, _, gen_time = getter()
                motion_loaders[name] = motion_loader
                print(f'---> [{name}] Generation time: {gen_time:.2f}s', file=f, flush=True)

            if replication_times > 1:
                rep_msg = f'\n==================== Replication {replication+1}/{replication_times} ===================='
                print(rep_msg)
                print(rep_msg, file=f, flush=True)

            # 计算 R_precision
            _, R_dict, activation_dict = evaluate_matching_score(eval_wrapper, motion_loaders, f)

            # 收集 R_precision
            for model_name, value in R_dict.items():
                if model_name not in all_metrics['R_precision']:
                    all_metrics['R_precision'][model_name] = []
                all_metrics['R_precision'][model_name].append(value)

            # 计算 Motion MMD
            mmd_dict = evaluate_mmd(gt_loader, activation_dict, eval_wrapper, f)
            for model_name, mmd_value in mmd_dict.items():
                if model_name not in all_metrics['Motion MMD']:
                    all_metrics['Motion MMD'][model_name] = []
                all_metrics['Motion MMD'][model_name].append(mmd_value)

        # 统计最终指标
        final_metrics = OrderedDict()
        final_msg = '\n\n==================== Final Results ===================='
        print(final_msg)
        print(final_msg, file=f, flush=True)
        
        # 输出 R_precision
        for model_name, values in all_metrics['R_precision'].items():
            values = np.array(values)
            mean = np.mean(values, axis=0)
            std = np.std(values, axis=0)
            conf_interval = 1.96 * std / np.sqrt(replication_times)
            
            if values.ndim == 1:
                final_metrics[f'R_precision_{model_name}'] = mean
                line = f'---> [{model_name}] R_precision Mean: {mean:.4f} CInterval: {conf_interval:.4f}'
            else:
                final_metrics[f'R_precision_{model_name}'] = mean
                line = '---> [{}] '.format(model_name) + '; '.join(
                    [f'(top {i+1}) Mean: {m:.4f} CInterval: {c:.4f}'
                     for i, (m, c) in enumerate(zip(mean, conf_interval))]
                )
            print(line)
            print(line, file=f, flush=True)
        
        # 输出 Motion MMD
        for model_name, values in all_metrics['Motion MMD'].items():
            values = np.array(values)
            mean = np.mean(values, axis=0)
            std = np.std(values, axis=0)
            conf_interval = 1.96 * std / np.sqrt(replication_times)
            
            final_metrics[f'Motion_MMD_{model_name}'] = mean
            line = f'---> [{model_name}] Motion MMD Mean: {mean:.4f} CInterval: {conf_interval:.4f}'
            print(line)
            print(line, file=f, flush=True)
        
        return final_metrics

