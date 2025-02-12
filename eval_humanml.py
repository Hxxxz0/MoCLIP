from utils.parser_util import evaluation_parser
from utils.fixseed import fixseed
from datetime import datetime
from data_loaders.humanml.motion_loaders.model_motion_loaders import get_mdm_loader  # get_motion_loader
from data_loaders.humanml.utils.metrics import *
from data_loaders.humanml.networks.evaluator_wrapper import EvaluatorMDMWrapper
from collections import OrderedDict
from data_loaders.humanml.scripts.motion_process import *
from data_loaders.humanml.utils.utils import *
from utils.model_util import create_model_and_diffusion, load_model_wo_clip

from diffusion import logger
from utils import dist_util
from data_loaders.get_data import get_dataset_loader
from model.cfg_sampler import ClassifierFreeSampleModel

import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, CLIPTokenizer
from collections import OrderedDict
from datetime import datetime

torch.multiprocessing.set_sharing_strategy('file_system')

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
# 添加 motion MMD 计算函数
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

def evaluate_motion_mmd(gt_loader, activation_dict, file):
    """
    计算 ground truth motion 与各模型生成 motion 之间的 MMD 值

    Args:
         gt_loader: ground truth 数据的 DataLoader
         activation_dict: dict，键为模型名称，值为该模型生成的 motion embedding（numpy 数组）
         file: 日志文件句柄，用于写日志
    Returns:
         OrderedDict，记录各模型的 motion MMD 值
    """
    print('========== Evaluating Motion MMD ==========')
    # 计算 ground truth motion 的 embedding
    gt_motion_embeddings = []
    with torch.no_grad():
        for idx, batch in enumerate(gt_loader):
            try:
                # 有些 loader 返回多个值
                word_embeddings, pos_one_hots, captions, sent_lens, motions, m_lens, _ = batch
            except:
                # 如果返回的值较少，则直接解包
                captions, motions, m_lens = batch
            # 这里仅需要 motion 的 embedding，调用 get_co_embeddings_2 接口
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
    #model.load_state_dict(state_dict)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    # 缓存到全局变量
    GLOBAL_CACHE["clip_model"] = clip_model
    GLOBAL_CACHE["clip_tokenizer"] = clip_tokenizer
    GLOBAL_CACHE["motion_encoder"] = motion_encoder
    GLOBAL_CACHE["clip_motion_align_model"] = model


# ---------------------------
# 定义获取文本与动作编码的函数（保持原接口）
# ---------------------------
def get_co_embeddings_2(captions, motions, model_path="/home/user/dxc/motion/StableMoFusion/clip_motion_align_epoch_18.pt"):
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


def evaluate_matching_score(eval_wrapper, motion_loaders, file):
    match_score_dict = OrderedDict({})
    R_precision_dict = OrderedDict({})
    activation_dict = OrderedDict({})
    print('========== Evaluating Matching Score ==========')
    for motion_loader_name, motion_loader in motion_loaders.items():
        all_motion_embeddings = []
        score_list = []
        all_size = 0
        matching_score_sum = 0
        top_k_count = 0
        # print(motion_loader_name)
        with torch.no_grad():
            for idx, batch in enumerate(motion_loader):
                try:
                    word_embeddings, pos_one_hots, captions, sent_lens, motions, m_lens, _ = batch
                except:
                    captions, motions, m_lens = batch

                # 这里保持调用 get_co_embeddings_2 的原接口，不做修改
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


# def evaluate_fid(eval_wrapper, groundtruth_loader, activation_dict, file):
#     eval_dict = OrderedDict({})
#     gt_motion_embeddings = []
#     print('========== Evaluating FID ==========')
#     with torch.no_grad():
#         for idx, batch in enumerate(groundtruth_loader):
#             word_embeddings, pos_one_hots, captions, sent_lens, motions, m_lens, _ = batch

#             text_embeddings, motion_embeddings = get_co_embeddings_2(
#                     captions,
#                     motions,
#                 )
            

#             gt_motion_embeddings.append(motion_embeddings.cpu().numpy())
#     gt_motion_embeddings = np.concatenate(gt_motion_embeddings, axis=0)
#     gt_mu, gt_cov = calculate_activation_statistics(gt_motion_embeddings)

#     for model_name, motion_embeddings in activation_dict.items():
#         mu, cov = calculate_activation_statistics(motion_embeddings)
#         fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)
#         print(f'---> [{model_name}] FID: {fid:.4f}')
#         print(f'---> [{model_name}] FID: {fid:.4f}', file=file, flush=True)
#         eval_dict[model_name] = fid
#     return eval_dict

def evaluate_matching_score_2(eval_wrapper, motion_loaders, file):
    match_score_dict = OrderedDict({})
    R_precision_dict = OrderedDict({})
    activation_dict = OrderedDict({})
    print('========== Evaluating Matching Score ==========')
    for motion_loader_name, motion_loader in motion_loaders.items():
        all_motion_embeddings = []
        score_list = []
        all_size = 0
        matching_score_sum = 0
        top_k_count = 0
        # print(motion_loader_name)
        with torch.no_grad():
            for idx, batch in enumerate(motion_loader):
                word_embeddings, pos_one_hots, _, sent_lens, motions, m_lens, _ = batch
                text_embeddings, motion_embeddings = eval_wrapper.get_co_embeddings(
                    word_embs=word_embeddings,
                    pos_ohot=pos_one_hots,
                    cap_lens=sent_lens,
                    motions=motions,
                    m_lens=m_lens
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


        line = f'---> [{motion_loader_name}] R_precision: '
        for i in range(len(R_precision)):
            line += '(top %d): %.4f ' % (i+1, R_precision[i])
        print(line)
        print(line, file=file, flush=True)

    return match_score_dict, R_precision_dict, activation_dict



def evaluate_fid(eval_wrapper, groundtruth_loader, activation_dict, file):
    eval_dict = OrderedDict({})
    gt_motion_embeddings = []
    print('========== Evaluating FID ==========')
    with torch.no_grad():
        for idx, batch in enumerate(groundtruth_loader):
            _, _, _, sent_lens, motions, m_lens, _ = batch
            motion_embeddings = eval_wrapper.get_motion_embeddings(
                motions=motions,
                m_lens=m_lens
            )
            gt_motion_embeddings.append(motion_embeddings.cpu().numpy())
    gt_motion_embeddings = np.concatenate(gt_motion_embeddings, axis=0)
    gt_mu, gt_cov = calculate_activation_statistics(gt_motion_embeddings)

    # print(gt_mu)
    for model_name, motion_embeddings in activation_dict.items():
        mu, cov = calculate_activation_statistics(motion_embeddings)
        # print(mu)
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


def get_metric_statistics(values, replication_times):
    mean = np.mean(values, axis=0)
    std = np.std(values, axis=0)
    conf_interval = 1.96 * std / np.sqrt(replication_times)
    return mean, conf_interval


def evaluation(eval_wrapper, gt_loader, eval_motion_loaders, log_file, replication_times, diversity_times, mm_num_times, run_mm=False):
    with open(log_file, 'w') as f:
        all_metrics = OrderedDict({
            'Matching Score': OrderedDict({}),
            'R_precision': OrderedDict({}),
            'FID': OrderedDict({}),
            'Diversity': OrderedDict({}),
            'MultiModality': OrderedDict({}),
            'Motion MMD': OrderedDict({})
        })
        for replication in range(replication_times):
            motion_loaders = {}
            mm_motion_loaders = {}
            motion_loaders['ground truth'] = gt_loader
            for motion_loader_name, motion_loader_getter in eval_motion_loaders.items():
                motion_loader, mm_motion_loader = motion_loader_getter()
                motion_loaders[motion_loader_name] = motion_loader
                mm_motion_loaders[motion_loader_name] = mm_motion_loader

            print(f'==================== Replication {replication} ====================')
            print(f'==================== Replication {replication} ====================', file=f, flush=True)
            print(f'Time: {datetime.now()}')
            print(f'Time: {datetime.now()}', file=f, flush=True)
            mat_score_dict, R_precision_dict, acti_dict = evaluate_matching_score(eval_wrapper, motion_loaders, f)
            mat_score_dict_2, R_precision_dict_2, acti_dict_2 = evaluate_matching_score_2(eval_wrapper, motion_loaders, f)

            print(f'Time: {datetime.now()}')
            print(f'Time: {datetime.now()}', file=f, flush=True)
            fid_score_dict = evaluate_fid(eval_wrapper, gt_loader, acti_dict_2, f)

            print(f'Time: {datetime.now()}')
            print(f'Time: {datetime.now()}', file=f, flush=True)
            div_score_dict = evaluate_diversity(acti_dict, f, diversity_times)

            if run_mm:
                print(f'Time: {datetime.now()}')
                print(f'Time: {datetime.now()}', file=f, flush=True)
                mm_score_dict = evaluate_multimodality(eval_wrapper, mm_motion_loaders, f, mm_num_times)

            # ---------------------------
            # 新增：计算 Motion MMD
            # ---------------------------
            motion_mmd_dict = evaluate_motion_mmd(gt_loader, acti_dict, f)

            print(f'!!! DONE !!!')
            print(f'!!! DONE !!!', file=f, flush=True)

            for key, item in mat_score_dict.items():
                if key not in all_metrics['Matching Score']:
                    all_metrics['Matching Score'][key] = [item]
                else:
                    all_metrics['Matching Score'][key] += [item]

            for key, item in R_precision_dict.items():
                if key not in all_metrics['R_precision']:
                    all_metrics['R_precision'][key] = [item]
                else:
                    all_metrics['R_precision'][key] += [item]

            for key, item in fid_score_dict.items():
                if key not in all_metrics['FID']:
                    all_metrics['FID'][key] = [item]
                else:
                    all_metrics['FID'][key] += [item]

            for key, item in div_score_dict.items():
                if key not in all_metrics['Diversity']:
                    all_metrics['Diversity'][key] = [item]
                else:
                    all_metrics['Diversity'][key] += [item]
            if run_mm:
                for key, item in mm_score_dict.items():
                    if key not in all_metrics['MultiModality']:
                        all_metrics['MultiModality'][key] = [item]
                    else:
                        all_metrics['MultiModality'][key] += [item]
            for key, item in motion_mmd_dict.items():
                if key not in all_metrics['Motion MMD']:
                    all_metrics['Motion MMD'][key] = [item]
                else:
                    all_metrics['Motion MMD'][key] += [item]

        # Summary 输出各指标的均值及置信区间
        mean_dict = {}
        for metric_name, metric_dict in all_metrics.items():
            print('========== %s Summary ==========' % metric_name)
            print('========== %s Summary ==========' % metric_name, file=f, flush=True)
            for model_name, values in metric_dict.items():
                mean, conf_interval = get_metric_statistics(np.array(values), replication_times)
                mean_dict[metric_name + '_' + model_name] = mean
                if isinstance(mean, np.float64) or isinstance(mean, np.float32):
                    print(f'---> [{model_name}] Mean: {mean:.4f} CInterval: {conf_interval:.4f}')
                    print(f'---> [{model_name}] Mean: {mean:.4f} CInterval: {conf_interval:.4f}', file=f, flush=True)
                elif isinstance(mean, np.ndarray):
                    line = f'---> [{model_name}]'
                    for i in range(len(mean)):
                        line += '(top %d) Mean: %.4f CInt: %.4f;' % (i+1, mean[i], conf_interval[i])
                    print(line)
                    print(line, file=f, flush=True)
        return mean_dict


if __name__ == '__main__':
    args = evaluation_parser()
    fixseed(args.seed)
    args.batch_size = 32  # This must be 32! Don't change it! otherwise it will cause a bug in R precision calc!
    name = os.path.basename(os.path.dirname(args.model_path))
    niter = os.path.basename(args.model_path).replace('model', '').replace('.pt', '')
    log_file = os.path.join(os.path.dirname(args.model_path), 'eval_humanml_{}_{}'.format(name, niter))
    if args.guidance_param != 1.:
        log_file += f'_gscale{args.guidance_param}'
    log_file += f'_{args.eval_mode}'
    log_file += '.log'

    print(f'Will save to log file [{log_file}]')

    print(f'Eval mode [{args.eval_mode}]')
    if args.eval_mode == 'debug':
        num_samples_limit = 1000  # None means no limit (eval over all dataset)
        run_mm = False
        mm_num_samples = 0
        mm_num_repeats = 0
        mm_num_times = 0
        diversity_times = 300
        replication_times = 5  # about 3 Hrs
    elif args.eval_mode == 'wo_mm':
        num_samples_limit = 1000
        run_mm = False
        mm_num_samples = 0
        mm_num_repeats = 0
        mm_num_times = 0
        diversity_times = 300
        replication_times = 20 # about 12 Hrs
    elif args.eval_mode == 'mm_short':
        num_samples_limit = 1000
        run_mm = True
        mm_num_samples = 100
        mm_num_repeats = 30
        mm_num_times = 10
        diversity_times = 300
        replication_times = 5  # about 15 Hrs
    else:
        raise ValueError()

    dist_util.setup_dist(args.device)
    set_global_device(dist_util.dev())
    logger.configure()

    logger.log("creating data loader...")
    split = 'test'
    gt_loader = get_dataset_loader(name=args.dataset, batch_size=args.batch_size, num_frames=None, split=split, hml_mode='gt')
    gen_loader = get_dataset_loader(name=args.dataset, batch_size=args.batch_size, num_frames=None, split=split, hml_mode='eval')
    num_actions = gen_loader.dataset.num_actions

    logger.log("Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(args, gen_loader)

    logger.log(f"Loading checkpoints from [{args.model_path}]...")
    state_dict = torch.load(args.model_path, map_location='cpu')
    load_model_wo_clip(model, state_dict)

    if args.guidance_param != 1:
        model = ClassifierFreeSampleModel(model)   # wrapping model with the classifier-free sampler
    model.to(dist_util.dev())
    model.eval()  # disable random masking

    eval_motion_loaders = {
        ################
        ## HumanML3D Dataset##
        ################
        'vald': lambda: get_mdm_loader(
            model, diffusion, args.batch_size,
            gen_loader, mm_num_samples, mm_num_repeats, gt_loader.dataset.opt.max_motion_length, num_samples_limit, args.guidance_param
        )
    }

    eval_wrapper = EvaluatorMDMWrapper(args.dataset, dist_util.dev())
    evaluation(eval_wrapper, gt_loader, eval_motion_loaders, log_file, replication_times, diversity_times, mm_num_times, run_mm=run_mm)