import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import numpy as np
import pprint
import time  
import copy
import gc
import psutil
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from typing import Any, Dict, Optional, Tuple, Union, List
import torch
import torch.nn.functional as F
import math
import argparse

import json
from PIL import Image
import matplotlib.pyplot as plt
from diffusers import FluxPipeline,RfSolverFluxPipeline, RfSolverFluxTransformer2DModel
from diffusers.models.attention_processor import Attention
from torch import Tensor
from torchvision import transforms
from torch.utils.data import DataLoader
import logging

from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig
from transformers import BitsAndBytesConfig as TransformersBitsAndBytesConfig
from datasets import get_dataloader

from utils.utils import *
from utils.metrics import *


logger = logging.getLogger(__name__)   # pylint: disable=invalid-name


class RfSolverFluxAttnProcessor2_0_3opt:
    """Attention processor used typically in processing the SD3-like self-attention projections."""

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("FluxAttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    def __call__(
            self,
            attn:Attention,
            hidden_states: torch.FloatTensor,
            encoder_hidden_states: torch.FloatTensor = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            image_rotary_emb: Optional[torch.Tensor] = None,
            id: Optional[int] = None,
            inject: Optional[bool] = False,
            feature: Optional[Dict[str, Any]] = None,
            t: Optional[int] = None,
            second_order: Optional[Any] = None,
            inverse: Optional[bool] = False,
            type: Optional[str] = None,
            truncated_tokens: Optional[list] = None,
            feature_path: Optional[str] = None,
            inject_step: Optional[int] = None,
            attn_map_out: Optional[bool] = False,
            attn_map_out_path: Optional[str] = None,
            stable_flow: Optional[bool] = False,
            feature_operate: Optional[bool] = False,
            fig_info_list: Optional[list] = None,
            test: Optional[bool] = False,
            enhanced: Optional[bool] = False,
            enhanced_list: Optional[list] = None,
            feature_num: Optional[int] = 1,
            

        ) -> torch.FloatTensor:
            batch_size, _, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape

            # `sample` projections.
            query = attn.to_q(hidden_states)
            key = attn.to_k(hidden_states)
            value = attn.to_v(hidden_states)

            inner_dim = key.shape[-1]
            head_dim = inner_dim // attn.heads

            query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

            if attn.norm_q is not None:
                query = attn.norm_q(query)
            if attn.norm_k is not None:
                key = attn.norm_k(key)
            # Save the features in the memory 此处进行Value的特征保存和注入
            if test == False:
                if feature_operate:
                    #satble-flow代码
                    if stable_flow: 

                        if encoder_hidden_states is not None:
                            if inject and id in [
                                # 0,
                                1,2,
                                17, 18] and type=='double':
                                feature_name = str(t) + '_' + str(second_order) + '_' + str(id) + '_' + type + '_' + 'Q'
                                if inverse:
                                    feature[feature_name] = query.cpu()
                                else:
                                    query =feature[feature_name].cuda()

                                feature_name2 = str(t) + '_' + str(second_order) + '_' + str(id) + '_' + type + '_' + 'V'
                                if inverse:
                                    feature[feature_name2] = value.cpu()
                                else:
                                    value =feature[feature_name2].cuda()
                        else:
                            if inject and id in [
                                6, 9, 
                                30,32,33,34, 35,36, 37] and type=='single':
                                feature_name = str(t) + '_' + str(second_order) + '_' + str(id) + '_' + type + '_' + 'Q'
                                if inverse:
                                    feature[feature_name] = query[:, :, 512:, :].cpu()
                                else:
                                    query[:, :, 512:, :] = feature[feature_name].cuda()

                                feature_name2 = str(t) + '_' + str(second_order) + '_' + str(id) + '_' + type + '_' + 'V'
                                if inverse:
                                    feature[feature_name2] = value[:, :, 512:, :].cpu()
                                else:
                                    value[:, :, 512:, :] = feature[feature_name2].cuda()

                    #原RF-Solver代码
                    else:
                        if inject and id > 19 and type=='single':
                            feature_name = str(t) + '_' + str(second_order) + '_' + str(id) + '_' + type + '_' + 'V'
                            if inverse:
                                feature[feature_name] = value.cpu()
                            else:
                                value =feature[feature_name].cuda()
        

            # the attention in FluxSingleTransformerBlock does not use `encoder_hidden_states`
            if encoder_hidden_states is not None:
                # `context` projections.
                encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
                encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
                encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

                encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                    batch_size, -1, attn.heads, head_dim
                ).transpose(1, 2)
                encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                    batch_size, -1, attn.heads, head_dim
                ).transpose(1, 2)
                encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                    batch_size, -1, attn.heads, head_dim
                ).transpose(1, 2)

                if attn.norm_added_q is not None:
                    encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
                if attn.norm_added_k is not None:
                    encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)

                # attention
                query = torch.cat([encoder_hidden_states_query_proj, query], dim=2)
                key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
                value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)

            if image_rotary_emb is not None:
                from diffusers.models.embeddings import apply_rotary_emb

                query = apply_rotary_emb(query, image_rotary_emb)
                key = apply_rotary_emb(key, image_rotary_emb)

######################################实验代码
            is_inject = False
            if test:
                if feature_operate:
                    if stable_flow: 


                        if inject and id in [
                            # 0,
                            1,2,
                            17, 18] and type=='double':
                            is_inject = True
                            feature_name = str(t) + '_' + str(second_order) + '_' + str(id) + '_' + type + '_' + 'Q'
                            if inverse:
                                x=query
                                x  = x [:, :, 512:, :]
                                if feature.get(feature_name) is None:
                                    feature[feature_name] = (x/feature_num).cpu()
                                else:
                                    feature[feature_name] += (x/feature_num).cpu()
                            else:
                                query[:, :, 512:, :] =feature[feature_name].cuda()

                            feature_name2 = str(t) + '_' + str(second_order) + '_' + str(id) + '_' + type + '_' + 'K'
                            if inverse:
                                x=key
                                x  = x [:, :, 512:, :]
                                if feature.get(feature_name2) is None:
                                    feature[feature_name2] = (x/feature_num).cpu()
                                else:
                                    feature[feature_name2] += (x/feature_num).cpu()
                            else:
                                key[:, :, 512:, :] =feature[feature_name2].cuda()

                        if inject and (id in [
                            6, 9] or id >19 )and type=='single':
                            is_inject = True
                            feature_name = str(t) + '_' + str(second_order) + '_' + str(id) + '_' + type + '_' + 'Q'
                            if inverse:
                                x=query
                                x  = x [:, :, 512:, :]
                                if feature.get(feature_name) is None:
                                    feature[feature_name] = (x/feature_num).cpu()
                                else:
                                    feature[feature_name] += (x/feature_num).cpu()
                            else:
                                query[:, :, 512:, :] = feature[feature_name].cuda()

                            feature_name2 = str(t) + '_' + str(second_order) + '_' + str(id) + '_' + type + '_' + 'K'
                            if inverse:
                                x=key
                                x  = x [:, :, 512:, :]
                                if feature.get(feature_name2) is None:
                                    feature[feature_name2] = (x/feature_num).cpu()
                                else:
                                    feature[feature_name2] += (x/feature_num).cpu()
                            else:
                                key[:, :, 512:, :] = feature[feature_name2].cuda()

###########################3.20测试代码#############
                        if t==1.0:
                            
                            if inject and ( id in [
                                6, 9, 
                                30,32,33,34, 35,36, 37] )and type=='single':
                                # is_inject = True
                                feature_name = str(t) + '_' + str(second_order) + '_' + str(id) + '_' + type + '_' + 'V'
                                if inverse:
                                    x=value
                                    x  = x [:, :, 512:, :]
                                    if feature.get(feature_name) is None:
                                        feature[feature_name] = (x/feature_num).cpu()
                                    else:
                                        feature[feature_name] += (x/feature_num).cpu()
                                else:
                                    value[:, :, 512:, :] = feature[feature_name].cuda()
##############################################################
            
            if attn_map_out:




                def visualize_attention():
                    # 条件判断保留原有逻辑
                    if not ((type == 'single' and id in [
                        # 6,9
                        34,35,37] and second_order == 'k1') 
                            or (type == 'double' and id in [
                                # 0,1,2,
                                17,18] and second_order == 'k1')):
                        return

                    # 常量定义
                    NUM_TOKENS = 30
                    MAX_PER_PAGE = 50
                    FIG_COLS = 5

                    # GPU加速的注意力计算
                    with torch.no_grad():
                        # 保持所有计算在GPU上
                        attn_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(head_dim)
                        attn_map = torch.softmax(attn_scores, dim=-1)
                        attn_map = attn_map.float().mean(dim=0).mean(dim=0)  # 保持GPU tensor

                    # 向量化处理注意力矩阵
                    text_len, vision_len = 512, 4096
                    text_vision_attn = attn_map[:text_len, text_len:text_len+vision_len]

                    # 向量化reshape和切片操作
                    batch_size = min(NUM_TOKENS, text_len)
                    token_attn_3d = text_vision_attn[:batch_size].view(batch_size, 64, 64)
                    token_attn_3d = token_attn_3d[:, 1:-1, 1:-1]  # 向量化切片

                    # 在GPU上计算最大值
                    max_val = token_attn_3d.max().item()

                    # 一次性转移到CPU
                    token_attn_2d_list = token_attn_3d.cpu().unbind(0)

                    os.makedirs(attn_map_out_path, exist_ok=True)

                    # 分页处理
                    for page_idx in range(0, batch_size, MAX_PER_PAGE):
                        page_data = token_attn_2d_list[page_idx:page_idx+MAX_PER_PAGE]
                        current_count = len(page_data)
                        
                        # 动态计算布局
                        rows = math.ceil(current_count / FIG_COLS)
                        cols = FIG_COLS

                        fig, axs = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
                        axs = axs.reshape(-1, cols)  # 统一维度处理

                        # 预计算所有标题
                        titles = [truncated_tokens[i] if i < len(truncated_tokens) else "<pad>" 
                                for i in range(page_idx, page_idx+current_count)]

                        # 向量化绘图
                        for idx, (ax, attn_2d, title) in enumerate(zip(axs.ravel(), page_data, titles)):
                            # 动态计算宽高比例
                            height, width = attn_2d.shape
                            dynamic_extent = [0.5, width + 0.5, 0.5, height + 0.5]
                            
                            im = ax.imshow(
                                attn_2d,
                                aspect='equal',
                                interpolation="nearest",
                                cmap="viridis",
                                vmin=0,
                                vmax=max_val,
                                extent=dynamic_extent
                            )
                            ax.set_title(title, fontsize=8)
                            ax.axis("off")

                        # 关闭多余子图
                        for ax in axs.ravel()[current_count:]:
                            ax.axis('off')

                        # 优化布局
                        plt.tight_layout(rect=[0, 0, 0.85, 1])
                        
                        # 智能colorbar定位
                        active_axes = [ax for ax in axs.ravel() if ax.has_data()]
                        if active_axes:
                            positions = [ax.get_position() for ax in active_axes]
                            top = max(pos.ymax for pos in positions)
                            bottom = min(pos.ymin for pos in positions)
                            cbar_height = top - bottom
                            cbar_bottom = bottom
                        else:
                            cbar_bottom, cbar_height = 0.15, 0.7

                        cbar_ax = fig.add_axes([0.87, cbar_bottom, 0.02, cbar_height])
                        fig.colorbar(im, cax=cbar_ax)

                        # 保存优化
                        filename = f"{t}_{second_order}_{id}_{type}_inv{inverse}_feop{feature_operate}_page{page_idx//MAX_PER_PAGE + 1}.png"
                        save_path = os.path.join(attn_map_out_path, filename)
                        # 将图形对象和保存路径存入列表（而不是直接保存）
                        fig_info_list.append({
                            'fig': fig,
                            'save_path': save_path
                        })
                        # print(f"Saved attention map page to: {save_path}")

                # 调用可视化函数
                visualize_attention()


################################################3.20测试代码####################################
            def SVD_attention_enhanced(query, key, value, alpha=2.0, v=1.5, K=3):
                """
                基于SVD的注意力增强机制（并行优化版）
                
                参数说明:
                    query (Tensor): 查询向量，形状为 [B, H, L, D]
                    key (Tensor): 键向量，形状同query
                    value (Tensor): 值向量，形状同query
                    alpha (float): 注意力权重放大系数，默认2.0
                    v (float): Sigmoid斜率系数，默认1.5
                    K (int): 需要增强的基数量，默认3
                    
                返回:
                    Tensor: 增强后的注意力输出，形状与query相同
                
                注:
                    B: batch_size, H: num_heads, L: seq_len, D: head_dim
                    text_token_1_end=512, text_token_2_end=1024 为预定义常量
                """
                with torch.no_grad():
                    # ==================== 基础注意力计算 ====================
                    dk = query.size(-1)
                    scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(dk)
                    attn = F.softmax(scores, dim=-1)  # [B, H, L, L]
                    attn = attn.transpose(2, 3)
                    B, H, L, _ = attn.shape

                    # ==================== 常量定义 ====================
                    image_token_start = 1024     # 图像token起始位置
                    text_token_1_end = 512       # 第一段文本token结束位置
                    text_token_2_end = 1024      # 第二段文本token结束位置
                    
                    # ==================== 注意力矩阵切片 ====================
                    # 提取image->text2的注意力矩阵 [B, H, 1024:, 512:1024]
                    image_to_text2_attn = attn[:, :, image_token_start:, text_token_1_end:text_token_2_end]
                    # 提取image->text1的注意力矩阵 [B, H, 1024:, :512]
                    image_to_text1_attn = attn[:, :, image_token_start:, :text_token_1_end]

                    # ==================== 并行SVD处理 ====================
                    for b in range(B):  # 保持batch维度循环（通常batch_size较小）
                        # 当前batch的所有head并行处理
                        # image_to_text2_attn: [H, img_seq, text2_seq] (img_seq = L - 1024)
                        A_text2 = image_to_text2_attn[b]  # [H, img_seq, text2_seq]
                        
                        # 转换数据类型为float32
                        A_text2_float = A_text2.to(torch.float32)

                        # 批量SVD分解（并行处理所有head）
                        # U: [H, img_seq, k], S: [H, k], Vh: [H, k, text2_seq]
                        _, _, Vh = torch.linalg.svd(A_text2_float, full_matrices=False)
                        
                        # 转换回原始数据类型
                        Vh = Vh.to(A_text2.dtype)

                        # ==================== 投影与增强 ====================
                        # 将image->text1的注意力投影到基空间 [H, img_seq, text1_seq] × [H, text2_seq, k] -> [H, img_seq, k]
                        proj_coeff = torch.matmul(image_to_text1_attn[b], Vh.transpose(-1, -2))
                        
                        # 动态计算实际增强维度
                        k_dim = proj_coeff.size(-1)
                        effective_K = min(K, k_dim)
                        
                        # 非线性增强（仅处理前K个基）
                        if effective_K > 0:
                            # 分离需要增强的维度 [H, img_seq, K]
                            enhanced_dims = proj_coeff[..., :effective_K]
                            # 应用带参数的sigmoid门控
                            enhanced_dims = alpha * torch.sigmoid(v * enhanced_dims)
                            # 写回增强后的系数
                            proj_coeff = torch.cat([enhanced_dims, proj_coeff[..., effective_K:]], dim=-1)
                        
                        # ==================== 注意力重构 ====================
                        # 使用增强后的系数重构注意力矩阵 [H, img_seq, k] × [H, k, text2_seq] -> [H, img_seq, text2_seq]
                        reconstructed_attn = torch.matmul(proj_coeff, Vh)
                        
                        # ==================== 写回原矩阵 ====================
                        # 将重构后的注意力权重写回image->text1的位置
                        attn[b, :, image_token_start:, :text_token_1_end] = reconstructed_attn
                        attn = attn.transpose(2, 3)

                    # ==================== 最终输出 ====================
                    return torch.matmul(attn, value)


################################################################################################


            def enhanced_scaled_dot_product_attention(query, key, value, alpha=2.0, v=1.5, i_list=None, K=1):
                """
                增强型缩放点积注意力机制，支持基于语义目标的多头注意力增强
                
                参数说明:
                    query (Tensor): 查询向量，形状为 [batch_size, num_heads, seq_len, head_dim]
                    key (Tensor): 键向量，形状同query
                    value (Tensor): 值向量，形状同query
                    alpha (float): 注意力权重放大系数，默认2.0，控制增强幅度
                    v (float): Sigmoid函数的斜率系数，默认1.5，控制增强曲线的陡峭程度
                    i_list (list[int]): 需要增强的文本token索引列表，例如[256, 301]表示需要增强这两个位置的语义
                    K (int): 需要增强的注意力头数量，默认1
                
                返回:
                    Tensor: 增强后的注意力输出，形状与基础attn相同
                """
                if i_list is None:
                    i_list = []

                with torch.no_grad():  # 禁用梯度计算以提升性能
                    # ==================== 基础注意力计算 ====================
                    dk = query.size(-1)  # 获取每个注意力头的维度
                    # 计算缩放点积注意力分数
                    scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(dk)
                    # 应用softmax得到归一化注意力权重 [batch, heads, seq, seq]
                    attn = F.softmax(scores, dim=-1)
                    batch_size, num_heads, seq_len, _ = attn.shape
                    attn = attn.transpose(2, 3)

                    # ==================== 语义增强处理 ====================
                    image_token_start = 1024  # 假设图像token从第512个位置开始
                    text_token_end = 1024     # 文本token结束位置（CLIP等模型常见设置）
                    
                    # 过滤有效索引（防止超出文本区域）
                    valid_i_list = [i for i in i_list if i < text_token_end]
                    
                    # 当存在有效目标token且存在图像token时进行处理
                    if valid_i_list and image_token_start < seq_len:
                        # 转换为张量以便GPU并行计算
                        valid_i_tensor = torch.tensor(valid_i_list, device=attn.device)
                        num_image_tokens = seq_len - image_token_start  # 计算图像token数量

                        # 逐批次处理（由于不同batch可能有不同语义，无法完全并行化）
                        for b in range(batch_size):
                            # ========== 阶段1：并行化头部选择 ==========
                            # 提取文本到图像的注意力子矩阵 [num_heads, 512, num_image_tokens]
                            text_scores = attn[b, :, :text_token_end, image_token_start:]
                            
                            # 计算每个文本token对图像的整体影响力（沿图像维度求和）
                            # 结果形状 [num_heads, 512]
                            text_total = text_scores.sum(dim=-1)
                            
                            # 对每个注意力头中的文本token按影响力排序（降序）
                            # sorted_indices形状 [num_heads, 512]
                            sorted_indices = torch.argsort(text_total, dim=1, descending=True)
                            
                            # --------- 并行计算目标token的排名 ---------
                            # 扩展维度以便广播计算
                            # valid_i_tensor从[L]变为[L, 1, 1]（L为目标token数量）
                            target_expanded = valid_i_tensor.view(-1, 1, 1)
                            # sorted_indices从[H, 512]变为[1, H, 512]（H为注意力头数量）
                            sorted_expanded = sorted_indices.unsqueeze(0)
                            
                            # 生成位置掩码：找出每个目标token在排序中的位置
                            # pos_mask形状 [L, H, 512]
                            pos_mask = (sorted_expanded == target_expanded)
                            
                            # 检查每个头是否包含目标token（存在性判断）
                            # exists形状 [L, H]
                            exists = pos_mask.any(dim=-1)
                            
                            # 获取目标token在各头中的排名位置（第一个匹配的位置）
                            # 使用argmax在最后一个维度（512）找第一个True的位置
                            # rank_pos形状 [L, H]
                            rank_pos = pos_mask.int().argmax(dim=-1)
                            
                            # 对不包含目标token的头部赋予最大排名值（惩罚机制）
                            # sorted_indices.size(1)=512，即最大可能的排名值+1
                            rank_pos = torch.where(exists, rank_pos, sorted_indices.size(1))
                            
                            # 计算综合排名：各目标token排名的总和
                            # total_ranks形状 [H]
                            total_ranks = rank_pos.sum(dim=0)
                            
                            # --------- 选择Top-K头部 ---------
                            valid_K = min(K, num_heads)  # 处理K超过头数的情况
                            if valid_K > 0:
                                # 选择综合排名最小的K个头部（总和最小表示对目标token集合关注度最高）
                                _, topk_indices = torch.topk(total_ranks, valid_K, largest=False)
                                best_heads = topk_indices.tolist()  # 转换为列表形式

                                # ========== 阶段2：矩阵化增强计算 ==========
                                if best_heads:
                                    # 提取选中头部的注意力子矩阵 [K, 512, num_image_tokens]
                                    selected_attn = attn[b, best_heads, :text_token_end, image_token_start:]
                                    
                                    # 归一化：每个文本token对各图像token的注意力权重之和为1
                                    sum_image = selected_attn.sum(dim=-1, keepdim=True) + 1e-8  # 防止除零
                                    normalized = selected_attn / sum_image  # [K, 512, num_image]
                                    
                                    # 应用增强公式：alpha * sigmoid(v * normalized)
                                    # 通过sigmoid实现非线性增强，v控制斜率，alpha控制幅度
                                    enhanced_attn = alpha * torch.sigmoid(v * normalized)
                                    
                                    # 将增强后的权重写回原矩阵
                                    attn[b, best_heads, :text_token_end, image_token_start:] = enhanced_attn

                    # ==================== 最终输出计算 ====================
                    # 使用增强后的注意力权重与value矩阵相乘
                    # attn = attn.transpose(2, 3)
                    out = torch.matmul(attn, value)


                return out




###########################3.20 测试代码#############
            if test and inverse == False and enhanced and inject and is_inject and t<=0.96 and t>=0.85:

                hidden_states = enhanced_scaled_dot_product_attention(query, key, value, alpha=3, v=3, i_list=enhanced_list,K=5)

                # hidden_states = SVD_attention_enhanced(query, key, value, alpha=5, v=3,K=3)

###################################################

            else:
                hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)

            hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
            hidden_states = hidden_states.to(query.dtype)           

            if encoder_hidden_states is not None:
                encoder_hidden_states, hidden_states = (
                    hidden_states[:, : encoder_hidden_states.shape[1]],
                    hidden_states[:, encoder_hidden_states.shape[1] :],
                )

                # linear proj
                hidden_states = attn.to_out[0](hidden_states)
                # RfSolver不dropout
                #hidden_states = attn.to_out[1](hidden_states)
                encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

                return hidden_states, encoder_hidden_states
            else:
                return hidden_states


@torch.inference_mode()
def interpolated_inversion(
    pipeline, 
    latents,
    DTYPE,
    joint_attention_kwargs,
    num_steps=28,
    use_shift_t_sampling=True, 
    t5_prompt="",
    clip_prompt="",
    guidance_scale = 1.0
):


    # 源文本提示
    prompt_embeds, pooled_prompt_embeds, text_ids = pipeline.encode_prompt(
        prompt=t5_prompt, 
        prompt_2=clip_prompt
    )

    # 源文本提示T5 tokenizer_2输出
    source_text_inputs = pipeline.tokenizer_2(
        t5_prompt,
        padding="max_length",
        max_length=512,
        truncation=True,
        return_tensors="pt",
    )
    source_text_ids = source_text_inputs.input_ids

    source_tokens = pipeline.tokenizer_2.convert_ids_to_tokens(source_text_ids[0])

    end_index = source_tokens.index('</s>')
    truncated_tokens = source_tokens[: end_index + 1]
    joint_attention_kwargs['truncated_tokens']=truncated_tokens

    print("source_tokens", truncated_tokens)
    print("source_text_ids", source_text_ids.shape)

    #print("latents", latents.shape)
    # 准备潜变量图像ID
    latent_image_ids = pipeline._prepare_latent_image_ids(
        latents.shape[0],
        latents.shape[2],
        latents.shape[3],
        latents.device, 
        DTYPE,
    )
    #print("latent_image_ids", latent_image_ids.shape)

    # 打包潜变量
    packed_latents = pipeline._pack_latents(
        latents,
        batch_size=latents.shape[0],
        num_channels_latents=latents.shape[1],
        height=latents.shape[2],
        width=latents.shape[3],
    )
    # 获取时间步长调度表
    
    timesteps = get_schedule( 
                num_steps=num_steps,
                image_seq_len=(packed_latents.shape[1] ), # vae_scale_factor = 16
                shift=use_shift_t_sampling,
            )
    
    
    
    # 准备指导向量
    guidance_vec = torch.full((packed_latents.shape[0],), guidance_scale, device=packed_latents.device, dtype=packed_latents.dtype)

    inject_list = [True] * joint_attention_kwargs['inject_step'] + [False] * (len(timesteps[:-1]) - joint_attention_kwargs['inject_step'])
    #反演过程反转调度表和inject_list
    timesteps = timesteps[::-1]
    inject_list = inject_list[::-1]

    pipeline.text_encoder_2.to('cpu')
    pipeline.vae.to('cpu')
    torch.cuda.empty_cache()   
    # 使用三阶 Runge-Kutta 方法进行图像反演
    with pipeline.progress_bar(total=len(timesteps)-1) as progress_bar:
        for i, (t_curr, t_prev) in enumerate(zip(timesteps[:-1], timesteps[1:])):
            h = t_prev - t_curr  # 时间步长
            t_vec = torch.full((packed_latents.shape[0],), t_curr, dtype=packed_latents.dtype, device=packed_latents.device)

            joint_attention_kwargs['t'] = t_prev 
            joint_attention_kwargs['inverse'] = True
            joint_attention_kwargs['second_order'] = 'k1'
            joint_attention_kwargs['inject'] = inject_list[i]
            #print("joint_attention_kwargs id:",id(joint_attention_kwargs))
            # 第一步：计算 k1
            k1 ,joint_attention_kwargs= pipeline.transformer(
                    hidden_states=packed_latents,
                    timestep=t_vec,
                    guidance=guidance_vec,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    joint_attention_kwargs=joint_attention_kwargs,  #TODO:此处可以传递inject的相关参数，详细仍需再研究，考虑形如 joint_attention_kwargs['inject'] = inject_list[i]  24/11/19 修改到此
                    return_dict=pipeline,
                )
            k1=k1[0]
            #print("joint_attention_kwargs id:",id(joint_attention_kwargs))


            # 第二步：计算 k2
            joint_attention_kwargs['second_order'] = 'k2'
            packed_latents_k2 = packed_latents + 0.5 * h * k1
            t_k2 = t_curr + 0.5 * h
            t_vec_k2 = torch.full((packed_latents.shape[0],), t_k2, dtype=packed_latents.dtype, device=packed_latents.device)
            k2 ,joint_attention_kwargs= pipeline.transformer(
                    hidden_states=packed_latents_k2,
                    timestep=t_vec_k2,
                    guidance=guidance_vec,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    joint_attention_kwargs=joint_attention_kwargs,  
                    return_dict=pipeline,
                )
            k2=k2[0]

            # 第三步：计算 k3
            packed_latents_k3 = packed_latents - h * k1 + 2 * h * k2
            t_k3 = t_curr + h
            t_vec_k3 = torch.full((packed_latents.shape[0],), t_k3, dtype=packed_latents.dtype, device=packed_latents.device)
            joint_attention_kwargs['second_order'] = 'k3'
            #joint_attention_kwargs['third_order'] = True  # 标记为三阶计算
            k3, joint_attention_kwargs = pipeline.transformer(
                    hidden_states=packed_latents_k3,
                    timestep=t_vec_k3,
                    guidance=guidance_vec,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    joint_attention_kwargs=joint_attention_kwargs,
                    return_dict=pipeline,
                )
            k3 = k3[0]          

            # 防止精度问题
            packed_latents = packed_latents.to(torch.float32)
            k1 = k1.to(torch.float32)
            k2 = k2.to(torch.float32)
            k3 = k3.to(torch.float32)
            # 更新潜变量
            packed_latents = packed_latents + (h / 6) * (k1 + 4 * k2 + k3)
              
            packed_latents = packed_latents.to(DTYPE)
            progress_bar.update()
            
    
    
    # 解包潜变量
    latents = pipeline._unpack_latents(
            packed_latents,
            height=args.height,
            width=args.width,
            vae_scale_factor=pipeline.vae_scale_factor,
    )
    latents = latents.to(DTYPE)
    return latents ,joint_attention_kwargs



@torch.inference_mode()
def interpolated_denoise(
    pipeline, 
    joint_attention_kwargs,
    inversed_latents,            # 如果不使用反转潜变量，可以为 None
    use_inversed_latents=True,
    guidance_scale=4.0,
    target_prompt='photo of a tiger',
    DTYPE=torch.bfloat16,
    num_steps=28,
    use_shift_t_sampling=True, 
):

    pipeline.text_encoder_2.to('cuda')

    # 编码提示文本
    prompt_embeds, pooled_prompt_embeds, text_ids = pipeline.encode_prompt(
        prompt=target_prompt, 
        prompt_2=target_prompt
    )


        # 目标文本提示T5 tokenizer_2输出
    target_text_inputs = pipeline.tokenizer_2(
        target_prompt,
        padding="max_length",
        max_length=512,
        truncation=True,
        return_tensors="pt",
    )
    target_text_ids = target_text_inputs.input_ids

    target_tokens = pipeline.tokenizer_2.convert_ids_to_tokens(target_text_ids[0])

    end_index = target_tokens.index('</s>')
    truncated_tokens = target_tokens[: end_index + 1]
    joint_attention_kwargs['truncated_tokens']=truncated_tokens

    print("target_tokens", truncated_tokens)
    print("target_text_ids", target_text_ids.shape)
################3.20测试代码#############

    # new_list = [
    #     truncated_tokens[i] 
    #     for i in joint_attention_kwargs['enhanced_list']
    # ]

    # # 输出结果
    # print(new_list)  # 例如输出：["token1", "token2"]    
    # prompt_en = ''.join(token.replace('▁', ' ') for token in new_list).strip()
    # print("prompt_en", prompt_en)

    # prompt_embeds__enh, pooled_prompt_embeds_enh, text_ids_enh = pipeline.encode_prompt(
    #     prompt=target_prompt, 
    #     prompt_2=prompt_en
    # )
    # del pooled_prompt_embeds_enh
    # prompt_embeds=torch.cat([prompt_embeds, prompt_embeds__enh], dim=1)
    # text_ids=torch.cat([text_ids, text_ids_enh], dim=0)
#############################################################################

    # 准备潜变量图像ID
    latent_image_ids = pipeline._prepare_latent_image_ids(
        inversed_latents.shape[0],
        inversed_latents.shape[2],
        inversed_latents.shape[3],
        inversed_latents.device,
        DTYPE,
    )

    if use_inversed_latents:
        # 使用反转潜变量
        packed_latents = pipeline._pack_latents(
            inversed_latents,
            batch_size=inversed_latents.shape[0],
            num_channels_latents=inversed_latents.shape[1],
            height=inversed_latents.shape[2],
            width=inversed_latents.shape[3],
        )
    else:
        # 生成随机潜变量
        tmp_latents = torch.randn_like(img_latents)
        packed_latents = pipeline._pack_latents(
            tmp_latents,
            batch_size=tmp_latents.shape[0],
            num_channels_latents=tmp_latents.shape[1],
            height=tmp_latents.shape[2],
            width=tmp_latents.shape[3],
        )


    # 获取时间步长调度表
    timesteps = get_schedule( 
                num_steps=num_steps,
                image_seq_len=(packed_latents.shape[1] ), # vae_scale_factor = 16
                shift=use_shift_t_sampling,
            )
    

    guidance_vec = torch.full((packed_latents.shape[0],), guidance_scale, device=packed_latents.device, dtype=packed_latents.dtype)
    inject_list = [True] * joint_attention_kwargs['inject_step'] + [False] * (len(timesteps[:-1]) - joint_attention_kwargs['inject_step'])

    pipeline.text_encoder_2.to('cpu')
    torch.cuda.empty_cache()  
    # 使用三阶 Runge-Kutta进行去噪
    with pipeline.progress_bar(total=len(timesteps)-1) as progress_bar:
        for i, (t_curr, t_prev) in enumerate(zip(timesteps[:-1], timesteps[1:])):
            h = t_prev - t_curr  # 时间步长
            t_vec = torch.full((packed_latents.shape[0],), t_curr, dtype=packed_latents.dtype, device=packed_latents.device)

            joint_attention_kwargs['t'] = t_curr 
            joint_attention_kwargs['inverse'] = False
            joint_attention_kwargs['second_order'] = 'k1'
            joint_attention_kwargs['inject'] = inject_list[i]
            #print("joint_attention_kwargs id:",id(joint_attention_kwargs))
            # 第一步：计算 k1
            k1 ,joint_attention_kwargs= pipeline.transformer(
                    hidden_states=packed_latents,
                    timestep=t_vec,
                    guidance=guidance_vec,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    joint_attention_kwargs=joint_attention_kwargs,  
                    return_dict=pipeline,
                )
            k1=k1[0]
            #print("joint_attention_kwargs id:",id(joint_attention_kwargs))


            # 第二步：计算 k2
            joint_attention_kwargs['second_order'] = 'k2'
            packed_latents_k2 = packed_latents + 0.5 * h * k1
            t_k2 = t_curr + 0.5 * h
            t_vec_k2 = torch.full((packed_latents.shape[0],), t_k2, dtype=packed_latents.dtype, device=packed_latents.device)
            k2 ,joint_attention_kwargs= pipeline.transformer(
                    hidden_states=packed_latents_k2,
                    timestep=t_vec_k2,
                    guidance=guidance_vec,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    joint_attention_kwargs=joint_attention_kwargs,  
                    return_dict=pipeline,
                )
            k2=k2[0]

            # 第三步：计算 k3
            packed_latents_k3 = packed_latents - h * k1 + 2 * h * k2
            t_k3 = t_curr + h
            t_vec_k3 = torch.full((packed_latents.shape[0],), t_k3, dtype=packed_latents.dtype, device=packed_latents.device)
            joint_attention_kwargs['second_order'] = 'k3'
            #joint_attention_kwargs['third_order'] = True  # 标记为三阶计算
            k3, joint_attention_kwargs = pipeline.transformer(
                    hidden_states=packed_latents_k3,
                    timestep=t_vec_k3,
                    guidance=guidance_vec,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    joint_attention_kwargs=joint_attention_kwargs,
                    return_dict=pipeline,
                )
            k3 = k3[0]          

            # 防止精度问题
            packed_latents = packed_latents.to(torch.float32)
            k1 = k1.to(torch.float32)
            k2 = k2.to(torch.float32)
            k3 = k3.to(torch.float32)
            # 更新潜变量
            packed_latents = packed_latents + (h / 6) * (k1 + 4 * k2 + k3) 
            
            packed_latents = packed_latents.to(DTYPE)
            progress_bar.update()
    
    # 解包潜变量
    latents = pipeline._unpack_latents(
            packed_latents,
            height=args.height,
            width=args.width,
            vae_scale_factor=pipeline.vae_scale_factor,
    )
    latents = latents.to(DTYPE)
    return latents ,joint_attention_kwargs

def calculate_ex_t_squared(x_t: torch.Tensor) -> torch.Tensor:
    """
    计算扩散模型中潜在变量x_t的平均相对能量（公式8）
    Args:
        x_t: 输入张量，形状为 [B, C, H, W] 或 [C, H, W]
    Returns:
        Ex_t_squared: 平均能量，形状为 [B]（有批量）或标量（无批量）
    """
    # 计算所有元素的平方和（自动处理批量维度）
    sum_squares = torch.sum(x_t**2, dim=tuple(range(-3, 0)))  # 对C,H,W求和
    # 计算归一化因子 C*H*W
    chw = x_t.shape[-3] * x_t.shape[-2] * x_t.shape[-1]
    # 返回平均能量
    return sum_squares / chw

def calculate_max_workers() -> int:
    """动态计算最优线程数"""
    try:
        # 获取系统资源状态
        cpu_cores = psutil.cpu_count(logical=False) or 1
        cpu_usage = psutil.cpu_percent(interval=0.1)
        disk_usage = psutil.disk_usage('/').percent
        
        # 计算可用线程数（经验公式）
        available_workers = max(1, int(
            (cpu_cores * 0.8) *                     # 80%物理核心数
            (1 - cpu_usage/100) *                  # CPU空闲率因子
            (1 - min(disk_usage, 90)/100 * 0.5)    # 磁盘压力因子
        ))
        
        return min(available_workers, 32)  # 最大不超过32线程
    except:
        return 4  # 异常时返回安全值

def _save_single_figure(fig: plt.Figure, save_path: str) -> None:
    """原子化保存操作（带重试机制）"""
    try:
        # 第一次尝试保存
        fig.savefig(save_path, bbox_inches="tight")
    except Exception as e:
        # 失败时重试一次
        try:
            fig.savefig(save_path, bbox_inches="tight")
        except:
            raise RuntimeError(f"保存失败: {save_path} ({str(e)})")
    finally:
        # 确保资源释放
        plt.close(fig)
        del fig
        
def save_all_figures(fig_info_list: List[Dict[str, Any]]) -> None:
    """智能并行保存图形（带进度显示）"""
    if not fig_info_list:
        return

    # 预创建所有目录（避免并行竞争）
    dir_paths = {os.path.dirname(info["save_path"]) for info in fig_info_list}
    for d in dir_paths:
        os.makedirs(d, exist_ok=True)

    # 动态计算线程数
    max_workers = calculate_max_workers()
    print(f"使用并行线程数: {max_workers}")

    # 初始化进度条
    progress = tqdm(
        total=len(fig_info_list),
        desc="保存图形",
        unit="fig",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
    )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for info in fig_info_list:
            future = executor.submit(
                _save_single_figure,
                fig=info["fig"],
                save_path=info["save_path"]
            )
            future.add_done_callback(lambda _: progress.update(1))
            futures.append(future)

        # 异常收集和处理
        errors = []
        for future in futures:
            try:
                future.result()
            except Exception as e:
                errors.append(str(e))

    progress.close()
    
    # 错误报告
    if errors:
        print(f"\n 完成保存但有 {len(errors)} 个错误:")
        for err in set(errors[:3]):  # 显示前3个不同错误
            print(f"  - {err}")
        if len(errors) > 3:
            print(f"  更多错误已隐藏...")

    # 清空列表
    fig_info_list.clear()



class SentenceExtractor:
    def __init__(self, json_path: str):
        self.data = self._load_and_clean_data(json_path)  # 修改方法名称
        self.total_entries = len(self.data)
        
    def _load_and_clean_data(self, file_path: str) -> List[dict]:
        """加载数据并进行清洗"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            raise ValueError("Invalid JSON format: expected list at top level")
            
        cleaned_data = []
        for idx, entry in enumerate(data):
            if not entry:  # 过滤空条目
                continue
                
            # 验证必须字段
            if 'original_data' not in entry or 'variations' not in entry:
                raise ValueError(f"Missing required fields in entry {idx}")
            
            # 清洗variations数据
            self._clean_variations(entry['variations'])
            cleaned_data.append(entry)
            
        return cleaned_data
    
    def _clean_variations(self, variations: Dict) -> None:
        """清洗所有variation的句子"""
        for var in variations.values():
            if 'new_target_sentence' in var:
                # 删除</s>并清理空格
                cleaned = var['new_target_sentence'].replace('</s>', '').strip()
                var['new_target_sentence'] = cleaned
    
    def get_sentences(self, index: int) -> List[str]:
        """获取指定序号对应的10个新句子"""
        if index < 0 or index >= self.total_entries:
            raise IndexError(f"Index out of range. Valid range: 0-{self.total_entries-1}")
            
        entry = self.data[index]
        variations = entry['variations']
        
        # 按variation_id排序并提取句子
        sorted_variations = sorted(
            variations.items(),
            key=lambda x: int(x[0].split('_')[-1])
        )
        
        return [var['new_target_sentence'] for _, var in sorted_variations]


@torch.inference_mode()
def main(args):
    start_time = time.time()  # 记录开始时间

    if args.dtype == 'bfloat16':
        DTYPE = torch.bfloat16
    elif args.dtype == 'float16':
        DTYPE = torch.float16
    elif args.dtype == 'float32':
        DTYPE = torch.float32
    else:
        raise ValueError(f"不支持的数据类型: {args.dtype}")



    os.makedirs(args.output_dir, exist_ok=True)

    joint_attention_kwargs = {}
    joint_attention_kwargs['feature_path'] = args.feature_path
    joint_attention_kwargs['feature'] = {}
    joint_attention_kwargs['inject_step'] = args.inject
    joint_attention_kwargs['attn_map_out'] = args.attn_map_out
    joint_attention_kwargs['attn_map_out_path'] = args.attn_map_out_path
    joint_attention_kwargs['stable_flow'] = args.stable_flow
    joint_attention_kwargs['feature_operate']=False
    joint_attention_kwargs['fig_info_list'] = []
    joint_attention_kwargs['test'] = args.test 
    joint_attention_kwargs['enhanced'] = args.enhanced
    joint_attention_kwargs['feature_num'] = args.target_var_num+1





    if not os.path.exists(args.feature_path):
        os.mkdir(args.feature_path)
    if args.attn_map_out and not os.path.exists(args.attn_map_out_path):
        os.mkdir(args.attn_map_out_path)


    device = "cuda" if torch.cuda.is_available() else "cpu"
    #device = "cpu"

    metrics = metircs()

    # ******** Loading pipeline **********
    quant_config = DiffusersBitsAndBytesConfig(load_in_8bit=True,)
    transformer_8bit = RfSolverFluxTransformer2DModel.from_pretrained(
        args.model_path,
        subfolder="transformer",
        quantization_config=quant_config,
        torch_dtype=torch.bfloat16,
    )
    pipe = RfSolverFluxPipeline.from_pretrained(args.model_path, torch_dtype=DTYPE,transformer=transformer_8bit)
    print(pipe.hf_device_map)
    pipe.enable_model_cpu_offload()
    #pipe.enable_sequential_cpu_offload()

    pipe.transformer.set_attn_processor(RfSolverFluxAttnProcessor2_0_3opt())


    # my_attn = RfSolverAttention(
    #             query_dim=dim,
    #             cross_attention_dim=None,
    #             dim_head=attention_head_dim,
    #             heads=num_attention_heads,
    #             out_dim=dim,
    #             bias=True,
    #             processor=processor,
    #             qk_norm="rms_norm",
    #             eps=1e-6,
    #             pre_only=True,
    #         )

    # ******** Input processing **********
    # 参数互斥检查
    if bool(args.eval_datasets) == (args.image_path is not None):
        raise ValueError("必须且只能选择一种模式：--eval-datasets 或单张图片相关参数（--image-path等）")
    
    # 创建统一的数据加载结构
    if args.eval_datasets:
        # 数据集模式
        default_transform =  transforms.Compose(
                    [
                    transforms.Resize((args.height, args.width), interpolation=transforms.InterpolationMode.BILINEAR),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5])
                    ]
                )
        # 加载原始数据集
        full_dataset = get_dataloader(args.eval_datasets, default_transform)

        # 处理子集参数
        start_index = 0
        if args.num_samples is not None:
            if args.num_samples <= 0:
                raise ValueError("--num-samples 必须大于0")
            indices = list(range(args.num_samples))
            dataset = torch.utils.data.Subset(full_dataset, indices)
            start_index = 0  # 子集从原始数据集0开始
            
        elif args.subset_range is not None:
            start, end = args.subset_range
            if start < 0 or end <= start:
                raise ValueError("--subset-range 参数不合法，必须满足 START >=0 且 END > START")
            if end > len(full_dataset):
                raise ValueError(f"END 值 {end} 超过数据集总长度 {len(full_dataset)}")
            indices = list(range(start, end))
            dataset = torch.utils.data.Subset(full_dataset, indices)
            start_index = start  # 记录原始起始索引
            
        else:
            # 处理整个数据集
            dataset = full_dataset
            start_index = 0

        dataloader = DataLoader(
            dataset,
            batch_size=1,          # 每批64个样本
            shuffle=False,           # 训练时打乱数据
            num_workers=8,          # 使用4个子进程加载数据
            pin_memory=True         # 如果使用GPU，可以加速数据传输
        )

        dataloader_full = DataLoader(
            full_dataset,
            batch_size=1,          # 每批64个样本
            shuffle=False,           # 训练时打乱数据
            num_workers=8,          # 使用4个子进程加载数据
            pin_memory=True         # 如果使用GPU，可以加速数据传输
        )
      
        
        # 加载ilist配置并验证长度
        with open(args.ilist_json_path, 'r') as f:
            ilist_entries = json.load(f)
            
        # 验证子集范围是否超出ilist条目数
        max_required_index = start_index + len(dataset)
        if max_required_index > len(ilist_entries):
            raise ValueError(f"ilist条目数不足，需要至少 {max_required_index} 条，当前只有 {len(ilist_entries)} 条")

    
    else:
        # 单张图片模式
        if not all([args.source_prompt, args.target_prompt]):
            raise ValueError("单张图片模式需要提供 --source-prompt 和 --target-prompt")

        # 图像加载和预处理
        img = Image.open(args.image_path)
        train_transforms = transforms.Compose(
                    [
                    transforms.Resize((args.height, args.width), interpolation=transforms.InterpolationMode.BILINEAR),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5])
                    ]
                )
        img_tensor = train_transforms(img).unsqueeze(0)
        
        # 构建与数据集模式兼容的数据结构
        dataloader = [(img_tensor, args.source_prompt, args.target_prompt)]


    # ******** evaluation **********
    mean_clip_score = 0
    mean_clip_v_score = 0
    mean_mse_score = 0
    mean_psnr_score = 0
    mean_lpips_score = 0
    mean_ssim_score = 0
    mean_dino_score = 0
    count = 0

    extractor = SentenceExtractor("/home/chx/mySrc/diffusers-dev-Bob/codes/semantic_variations_EditEval_v1.json")
    
    target_prompts_list = []
    for img_float32, source_prompt, target_prompt in dataloader_full:
        target_prompts_list.extend(target_prompt)  # 直接扩展列表
    #     print("target_prompt", target_prompt)
    # print("target_prompts_list.len=", len(target_prompts_list))
    # print("target_prompts_list", target_prompts_list)

    for img_float32, source_prompt, target_prompt in dataloader:


        for single_target_prompt in target_prompt:
            target_prompts_list = [p for p in target_prompts_list if p != single_target_prompt]
        # print("当前剩余:", len(target_prompts_list))


        def sample_evenly_spaced(target_list, K):
            """
            从列表中均匀选取 K 个元素（包含首尾）
            
            参数:
                target_list (list): 原始列表（长度需 ≥1）
                K (int): 需要抽取的元素数量（1 ≤ K ≤ len(target_list)）
            
            返回:
                list: 均匀采样后的新列表
            """
            if K <= 0 or not target_list:
                return []
            
            N = len(target_list)
            indices = np.linspace(0, N-1, num=K, dtype=int).tolist()  # 生成等距索引
            return [target_list[i] for i in indices]
        
        target_var_list = sample_evenly_spaced(target_prompts_list, K=args.target_var_num)
        print("target_var_list", target_var_list)


        # 计算全局索引
        global_idx = start_index + count if args.eval_datasets else count
        


        # 处理增强列表
        if args.eval_datasets:
            current_ilist = ilist_entries[start_index + count]['ilist']  # 使用全局索引访问 ilist
        else:
            current_ilist = args.enhanced_list

        joint_attention_kwargs['enhanced_list'] = current_ilist

        print("source_prompt", source_prompt)
        print("target_prompt", target_prompt)



        img_float32 = img_float32.to(device)
        img = img_float32.to(DTYPE)
        # vae encode
        img_latent = encode_imgs(img, pipe, DTYPE)
        print("img_latent:", calculate_ex_t_squared(img_latent))
        
        #nudge 标量偏移
        if args.nudge != 1:
            img_latent_nudge = img_latent*args.nudge
            print("img_latent_nudge:", calculate_ex_t_squared(img_latent))

        if args.use_inversed_latents:
            # 进行反演1
        #     inversed_latent_s ,joint_attention_kwargs = interpolated_inversion(
        #         pipe, 
        #         img_latent*1.15, 
        #         DTYPE=DTYPE, 
        #         num_steps=args.num_steps, 
        #         use_shift_t_sampling=True,
        #         t5_prompt=source_prompt,
        #         clip_prompt=source_prompt,
        #         guidance_scale = 1.5,
        #         joint_attention_kwargs=joint_attention_kwargs)    
        # else:
            inversed_latent_s = None

#*******************************************************************************
            # 保存和加载部分
            if inversed_latent_s is None and joint_attention_kwargs is not None:
                # 计算全局索引
                global_idx = start_index + count if args.eval_datasets else count
                
                # 保存反演潜在变量
                latent_save_path = f"/data/chx/3op_data/inversed_latent_s/inversed_latent_{global_idx}.pt"
                # torch.save(inversed_latent_s.cpu(), latent_save_path)
                # print(f"已保存反演潜在变量{global_idx}到 {latent_save_path}")
                
                # # 保存注意力参数（兼容非张量数据）
                # joint_save_path = f"joint_attention_{global_idx}.pt"
                # kwargs_to_save = {
                #     k: v.cpu() if isinstance(v, torch.Tensor) else v
                #     for k, v in joint_attention_kwargs.items()
                # }
                # torch.save(kwargs_to_save, joint_save_path)
                
                # 重新加载
                inversed_latent_s = torch.load(latent_save_path, map_location=device).to(DTYPE)
                print(f"已加载反演潜在变量{global_idx}位于 {latent_save_path}")
                # loaded_kwargs = torch.load(joint_save_path, map_location=device)
                # joint_attention_kwargs = {
                #     k: v.to(device) if isinstance(v, torch.Tensor) else v
                #     for k, v in loaded_kwargs.items()
                # }
#*******************************************************************************
        print("inversed_latent_s:", calculate_ex_t_squared(inversed_latent_s))
        joint_attention_kwargs['feature_operate']=True,


        if args.use_inversed_latents:
            # 进行反演2
            inversed_latent_t ,joint_attention_kwargs = interpolated_inversion(
                pipe, 
                img_latent_nudge,
                DTYPE=DTYPE, 
                num_steps=args.num_steps, 
                use_shift_t_sampling=True,
                t5_prompt=target_prompt,
                clip_prompt=source_prompt,
                guidance_scale = 1,
                joint_attention_kwargs=joint_attention_kwargs)    
        else:
            inversed_latent_t = None
        
        # 进行语义丰富化
        # target_var_list=extractor.get_sentences(global_idx)
        # target_var_list=target_var_list[:args.target_var_num]
        
        # for i in joint_attention_kwargs['feature'].values(): 
        #     i=i/(args.target_var_num+1)

        # joint_attention_kwargs_var = {
        #     key: copy.deepcopy(value) if key != 'feature' else {}
        #     for key, value in joint_attention_kwargs.items()
        # }
        # 变量反演循环
        for target_prompt_var in target_var_list:
            
        
            _, joint_attention_kwargs = interpolated_inversion(
                pipe, img_latent,
                DTYPE=DTYPE, 
                num_steps=args.num_steps, 
                use_shift_t_sampling=True,
                t5_prompt=target_prompt_var,
                clip_prompt=source_prompt,
                guidance_scale=1,
                joint_attention_kwargs=joint_attention_kwargs
            )
            
            # 特征融合
            # for k in joint_attention_kwargs['feature']:
            #     joint_attention_kwargs['feature'][k] += joint_attention_kwargs_var['feature'][k] / (args.target_var_num + 1)
            
        #     # 及时清理
        # del joint_attention_kwargs_var, target_prompt_var


        print("inversed_latent_t:", calculate_ex_t_squared(inversed_latent_t))
        # 进行去噪
        img_latents,joint_attention_kwargs = interpolated_denoise(
            pipe, 
            inversed_latents=inversed_latent_s,
            use_inversed_latents=args.use_inversed_latents,
            joint_attention_kwargs=joint_attention_kwargs,
            guidance_scale=args.guidance_scale,
            target_prompt=target_prompt,
            DTYPE=DTYPE,
            num_steps=args.num_steps,
            use_shift_t_sampling=True,
        )

        print("img_latents_nudge:", calculate_ex_t_squared(img_latents))
        torch.cuda.empty_cache()  
        pipe.text_encoder_2.to('cpu')
        pipe.vae.to('cuda')
        torch.cuda.empty_cache() 

        # 将潜变量解码为图像
        out = decode_imgs(img_latents, pipe,output_type="pil")[0]
        out_latent_float32=transforms.Compose(
                    [
                    transforms.Resize((args.height, args.width), interpolation=transforms.InterpolationMode.BILINEAR),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5])
                    ]
                    )(out).unsqueeze(0).to(device)

        # evaluation  img, out均为[-1,1]
        # clip score
        clip_score = metrics.clip_scores( out_latent_float32,target_prompt)
        print(f"==> clip-T score: {clip_score:.4f}")
        mean_clip_score += clip_score

        clip_v_score = metrics.clip_scores( out_latent_float32,img_float32)
        print(f"==> clip-I score: {clip_v_score:.4f}")
        mean_clip_v_score += clip_v_score
        # mse score
        mse_score = metrics.mse_scores(img_float32, out_latent_float32)
        print(f"==> mse score: {mse_score:.4f}")
        mean_mse_score += mse_score
        #psnr score
        psnr_score = metrics.psnr_scores(img_float32, out_latent_float32)
        print(f"==> psnr score: {psnr_score:.4f}")
        mean_psnr_score += psnr_score
        #lpips score
        lpips_score = metrics.lpips_scores(img_float32, out_latent_float32)
        print(f"==> lpips score: {lpips_score:.4f}")
        mean_lpips_score += lpips_score
        #ssim score
        ssim_score = metrics.ssim_scores(img_float32, out_latent_float32)
        print(f"==> ssim score: {ssim_score:.4f}")
        mean_ssim_score += ssim_score
        #dino score
        dino_score = metrics.dino_scores(img_float32, out_latent_float32)
        print(f"==> dino score: {dino_score:.4f}")
        mean_dino_score += dino_score


        count += 1

        # 保存输出图像
        output_filename = f"num_steps{args.num_steps}_inject{args.inject}_inversed{args.use_inversed_latents}_guidance{args.guidance_scale}.png"
        output_path = os.path.join(args.output_dir, output_filename)
        joint_attention_kwargs['feature'] = {}
        
        # 如果文件重名则自动加序号
        base, ext = os.path.splitext(output_path)
        counter = 1
        while os.path.exists(output_path):
            output_path = f"{base}_{counter}{ext}"
            counter += 1
        out.save(output_path)
        print(f"已保存输出图像到 {output_path}，参数为:num_steps={args.num_steps} inject={args.inject} inversed={args.use_inversed_latents} guidance_scale={args.guidance_scale} nudge={args.nudge}")


    print('######### Evaluation Results ###########')
    mean_clip_score = mean_clip_score / count
    print(f"==> clip-T score: {mean_clip_score:.4f}")
    mean_clip_v_score = mean_clip_v_score / count
    print(f"==> clip-I score: {mean_clip_v_score:.4f}")    
    mean_mse_score = mean_mse_score / count
    print(f"==> mse score: {mean_mse_score:.4f}")
    mean_psnr_score = mean_psnr_score / count
    print(f"==> psnr score: {mean_psnr_score:.4f}")
    mean_lpips_score = mean_lpips_score / count
    print(f"==> lpips score: {mean_lpips_score:.4f}")
    mean_ssim_score = mean_ssim_score / count
    print(f"==> ssim score: {mean_ssim_score:.4f}")
    mean_dino_score = mean_dino_score / count
    print(f"==> dino score: {mean_dino_score:.4f}")
    print('#######################################')

    # 显式删除不再需要的变量
    pipe.maybe_free_model_hooks()
    del out, img_latent, img_latents, inversed_latent_s,inversed_latent_t, pipe, img

    # 强制进行垃圾收集
    gc.collect()

    # 计算并输出运行时间
    end_time = time.time()
    total_time = end_time - start_time
    # 将总秒数转换为小时、分钟、秒
    hours = int(total_time // 3600)
    remaining = total_time % 3600
    minutes = int(remaining // 60)
    seconds = remaining % 60

    print(f"开始时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")
    print(f"结束时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")
    print(f"总运行时间: {hours}时{minutes}分{seconds:.2f}秒")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='使用不同参数测试 interpolated_denoise。')

    # 创建互斥组
    group = parser.add_mutually_exclusive_group(required=True)


    # 数据集模式参数
    group.add_argument('--eval-datasets', type=str, default='', 
                      help='选择要编辑的数据集: EditEval_v1, PIE-Bench')
    parser.add_argument('--ilist_json_path', type=str, default='',
                      help='ilist配置文件路径（数据集模式专用）')
    
    # 单张图片模式参数
    group.add_argument('--image_path', type=str, 
                      help='输入图像路径（单张图片模式必需）')
    parser.add_argument('--source-prompt', type=str,
                      help='源图像描述（单张图片模式必需）')
    parser.add_argument('--target-prompt', type=str,
                      help='编辑目标描述（单张图片模式必需）')
    parser.add_argument('--enhanced_list', type=int, nargs='+', default=[],
                      help='增强层/步骤列表 （用空格分隔多个数字）（单张图片模式专用）')


    # 部分数据选择参数
    parser.add_argument('--num-samples', type=int, 
                    help='处理数据集的前N个样本（仅数据集模式有效）')
    parser.add_argument('--subset-range', type=int, nargs=2, metavar=('START', 'END'),
                    help='处理指定索引范围的数据（包含START，不包含END，仅数据集模式有效）')

    parser.add_argument('--model_path', type=str, default='/root/autodl-tmp/Flux-dev', help='预训练模型的路径')
    parser.add_argument('--output_dir', type=str, default='outputs', help='保存输出图像的目录')
    parser.add_argument('--use_inversed_latents', action='store_true', help='使用反转潜变量')
    parser.add_argument('--guidance_scale', type=float, default=3.5, help='interpolated_denoise 的引导比例')
    parser.add_argument('--num-steps', type=int, default=30, help='时间步长的数量')
    parser.add_argument('--shift', action='store_true', help='在 get_schedule 中使用 shift')

    parser.add_argument('--dtype', type=str, default='bfloat16', choices=['float16', 'bfloat16', 'float32'], help='计算的数据类型')
    
    parser.add_argument('--feature_path', type=str, default='features',
                        help='the path to save the feature ')
    parser.add_argument('--inject', type=int, default=5,
                        help='the number of timesteps which apply the feature sharing')
    parser.add_argument('--height', type=int, default=1024,
                        help='输出图像的高度')
    parser.add_argument('--width', type=int, default=1024,
                        help='输出图像的宽度') 
     
    parser.add_argument('--attn_map_out', action='store_true', help='输出注意力图')
    parser.add_argument('--attn_map_out_path', type=str, default='attn_map_out',
                        help='the path to save the attn_map ')    
    parser.add_argument('--stable_flow', action='store_true', help='切换至stable_flow方法进行inject')
    parser.add_argument('--nudge', type=float, default=1, help='潜变量标量偏移1.15')
    parser.add_argument('--test', action='store_true', help='切换至test 实验方法 进行inject')
    parser.add_argument('--enhanced', action='store_true', help='切换至enhancedt 实验方法 进行多头增强')

    parser.add_argument('--target_var_num', type=int, default=0, help='进行语义丰富的数量')
    
 
    args = parser.parse_args()
    main(args)
