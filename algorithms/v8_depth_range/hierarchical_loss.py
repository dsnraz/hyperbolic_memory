"""
多层级双曲损失函数。

针对层级结构中 1:n 父子包含关系设计的损失函数，
采用子树采样 + 批次内负采样策略。

核心思想：
- 父节点形成蕴涵锥，子节点应落在锥内
- 不同层级在双曲空间中占据不同深度位置
- 批次内其他父节点的子节点作为负样本

支持两种数据格式：
- SubtreeBatch: 子树采样批次（小批量，适合迭代训练）
- GlobalLevelBatch: 全局批次（顶层节点少的情况）
"""

import torch
from torch import nn
import torch.nn.functional as F
from . import lorentz as L
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass


class HierarchicalEntailmentLoss(nn.Module):
    """
    层级蕴涵锥损失 (支持子树采样批次)。
    
    核心思想：
    - 父节点形成蕴涵锥
    - 所有子节点应落在锥内
    - 批次内计算每个父子对的外角和孔径
    
    数学公式：
    L = max(0, angle - aperture * scale)  对每个父子对
    
    其中：
    - angle: 父节点到子节点的外角 (exterior_angle)
    - aperture: 父节点的半孔径角 (half_aperture)
    - 当 angle < aperture 时，子节点在锥内，损失为 0
    """
    
    def __init__(
        self, 
        aperture_scale: float = 1.0,
        level_weights: Optional[Dict[Tuple[str, str], float]] = None
    ):
        """
        参数:
            aperture_scale: 孔径缩放因子，控制锥的大小
            level_weights: 不同层级对的权重
        """
        super().__init__()
        self.aperture_scale = aperture_scale
        self.level_weights = level_weights or {
            ("DOMAIN", "CATEGORY"): 1.0,
            ("CATEGORY", "KEYWORD"): 0.8,
            ("KEYWORD", "DIALOGUE"): 0.6,
        }
    
    def forward(
        self,
        batch,
        curv: float,
    ) -> Dict[str, torch.Tensor]:
        """
        计算蕴涵损失。
        
        参数:
            batch: SubtreeBatch 或 GlobalLevelBatch
            curv: 曲率参数
        
        返回:
            损失字典
        """
        attrs = batch.get_batch_attrs()
        parent_feats = attrs['parent_feats']
        child_feats = attrs['child_feats']
        parent_child_mask = attrs['parent_child_mask']
        parent_level = attrs['parent_level']
        child_level = attrs['child_level']

        positive_pairs = parent_child_mask.nonzero(as_tuple=False)
        if positive_pairs.numel() == 0:
            zero_loss = torch.tensor(0.0, device=parent_feats.device)
            level_pair = (parent_level, child_level)
            weight = self.level_weights.get(level_pair, 1.0)
            return {
                'loss': zero_loss,
                'raw_loss': zero_loss,
                'exterior_angles': torch.empty(0, device=parent_feats.device),
                'half_apertures': torch.empty(0, device=parent_feats.device),
                'in_cone_ratio': 0.0,
                'weight': weight,
                'level_pair': level_pair,
            }

        positive_parent_feats = parent_feats[positive_pairs[:, 0]]
        positive_child_feats = child_feats[positive_pairs[:, 1]]

        # 对所有正父子关系逐对计算外角和孔径
        exterior_angles = L.cone_vertex_exterior_angle_vectors(
            positive_parent_feats, positive_child_feats, curv
        )
        half_apertures = L.half_aperture_vectors(
            positive_parent_feats, curv
        )

        raw_loss = torch.clamp(
            exterior_angles - self.aperture_scale * half_apertures,
            min=0
        )
        
        # 按层级权重加权
        level_pair = (parent_level, child_level)
        weight = self.level_weights.get(level_pair, 1.0)
        
        total_loss = weight * raw_loss.mean()
        
        # 统计信息
        in_cone_count = (exterior_angles < half_apertures).sum().item()
        total_positive_pairs = positive_pairs.shape[0]
        
        return {
            'loss': total_loss,
            'raw_loss': raw_loss.mean(),
            'exterior_angles': exterior_angles.detach(),
            'half_apertures': half_apertures.detach(),
            'in_cone_ratio': in_cone_count / max(total_positive_pairs, 1),
            'weight': weight,
            'level_pair': level_pair,
        }


# ============================================================================
# 层级对比损失（批次内负采样）
# ============================================================================

class HierarchicalContrastiveLoss(nn.Module):
    """
    层级对比损失（批次内负采样版本）。
    
    对于子树采样批次：
    - 正样本：父节点与其真实子节点（距离应小）
    - 负样本：batch内其他父节点的子节点（距离应大）
    
    对于全局批次：
    - 同样的逻辑，但包含所有节点
    
    核心改进：
    - 使用软标签处理 1:n 关系（一个父节点有多个子节点）
    - 使用硬标签处理 n:1 关系（一个子节点只有一个父节点）
    """
    
    def __init__(self, temperature: float = 0.1):
        """
        参数:
            temperature: 温度参数，控制softmax的平滑度
        """
        super().__init__()
        self.temperature = temperature
    
    def forward(
        self,
        batch,
        curv: torch.Tensor,
        logit_scale: torch.Tensor = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """
        计算对比损失。
        
        参数:
            batch: SubtreeBatch 或 GlobalLevelBatch
            curv: 曲率
            logit_scale: logits 缩放因子
        
        返回:
            损失字典
        """
        attrs = batch.get_batch_attrs()
        parent_feats = attrs['parent_feats']
        child_feats = attrs['child_feats']
        parent_child_mask = attrs['parent_child_mask']
        
        # 计算双曲距离矩阵
        dist_matrix = L.pairwise_dist_vectors(parent_feats, child_feats, curv)
        # dist_matrix: (N_parent, N_child)
        
        # 转换为 logits：距离越小，相似度越高
        logits = -dist_matrix / self.temperature
        
        # 父节点视角（软标签）
        parent_child_counts = parent_child_mask.sum(dim=1).clamp(min=1)
        soft_labels_parent = parent_child_mask.float() / parent_child_counts.unsqueeze(1)
        loss_parent = self._soft_cross_entropy(logit_scale * logits, soft_labels_parent)
        
        # 子节点视角（多父软标签）
        child_parent_counts = parent_child_mask.sum(dim=0).clamp(min=1)
        soft_labels_child = parent_child_mask.float().T / child_parent_counts.unsqueeze(1)
        loss_child = self._soft_cross_entropy(logit_scale * logits.T, soft_labels_child)
        
        total_loss = 0.5 * (loss_parent + loss_child)
        
        # 计算准确率（用于监控）
        with torch.no_grad():
            # 父节点视角：预测最相似的子节点是否真的是其子节点
            pred_child_idx = logits.argmax(dim=1)
            correct_parent = 0
            for p_idx, pred_c_idx in enumerate(pred_child_idx):
                if parent_child_mask[p_idx, pred_c_idx] > 0:
                    correct_parent += 1
            parent_accuracy = correct_parent / max(parent_feats.shape[0], 1)
            
            # 子节点视角：预测最相似的父节点是否属于任一真实父节点
            pred_parent_idx = logits.T.argmax(dim=1)
            child_indices = torch.arange(child_feats.shape[0], device=pred_parent_idx.device)
            child_accuracy = parent_child_mask[pred_parent_idx, child_indices].float().mean().item()
        
        return {
            'loss': total_loss,
            'loss_parent_view': loss_parent.item(),
            'loss_child_view': loss_child.item(),
            'parent_accuracy': parent_accuracy,
            'child_accuracy': child_accuracy,
            'distance_matrix': dist_matrix.detach(),
            'logits': logits.detach(),
        }
    
    def _soft_cross_entropy(self, logits: torch.Tensor, soft_labels: torch.Tensor) -> torch.Tensor:
        """软标签交叉熵损失。"""
        log_probs = F.log_softmax(logits, dim=-1)
        loss = -(soft_labels * log_probs).sum(dim=-1).mean()
        return loss


# ============================================================================
# 层级角度对比损失（批次内负采样）
# ============================================================================

class HierarchicalAngularContrastiveLoss(nn.Module):
    """
    层级角度对比损失（批次内负采样版本）。
    
    核心思想：
    1. 使用外角而非距离作为相似度度量
    2. 批次内负采样：其他父节点的子节点作为负样本
    3. 包含质心深度正则化
    
    数学公式：
    - alpha = 外角矩阵 (父节点视角)
    - beta = π - alpha (补角)
    - 正样本：外角小，补角大
    - 负样本：外角大，补角小
    """
    
    def __init__(
        self,
        lambda_centroid: float = 0.1,
        level_depth_targets: Optional[Dict[str, float]] = None,
        lambda_soft_norm: float = 0.0,          # v8_depth_range: 默认关闭 d_v 目标软 MSE
        lambda_rank: float = 0.0,               # v8_depth_range: 默认关闭层间顺序 margin
        rank_margin: float = 0.1,               # v8_depth_range
    ):
        """
        参数:
            lambda_centroid: 质心正则化权重
            level_depth_targets: 各层级目标深度比例
            lambda_soft_norm: v8_depth_range 新增。软模长约束权重（配合去掉硬投影使用）。
            lambda_rank: v8_depth_range 新增。层间质心顺序 margin 权重，默认 0 不启用。
            rank_margin: v8_depth_range 新增。层间 margin 大小。
        """
        super().__init__()
        self.lambda_centroid = lambda_centroid
        self.level_depth_targets = level_depth_targets or {
            "DOMAIN": 0.1,
            "CATEGORY": 0.3,
            "KEYWORD": 0.5,
            "DIALOGUE": 0.7,
        }
        self.eps = 1e-6
        self.lambda_soft_norm = lambda_soft_norm     # v8_depth_range
        self.lambda_rank = lambda_rank               # v8_depth_range
        self.rank_margin = rank_margin               # v8_depth_range
    
    def forward(
        self,
        batch,
        curv: float,
        logit_scale: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """
        计算角度对比损失 + 质心正则化。
        
        参数:
            batch: SubtreeBatch 或 GlobalLevelBatch
            curv: 曲率
            logit_scale: logits 缩放
        
        返回:
            损失字典
        """
        attrs = batch.get_batch_attrs()
        parent_feats = attrs['parent_feats']
        child_feats = attrs['child_feats']
        parent_child_mask = attrs['parent_child_mask']
        parent_level = attrs['parent_level']
        child_level = attrs['child_level']
        
        R = 1.0 / torch.sqrt(curv + self.eps)

        
        
        # --- 1. 角度对比损失 ---
        alpha_matrix = L.pairwise_exterior_angle_vectors(
            parent_feats, child_feats, curv, self.eps
        )  # (N_parent, N_child)
        
        beta_matrix = torch.pi - alpha_matrix
        
        # 构建 logits
        logits_alpha_parent = -alpha_matrix
        logits_beta_parent = beta_matrix
        logits_alpha_child = -alpha_matrix.T
        logits_beta_child = beta_matrix.T
        
        # 软标签（父节点视角）
        parent_child_counts = parent_child_mask.sum(dim=1).clamp(min=1)
        soft_labels_parent = parent_child_mask.float() / parent_child_counts.unsqueeze(1)
        child_parent_counts = parent_child_mask.sum(dim=0).clamp(min=1)
        soft_labels_child = parent_child_mask.float().T / child_parent_counts.unsqueeze(1)
        
        
        # 计算四个方向的损失
        loss_alpha_parent = self._soft_cross_entropy(logit_scale * logits_alpha_parent, soft_labels_parent)
        loss_alpha_child = self._soft_cross_entropy(logit_scale * logits_alpha_child, soft_labels_child)
        loss_beta_parent = self._soft_cross_entropy(logit_scale * logits_beta_parent, soft_labels_parent)
        loss_beta_child = self._soft_cross_entropy(logit_scale * logits_beta_child, soft_labels_child)
        
        loss_angle = 0.25 * (loss_alpha_parent + loss_alpha_child + loss_beta_parent + loss_beta_child)
        
        # --- 2. 质心深度正则化 ---
        if parent_feats.shape[0] > 0:
            parent_centroid = L.lorentz_midpoint_sequences(parent_feats.unsqueeze(0), curv, keep_dim=False)
            origin = torch.zeros_like(parent_centroid)
            dist_parent_centroid = L.pairwise_dist_vectors(
                parent_centroid, origin, curv
            ).squeeze()
        else:
            dist_parent_centroid = torch.tensor(0.0, device=parent_feats.device)
        
        if child_feats.shape[0] > 0:
            child_centroid = L.lorentz_midpoint_sequences(child_feats.unsqueeze(0), curv, keep_dim=False)
            origin = torch.zeros_like(child_centroid)
            dist_child_centroid = L.pairwise_dist_vectors(
                child_centroid, origin, curv
            ).squeeze()
        else:
            dist_child_centroid = torch.tensor(0.0, device=child_feats.device)
        
        parent_target = self.level_depth_targets.get(parent_level, self.level_depth_targets[parent_level]) * R
        child_target = self.level_depth_targets.get(child_level, self.level_depth_targets[child_level]) * R

        loss_centroid = torch.abs(dist_parent_centroid - parent_target) + torch.abs(dist_child_centroid - child_target)

        # ====================================================================
        # v8_depth_range: 层间质心顺序 margin（默认 lambda_rank=0 不启用）
        # 期望：parent 比 child 浅（深度更小）。若违反，加惩罚。
        # ====================================================================
        loss_rank = torch.tensor(0.0, device=parent_feats.device)
        if self.lambda_rank > 0:
            # parent 应更浅：dist_parent < dist_child - margin
            loss_rank = F.relu(dist_parent_centroid - dist_child_centroid + self.rank_margin)

        # ====================================================================
        # v8_depth_range: 软模长 MSE
        # 使用 z_hat_E（exp_map0 前的切空间向量）的实际模长对比 target_norm，
        # 避免 sinh/asinh 的非线性。projector 会把这两个量挂到 projector 属性，
        # train.py 再转挂到 batch 上。
        # ====================================================================
        loss_soft_norm = torch.tensor(0.0, device=parent_feats.device)
        parent_target_norm = getattr(batch, 'parent_target_norm', None)
        child_target_norm = getattr(batch, 'child_target_norm', None)
        parent_actual_pre = getattr(batch, 'parent_actual_norm_pre_exp', None)
        child_actual_pre = getattr(batch, 'child_actual_norm_pre_exp', None)
        if (self.lambda_soft_norm > 0
                and parent_target_norm is not None and child_target_norm is not None
                and parent_actual_pre is not None and child_actual_pre is not None):
            loss_soft_norm = (
                F.mse_loss(parent_actual_pre, parent_target_norm.detach())
                + F.mse_loss(child_actual_pre, child_target_norm.detach())
            )

        # --- 3. 总损失 ---
        total_loss = (
            loss_angle
            + self.lambda_centroid * loss_centroid
            + self.lambda_rank * loss_rank
            + self.lambda_soft_norm * loss_soft_norm
        )

        return {
            'loss': total_loss,
            'angle_loss': loss_angle.item(),
            'centroid_loss': loss_centroid.item(),
            'rank_loss': float(loss_rank.item()) if isinstance(loss_rank, torch.Tensor) else 0.0,
            'soft_norm_loss': float(loss_soft_norm.item()) if isinstance(loss_soft_norm, torch.Tensor) else 0.0,
            'alpha_matrix': alpha_matrix.detach(),
            'beta_matrix': beta_matrix.detach(),
            'parent_centroid_depth': dist_parent_centroid.item() if isinstance(dist_parent_centroid, torch.Tensor) else 0,
            'child_centroid_depth': dist_child_centroid.item() if isinstance(dist_child_centroid, torch.Tensor) else 0,
            'level_pair': (parent_level, child_level),
        }
    
    def _soft_cross_entropy(self, logits: torch.Tensor, soft_labels: torch.Tensor) -> torch.Tensor:
        """软标签交叉熵。"""
        log_probs = F.log_softmax(logits, dim=-1)
        loss = -(soft_labels * log_probs).sum(dim=-1).mean()
        return loss