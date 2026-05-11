"""
双曲投影器训练脚本。

训练流程：
1. 逐层级对进行训练（DOMAIN→CATEGORY → CATEGORY→KEYWORD → KEYWORD→DIALOGUE）
2. 每个层级对只加载该层级对需要的节点数据
3. 创建 Hyperbolic_projector 模型
4. 设置损失函数组合
5. 迭代训练优化

使用方式：
    # 逐层级对训练（推荐）
    python -m model.hyperbolic_utils.train \
        --vector_store_path ./data/vector_store \
        --output_dir ./checkpoints \
    
    # 训练特定层级对
    python -m model.hyperbolic_utils.train \
        --level_pair_index 2 \
        --num_iterations 3000
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import argparse

from model.stores.hierarchical_vector_store import HierarchicalVectorStore
from model.hierarchical.hierarchy_types import HierarchyLevel
from model.hyperbolic_utils.hyperbolic_projector import Hyperbolic_projector
from model.hyperbolic_utils.hierarchical_dataset import (
    SubtreeDataset,
    SubtreeBatch,
    extract_nodes_from_store,
    LEVEL_PAIRS,
    subtree_collate_fn
)
from model.hyperbolic_utils.hierarchical_loss import (
    HierarchicalEntailmentLoss,
    HierarchicalContrastiveLoss,
    HierarchicalAngularContrastiveLoss,
)


# ============================================================================
# 训练参数配置
# ============================================================================

@dataclass
class TrainConfig:
    """训练配置参数。"""
    
    # 数据相关
    vector_store_path: str = "./data/vector_store"
    embedding_dim: int = 384
    hidden_dim: int = 256
    
    # 采样相关
    num_iterations: int = 5000               # 单层级对训练时的迭代次数
    iterations_map: Dict[int, int] = field(default_factory=lambda: {
        1: 5000,   # 顶层节点少，5000步足够
        2: 15000,  # 中间层，数据量增加
        3: 30000   # 底层，针对你13万Keyword的数据量，建议至少3万步
    })
    
    #num_iterations_per_level: int = 2000     # 逐层级对训练时每个层级对的迭代次数
    num_parents_per_batch: int = 16
    num_children_per_parent: int = 4
    max_children_per_parent: int = 10
    
    # 模型相关
    initial_curvature: float = 0.1
    alpha: float = 0.1          # 最小半径比例
    beta: float = 0.8           # 最大半径比例
    
    # 损失权重
    entailment_weight: float = 0
    contrastive_weight: float = 0
    angular_weight: float = 1
    lambda_centroid: float = 0.3
    
    # 训练相关
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    logit_scale: float = 2.6592
    aperture_scale: float = 1.0
    use_level_embedding: bool = False
    
    # 设备与输出
    device: str = "cuda"
    output_dir: str = "./checkpoints"
    log_interval: int = 100
    save_interval: int = 500
    
    # 训练模式
    level_pair_index: Optional[int] = None   # None=逐层级对训练, 1/2/3=特定层级对
    sequential_levels: bool = True           # 是否逐层级对顺序训练
    resume: Optional[str] = None


# ============================================================================
# 训练器类
# ============================================================================

class HyperbolicTrainer:
    """
    双曲投影器训练器。
    
    支持两种训练模式：
    1. 逐层级对训练：依次训练 DOMAIN→CATEGORY, CATEGORY→KEYWORD, KEYWORD→DIALOGUE
    2. 单层级对训练：只训练指定的层级对
    
    每次训练只加载当前层级对需要的节点数据，减少内存占用。
    """
    
    def __init__(self, config: TrainConfig):
        """初始化训练器。"""
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        
        # 训练状态记录
        self.global_step = 0
        self.level_step = 0        # 当前层级对的步数
        self.current_level_idx = 0  # 当前层级对索引
        self.best_loss = float('inf')
        self.loss_history: List[Dict] = []
        
        # 加载向量库（不提取节点，只加载存储）
        print(f"正在加载向量库: {self.config.vector_store_path}")
        self.vector_store = HierarchicalVectorStore(
            persist_directory=self.config.vector_store_path,
            embedding_function=None,
            delayed_write=False,)
        
        self._setup_model_and_losses()
    
    def _setup_model_and_losses(self):
        """设置模型和损失函数。"""
        # 创建模型
        self.model = Hyperbolic_projector(
            input_dim=self.config.embedding_dim,
            hidden_dim=self.config.hidden_dim,
            curvature=self.config.initial_curvature,
            alpha=self.config.alpha,
            beta=self.config.beta,
        ).to(self.device)
        
        # 初始化 logit_scale
        self.model.logit_scale.data.fill_(self.config.logit_scale)
        
        # 创建损失函数
        self.entailment_loss = HierarchicalEntailmentLoss(
            aperture_scale=self.config.aperture_scale
        ).to(self.device)
        
        self.contrastive_loss = HierarchicalContrastiveLoss(
            temperature=0.1
        ).to(self.device)
        
        self.angular_loss = HierarchicalAngularContrastiveLoss(
            lambda_centroid=self.config.lambda_centroid,
        ).to(self.device)
        
        
        # 创建优化器
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        
        print(f"模型和损失函数初始化完成")
        print(f"  模型参数量: {sum(p.numel() for p in self.model.parameters())}")
    
    def _setup_dataset_for_level(self, level_pair_index: int) -> SubtreeDataset:
        """
        为特定层级对设置数据集。
        
        只加载该层级对需要的节点数据，减少内存占用。
        
        参数:
            level_pair_index: 层级对索引 (1, 2, 3)
        
        返回:
            配置好的 SubtreeDataset
        """
        level_pair = LEVEL_PAIRS[level_pair_index - 1]
        parent_level, child_level = level_pair
        
        print(f"\n正在加载层级对 {level_pair_index}: {parent_level} → {child_level}")
        
        # 只提取该层级对的节点（不是全部节点）
        nodes_by_level = extract_nodes_from_store(
            self.vector_store,
            level_pair_index=level_pair_index
        )
        
        # 统计节点数量
        print(f"  {parent_level}: {len(nodes_by_level.get(parent_level, []))} 个节点")
        print(f"  {child_level}: {len(nodes_by_level.get(child_level, []))} 个节点")
        
        # 创建数据集
        num_iterations = self.config.iterations_map[level_pair_index] if self.config.sequential_levels else self.config.num_iterations
        
        dataset = SubtreeDataset(
            nodes_by_level=nodes_by_level,
            embedding_dim=self.config.embedding_dim,
            device=self.device,
            num_iterations=num_iterations,
            num_parents_per_batch=self.config.num_parents_per_batch,
            num_children_per_parent=self.config.num_children_per_parent,
            max_children_per_parent=self.config.max_children_per_parent,
            level_pair=level_pair,           # 固定层级对
            load_feats_by_level=False,       # 不预加载全部节点特征
            use_level_embedding=self.config.use_level_embedding,
        )
        
        return dataset
    
    def _create_scheduler(self, total_iterations: int):
        """创建学习率调度器。"""
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=total_iterations,
            eta_min=1e-6,
        )
    
    def _process_batch_feats(self, batch: SubtreeBatch) -> Tuple[torch.Tensor, torch.Tensor]:
        """将批次特征通过 projector 投影到双曲空间。"""
        _, parent_feats_H = self.model(batch.parent_feats)
        _, child_feats_H = self.model(batch.child_feats)
        return parent_feats_H, child_feats_H
    
    def _compute_losses(
        self,
        batch: SubtreeBatch,
        parent_feats_H: torch.Tensor,
        child_feats_H: torch.Tensor,
        curv: torch.Tensor
    ) -> Dict[str, object]:
        """按当前配置计算启用的损失并动态返回结果。"""
        # 构建投影后的批次
        projected_batch = SubtreeBatch(
            parent_level=batch.parent_level,
            child_level=batch.child_level,
            parent_feats=parent_feats_H,
            child_feats=child_feats_H,
            parent_child_mask=batch.parent_child_mask,
            parent_child_map=batch.parent_child_map,
            parent_ids=batch.parent_ids,
            child_ids=batch.child_ids,
            n_parent=batch.n_parent,
            n_child=batch.n_child,
        )

        losses: Dict[str, object] = {
            'curvature': curv,
            'level_pair': (batch.parent_level, batch.child_level),
        }
        total_loss = None

        # 计算各损失，只记录当前启用的部分
        if self.config.entailment_weight > 0:
            entailment_out = self.entailment_loss(projected_batch, curv)
            losses['entailment_loss'] = entailment_out['loss']
            losses['in_cone_ratio'] = entailment_out['in_cone_ratio']
            losses['level_pair'] = entailment_out.get('level_pair', losses['level_pair'])
            weighted_entailment_loss = self.config.entailment_weight * entailment_out['loss']
            total_loss = weighted_entailment_loss if total_loss is None else total_loss + weighted_entailment_loss

        if self.config.contrastive_weight > 0:
            contrastive_out = self.contrastive_loss(
                projected_batch, curv,
                logit_scale=self.model.logit_scale.item()
            )
            losses['contrastive_loss'] = contrastive_out['loss']
            losses['parent_accuracy'] = contrastive_out['parent_accuracy']
            losses['child_accuracy'] = contrastive_out['child_accuracy']
            weighted_contrastive_loss = self.config.contrastive_weight * contrastive_out['loss']
            total_loss = weighted_contrastive_loss if total_loss is None else total_loss + weighted_contrastive_loss

        if self.config.angular_weight > 0:
            angular_out = self.angular_loss(
                projected_batch, curv,
                logit_scale=self.model.logit_scale
            )
            losses['angular_loss'] = angular_out['loss']
            losses['level_pair'] = angular_out.get('level_pair', losses['level_pair'])
            weighted_angular_loss = self.config.angular_weight * angular_out['loss']
            total_loss = weighted_angular_loss if total_loss is None else total_loss + weighted_angular_loss

        if total_loss is None:
            raise ValueError("至少需要启用一个损失函数，请检查 loss weight 配置。")

        losses['total_loss'] = total_loss
        return losses
    
    def train_step(self, batch: SubtreeBatch) -> Dict[str, float]:
        """执行单步训练。"""
        self.model.train()
        self.optimizer.zero_grad()
        
        # 获取当前曲率
        curv = torch.nn.functional.softplus(self.model.c)

        with torch.autograd.detect_anomaly():
            curv = torch.nn.functional.softplus(self.model.c)
            parent_feats_H, child_feats_H = self._process_batch_feats(batch)
            losses = self._compute_losses(batch, parent_feats_H, child_feats_H, curv)
            

            # 反向传播
            losses['total_loss'].backward()
        print(f"\n" + "="*80)
        print(f"Step {self.global_step} 完整梯度清单:")
        
        with torch.no_grad():
            # 遍历模型中每一个包含参数的层
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    g_min = param.grad.min().item()
                    g_max = param.grad.max().item()
                    g_norm = param.grad.norm().item()
                    
                    # 检查是否包含 NaN
                    status = "OK"
                    if torch.isnan(param.grad).any():
                        status = "!!! NaN !!!"
                    elif torch.isinf(param.grad).any():
                        status = "!!! INF !!!"
                    
                    # 打印格式：参数名 | 范数 | 最大值 | 最小值 | 状态
                    print(f"层: {name:<30} | Norm: {g_norm:.6e} | Max: {g_max:.4e} | Min: {g_min:.4e} | {status}")
                else:
                    print(f"层: {name:<30} | 无梯度 (None)")

            # 特别输出全局范数
            total_grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1e10)
            print(f"\n[GLOBAL] 全局梯度范数: {total_grad_norm:.6f}")
        print("="*80 + "\n", flush=True)

        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        # 更新参数
        self.optimizer.step()
        self.scheduler.step()
        
        self.global_step += 1
        self.level_step += 1
        
        # 转换为 float 记录
        stats = {
            'global_step': self.global_step,
            'level_step': self.level_step,
            'level_idx': self.current_level_idx,
            'total_loss': losses['total_loss'].item(),
            'curvature': losses['curvature'].detach().item(),
            'lr': self.optimizer.param_groups[0]['lr'],
            'level_pair': str(losses['level_pair']),
        }

        optional_tensor_keys = ['entailment_loss', 'contrastive_loss', 'angular_loss']
        for key in optional_tensor_keys:
            if key in losses:
                stats[key] = losses[key].item()

        optional_scalar_keys = ['in_cone_ratio', 'parent_accuracy', 'child_accuracy']
        for key in optional_scalar_keys:
            if key in losses:
                stats[key] = losses[key]
        
        self.loss_history.append(stats)
        
        return stats
    
    def train_level_pair(self, level_pair_index: int):
        """
        训练单个层级对。
        
        参数:
            level_pair_index: 层级对索引 (1, 2, 3)
        """
        self.current_level_idx = level_pair_index
        self.level_step = 0
        
        level_pair = LEVEL_PAIRS[level_pair_index - 1]
        parent_level, child_level = level_pair
        
        print(f"\n{'='*60}")
        print(f"开始训练层级对 {level_pair_index}: {parent_level} → {child_level}")
        print(f"{'='*60}")
        
        # 设置数据集（只加载该层级对的节点）
        dataset = self._setup_dataset_for_level(level_pair_index)
        
        # 创建 DataLoader
        dataloader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=True,
            num_workers=0,
            collate_fn=subtree_collate_fn
        )
        
        # 设置学习率调度器
        num_iterations = self.config.iterations_map[level_pair_index] if self.config.sequential_levels else self.config.num_iterations
        self._create_scheduler(num_iterations)
        
        # 训练进度条
        pbar = tqdm(
            dataloader,
            total=num_iterations,
            desc=f"训练 {parent_level}→{child_level}",
        )
        
        for batch in pbar:
            if isinstance(batch, list):
                batch = batch[0]
            
            stats = self.train_step(batch)

            postfix = {
                'loss': f"{stats['total_loss']:.4f}",
                'c': f"{stats['curvature']:.4f}",
            }
            if 'in_cone_ratio' in stats:
                postfix['cone'] = f"{stats['in_cone_ratio']:.2f}"
            if 'child_accuracy' in stats:
                postfix['acc'] = f"{stats['child_accuracy']:.2f}"
            pbar.set_postfix(postfix)
            
            if self.level_step % self.config.log_interval == 0:
                self._log_stats(stats)
            
            if self.level_step % self.config.save_interval == 0:
                self._save_checkpoint(level_info=f"level{level_pair_index}_step{self.level_step}")
    
    def train_sequential(self):
        """
        逐层级对顺序训练。
        
        训练顺序：
        1. DOMAIN → CATEGORY (顶层，节点少)
        2. CATEGORY → KEYWORD (中层)
        3. KEYWORD → DIALOGUE (底层，节点多)
        
        每个层级对训练完成后保存检查点。
        """
        print(f"\n{'='*60}")
        print(f"开始逐层级对训练")
        print(f"{'='*60}")
        print(f"训练顺序:")
        print(f"  1. DOMAIN → CATEGORY ({self.config.iterations_map[1]} 步)")
        print(f"  2. CATEGORY → KEYWORD ({self.config.iterations_map[2]} 步)")
        print(f"  3. KEYWORD → DIALOGUE ({self.config.iterations_map[3]} 步)")
        
        # 逐层级对训练
        for level_idx in [1, 2, 3]:
            self.train_level_pair(level_idx)
            
            # 每个层级对完成后保存
            self._save_checkpoint(level_info=f"level{level_idx}_final", final_level=True)
        
        # 全部完成后保存最终模型
        self._save_checkpoint(final=True)
        print(f"\n全部层级对训练完成!")
    
    def train_single_level(self):
        """训练单个层级对。"""
        level_idx = self.config.level_pair_index
        self.train_level_pair(level_idx)
        self._save_checkpoint(final=True)
        print(f"\n层级对 {level_idx} 训练完成!")
    
    def train(self):
        """执行训练。"""
        if self.config.level_pair_index is not None:
            # 单层级对训练
            self.train_single_level()
        else:
            # 逐层级对训练
            self.train_sequential()
    
    def _log_stats(self, stats: Dict[str, float]):
        """记录训练统计。"""
        level_pair = stats['level_pair']
        print(f"\n[Step {stats['global_step']}] "
              f"层级对 {stats['level_idx']}: {level_pair}")
        print(f"  层级步数: {stats['level_step']}")
        print(f"  总损失: {stats['total_loss']:.4f}")

        if 'entailment_loss' in stats:
            entailment_msg = f"  ├─ 蕴涵损失: {stats['entailment_loss']:.4f}"
            if 'in_cone_ratio' in stats:
                entailment_msg += f" (锥内比例: {stats['in_cone_ratio']:.2%})"
            print(entailment_msg)

        if 'contrastive_loss' in stats:
            contrastive_msg = f"  ├─ 对比损失: {stats['contrastive_loss']:.4f}"
            metric_parts = []
            if 'parent_accuracy' in stats:
                metric_parts.append(f"父准确率: {stats['parent_accuracy']:.2%}")
            if 'child_accuracy' in stats:
                metric_parts.append(f"子准确率: {stats['child_accuracy']:.2%}")
            if metric_parts:
                contrastive_msg += f" ({', '.join(metric_parts)})"
            print(contrastive_msg)

        if 'angular_loss' in stats:
            print(f"  ├─ 角度损失: {stats['angular_loss']:.4f}")

        print(f"  曲率: {stats['curvature']:.4f}, "
              f"学习率: {stats['lr']:.6f}")

    def _prune_old_checkpoints(self, keep_paths: List[str]) -> None:
        """删除旧 checkpoint，仅保留指定文件。"""
        if not os.path.isdir(self.config.output_dir):
            return

        keep_abs_paths = {os.path.abspath(path) for path in keep_paths}
        for filename in os.listdir(self.config.output_dir):
            if not (filename.startswith("hyperbolic_projector") and filename.endswith(".pt")):
                continue

            file_path = os.path.abspath(os.path.join(self.config.output_dir, filename))
            if file_path in keep_abs_paths:
                continue

            try:
                os.remove(file_path)
                print(f"已删除旧检查点: {file_path}")
            except OSError as exc:
                print(f"删除旧检查点失败: {file_path}, 错误: {exc}")
    
    def _save_checkpoint(self, level_info: str = "", final: bool = False, final_level: bool = False):
        """保存训练检查点。"""
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        if final:
            filename = "hyperbolic_projector_final.pt"
        elif final_level:
            filename = f"hyperbolic_projector_{level_info}.pt"
        else:
            filename = f"hyperbolic_projector_{level_info}.pt"
        
        path = os.path.join(self.config.output_dir, filename)
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if hasattr(self, 'scheduler') else None,
            'global_step': self.global_step,
            'current_level_idx': self.current_level_idx,
            'level_step': self.level_step,
            'config': self.config.__dict__,
            'curvature': torch.nn.functional.softplus(self.model.c).item(),
        }
        
        torch.save(checkpoint, path)
        self._prune_old_checkpoints([path])

        print(f"检查点已保存: {path}")
        
        # 保存损失历史
        history_path = os.path.join(self.config.output_dir, "loss_history.json")
        with open(history_path, 'w') as f:
            json.dump(self.loss_history, f, indent=2)
    
    def load_checkpoint(self, path: str):
        """加载训练检查点。"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if checkpoint.get('scheduler_state_dict'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.global_step = checkpoint['global_step']
        self.current_level_idx = checkpoint.get('current_level_idx', 1)
        self.level_step = checkpoint.get('level_step', 0)
        
        print(f"检查点已加载: {path}")
        print(f"  全局步骤: {self.global_step}")
        print(f"  当前层级对: {self.current_level_idx}, 步骤: {self.level_step}")


# ============================================================================
# 主函数
# ============================================================================

def parse_args() -> TrainConfig:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="双曲投影器训练")
    
    # 数据参数
    parser.add_argument('--vector_store_path', type=str, default='/share/home/leiyh5/Memory/data/hierarchical_memory_locomo_category2')
    parser.add_argument('--embedding_dim', type=int, default=768)
    parser.add_argument('--hidden_dim', type=int, default=1024)
    
    # 采样参数
    parser.add_argument('--num_iterations', type=int, default=5000,
                        help='单层级对训练时的迭代次数')
    parser.add_argument('--iterations_map', type=Dict, default={1: 500,2: 4000,3: 8000},\
                        help='逐层级对训练时每个层级对的迭代次数')
    parser.add_argument('--num_parents_per_batch', type=int, default=16)
    parser.add_argument('--num_children_per_parent', type=int, default=4)
    parser.add_argument('--max_children_per_parent', type=int, default=10)
    
    # 模型参数
    parser.add_argument('--initial_curvature', type=float, default=0.1)
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--beta', type=float, default=0.8)
    
    # 损失权重
    parser.add_argument('--entailment_weight', type=float, default=0)
    parser.add_argument('--contrastive_weight', type=float, default=0)
    parser.add_argument('--angular_weight', type=float, default=1)
    parser.add_argument('--lambda_centroid', type=float, default=0.3,
                        help='centroid depth regularization weight')
    
    # 训练参数
    parser.add_argument('--learning_rate', type=float, default=1e-6)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--logit_scale', type=float, default=2.6592)
    parser.add_argument('--aperture_scale', type=float, default=1.0)
    parser.add_argument('--use_level_embedding', action='store_true',
                        help='训练时优先使用节点的 level_embedding')
    
    # 设备与输出
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--output_dir', type=str, default='./checkpoints_locomo_categorymorefact_c0p1_la0p3')
    parser.add_argument('--log_interval', type=int, default=100)
    parser.add_argument('--save_interval', type=int, default=500)
    
    # 训练模式
    parser.add_argument('--level_pair_index', type=int, default=None,
                        help='单层级对训练: 1=DOMAIN→CATEGORY, '
                             '2=CATEGORY→KEYWORD, 3=KEYWORD→DIALOGUE')
    
    # 恢复训练
    parser.add_argument('--resume', type=str, default="/share/home/leiyh5/Memory/checkpoints_locomo_categorymorefact_c0p1_la0p3",
                        help='恢复训练的检查点路径')
    
    args = parser.parse_args()
    
    return TrainConfig(
        vector_store_path=args.vector_store_path,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_iterations=args.num_iterations,
        iterations_map=args.iterations_map,
        num_parents_per_batch=args.num_parents_per_batch,
        num_children_per_parent=args.num_children_per_parent,
        max_children_per_parent=args.max_children_per_parent,
        initial_curvature=args.initial_curvature,
        alpha=args.alpha,
        beta=args.beta,
        entailment_weight=args.entailment_weight,
        contrastive_weight=args.contrastive_weight,
        angular_weight=args.angular_weight,
        lambda_centroid=args.lambda_centroid,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        logit_scale=args.logit_scale,
        aperture_scale=args.aperture_scale,
        use_level_embedding=args.use_level_embedding,
        device=args.device,
        output_dir=args.output_dir,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        level_pair_index=args.level_pair_index,
        sequential_levels=True,
    )


def main():
    """主函数入口。"""
    config = parse_args()
    
    print(f"\n{'='*60}")
    print(f"训练配置")
    print(f"{'='*60}")
    print(f"  向量库路径: {config.vector_store_path}")
    print(f"  嵌入维度: {config.embedding_dim}")
    print(f"  隐藏维度: {config.hidden_dim}")
    
    if config.level_pair_index is not None:
        level_pair = LEVEL_PAIRS[config.level_pair_index - 1]
        print(f"  训练模式: 单层级对 ({level_pair[0]}→{level_pair[1]})")
        print(f"  迭代次数: {config.num_iterations}")
    else:
        print(f"  训练模式: 逐层级对训练")
        print(f"  每层级对迭代次数: {config.iterations_map}")
    
    print(f"  每批次父节点数: {config.num_parents_per_batch}")
    print(f"  初始曲率: {config.initial_curvature}")
    print(f"  损失权重: 蕴涵={config.entailment_weight}, "
          f"对比={config.contrastive_weight}, "
          f"角度={config.angular_weight}")
    print(f"  特征类型: {'level_embedding' if config.use_level_embedding else 'embedding'}")
    print(f"  输出目录: {config.output_dir}")
    
    trainer = HyperbolicTrainer(config)
    
    # 恢复训练
    resume_path = parse_args().resume
    if resume_path:
        trainer.load_checkpoint(resume_path)
    
    trainer.train()


if __name__ == "__main__":
    main()