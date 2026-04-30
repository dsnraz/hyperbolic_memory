from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.hyperbolic_utils.hierarchical_loss import (
    HierarchicalAngularContrastiveLoss,
    HierarchicalContrastiveLoss,
    HierarchicalEntailmentLoss,
)
from model.hyperbolic_utils.hyperbolic_projector import Hyperbolic_projector
from model.hyperbolic_utils.no_category_hierarchical_dataset import (
    NO_CATEGORY_LEVEL_PAIRS,
    NoCategorySubtreeDataset,
    extract_no_category_nodes_from_store,
    subtree_collate_fn,
)
from model.stores.hierarchical_vector_store import HierarchicalVectorStore


@dataclass
class NoCategoryTrainConfig:
    vector_store_path: str = "./data/vector_store"
    embedding_dim: int = 384
    hidden_dim: int = 256
    num_iterations: int = 5000
    iterations_map: Dict[int, int] = field(default_factory=lambda: {1: 8000, 2: 16000})
    num_parents_per_batch: int = 16
    num_children_per_parent: int = 4
    max_children_per_parent: int = 10
    initial_curvature: float = 0.1
    alpha: float = 0.1
    beta: float = 0.8
    entailment_weight: float = 0.0
    contrastive_weight: float = 0.0
    angular_weight: float = 1.0
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    logit_scale: float = 2.6592
    aperture_scale: float = 1.0
    use_level_embedding: bool = False
    device: str = "cuda"
    output_dir: str = "./checkpoints_no_category"
    log_interval: int = 100
    save_interval: int = 500
    level_pair_index: Optional[int] = None
    sequential_levels: bool = True
    resume: Optional[str] = None


class NoCategoryHyperbolicTrainer:
    def __init__(self, config: NoCategoryTrainConfig):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.global_step = 0
        self.level_step = 0
        self.current_level_idx = 0
        self.loss_history: List[Dict] = []
        self.vector_store = HierarchicalVectorStore(
            persist_directory=self.config.vector_store_path,
            embedding_function=None,
            delayed_write=False,
        )
        self.model = Hyperbolic_projector(
            input_dim=self.config.embedding_dim,
            hidden_dim=self.config.hidden_dim,
            curvature=self.config.initial_curvature,
            alpha=self.config.alpha,
            beta=self.config.beta,
        ).to(self.device)
        self.model.logit_scale.data.fill_(self.config.logit_scale)
        self.entailment_loss = HierarchicalEntailmentLoss(
            aperture_scale=self.config.aperture_scale
        ).to(self.device)
        self.contrastive_loss = HierarchicalContrastiveLoss(temperature=0.1).to(self.device)
        self.angular_loss = HierarchicalAngularContrastiveLoss(lambda_centroid=0.1).to(self.device)
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

    def _setup_dataset_for_level(self, level_pair_index: int) -> NoCategorySubtreeDataset:
        level_pair = NO_CATEGORY_LEVEL_PAIRS[level_pair_index - 1]
        nodes_by_level = extract_no_category_nodes_from_store(
            self.vector_store,
            level_pair_index=level_pair_index,
        )
        num_iterations = (
            self.config.iterations_map[level_pair_index]
            if self.config.sequential_levels
            else self.config.num_iterations
        )
        return NoCategorySubtreeDataset(
            nodes_by_level=nodes_by_level,
            embedding_dim=self.config.embedding_dim,
            device=self.device,
            num_iterations=num_iterations,
            num_parents_per_batch=self.config.num_parents_per_batch,
            num_children_per_parent=self.config.num_children_per_parent,
            max_children_per_parent=self.config.max_children_per_parent,
            level_pair=level_pair,
            load_feats_by_level=False,
            use_level_embedding=self.config.use_level_embedding,
        )

    def _create_scheduler(self, total_iterations: int) -> None:
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=total_iterations,
            eta_min=1e-6,
        )

    def _process_batch_feats(self, batch):
        _, parent_feats_h = self.model(batch.parent_feats)
        _, child_feats_h = self.model(batch.child_feats)
        return parent_feats_h, child_feats_h

    def _compute_losses(self, batch, parent_feats_h, child_feats_h, curv):
        from model.hyperbolic_utils.hierarchical_dataset import SubtreeBatch

        projected_batch = SubtreeBatch(
            parent_level=batch.parent_level,
            child_level=batch.child_level,
            parent_feats=parent_feats_h,
            child_feats=child_feats_h,
            parent_child_mask=batch.parent_child_mask,
            parent_child_map=batch.parent_child_map,
            parent_ids=batch.parent_ids,
            child_ids=batch.child_ids,
            n_parent=batch.n_parent,
            n_child=batch.n_child,
        )

        losses = {"curvature": curv, "level_pair": (batch.parent_level, batch.child_level)}
        total_loss = None
        if self.config.entailment_weight > 0:
            entailment_out = self.entailment_loss(projected_batch, curv)
            losses["entailment_loss"] = entailment_out["loss"]
            losses["in_cone_ratio"] = entailment_out["in_cone_ratio"]
            total_loss = self.config.entailment_weight * entailment_out["loss"]
        if self.config.contrastive_weight > 0:
            contrastive_out = self.contrastive_loss(
                projected_batch,
                curv,
                logit_scale=self.model.logit_scale.item(),
            )
            losses["contrastive_loss"] = contrastive_out["loss"]
            losses["parent_accuracy"] = contrastive_out["parent_accuracy"]
            losses["child_accuracy"] = contrastive_out["child_accuracy"]
            part = self.config.contrastive_weight * contrastive_out["loss"]
            total_loss = part if total_loss is None else total_loss + part
        if self.config.angular_weight > 0:
            angular_out = self.angular_loss(projected_batch, curv, logit_scale=self.model.logit_scale)
            losses["angular_loss"] = angular_out["loss"]
            part = self.config.angular_weight * angular_out["loss"]
            total_loss = part if total_loss is None else total_loss + part
        if total_loss is None:
            raise ValueError("at least one loss must be enabled")
        losses["total_loss"] = total_loss
        return losses

    def train_step(self, batch):
        self.model.train()
        self.optimizer.zero_grad()
        curv = torch.nn.functional.softplus(self.model.c)
        parent_feats_h, child_feats_h = self._process_batch_feats(batch)
        losses = self._compute_losses(batch, parent_feats_h, child_feats_h, curv)
        losses["total_loss"].backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.scheduler.step()
        self.global_step += 1
        self.level_step += 1
        stats = {
            "global_step": self.global_step,
            "level_step": self.level_step,
            "level_idx": self.current_level_idx,
            "total_loss": losses["total_loss"].item(),
            "curvature": losses["curvature"].detach().item(),
            "lr": self.optimizer.param_groups[0]["lr"],
            "level_pair": str(losses["level_pair"]),
        }
        for key in (
            "entailment_loss",
            "contrastive_loss",
            "angular_loss",
            "in_cone_ratio",
            "parent_accuracy",
            "child_accuracy",
        ):
            if key in losses:
                value = losses[key]
                stats[key] = value.item() if hasattr(value, "item") else value
        self.loss_history.append(stats)
        return stats

    def train_level_pair(self, level_pair_index: int) -> None:
        self.current_level_idx = level_pair_index
        self.level_step = 0
        level_pair = NO_CATEGORY_LEVEL_PAIRS[level_pair_index - 1]
        dataset = self._setup_dataset_for_level(level_pair_index)
        dataloader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=True,
            num_workers=0,
            collate_fn=subtree_collate_fn,
        )
        num_iterations = (
            self.config.iterations_map[level_pair_index]
            if self.config.sequential_levels
            else self.config.num_iterations
        )
        self._create_scheduler(num_iterations)
        progress = tqdm(dataloader, total=num_iterations, desc=f"train {level_pair[0]}->{level_pair[1]}")
        for batch in progress:
            stats = self.train_step(batch)
            progress.set_postfix({"loss": f"{stats['total_loss']:.4f}", "c": f"{stats['curvature']:.4f}"})
            if self.level_step % self.config.save_interval == 0:
                self._save_checkpoint(level_info=f"level{level_pair_index}_step{self.level_step}")
        self._save_checkpoint(level_info=f"level{level_pair_index}_final", final_level=True)

    def train(self) -> None:
        if self.config.level_pair_index is not None:
            self.train_level_pair(self.config.level_pair_index)
        else:
            for level_idx in (1, 2):
                self.train_level_pair(level_idx)
        self._save_checkpoint(final=True)

    def _save_checkpoint(self, level_info: str = "", final: bool = False, final_level: bool = False) -> None:
        os.makedirs(self.config.output_dir, exist_ok=True)
        if final:
            filename = "hyperbolic_projector_final.pt"
        else:
            filename = f"hyperbolic_projector_{level_info}.pt"
        path = os.path.join(self.config.output_dir, filename)
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if hasattr(self, "scheduler") else None,
            "global_step": self.global_step,
            "current_level_idx": self.current_level_idx,
            "level_step": self.level_step,
            "config": self.config.__dict__,
            "curvature": torch.nn.functional.softplus(self.model.c).item(),
        }
        torch.save(checkpoint, path)
        with open(os.path.join(self.config.output_dir, "loss_history.json"), "w", encoding="utf-8") as handle:
            json.dump(self.loss_history, handle, indent=2)


def parse_args() -> NoCategoryTrainConfig:
    parser = argparse.ArgumentParser(description="train no-category hyperbolic projector")
    parser.add_argument("--vector_store_path", type=str, default="./data/vector_store")
    parser.add_argument("--embedding_dim", type=int, default=384)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--output_dir", type=str, default="./checkpoints_no_category")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--level_pair_index", type=int, default=None)
    args = parser.parse_args()
    return NoCategoryTrainConfig(
        vector_store_path=args.vector_store_path,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        output_dir=args.output_dir,
        device=args.device,
        level_pair_index=args.level_pair_index,
    )


def main() -> None:
    config = parse_args()
    trainer = NoCategoryHyperbolicTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
