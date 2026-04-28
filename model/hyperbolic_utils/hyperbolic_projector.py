import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .lorentz import exp_map0, log_map0
from .hierarchical_loss import HierarchicalContrastiveLoss, HierarchicalAngularContrastiveLoss

class Hyperbolic_projector(nn.Module):
    def __init__(self, input_dim, hidden_dim, curvature=0.1, alpha=0.1, beta=0.8):
        super(Hyperbolic_projector, self).__init__()

        # ====================================================================
        # v1_softplus_fix: 反解 softplus，让 `curvature` 的字面值真正是有效曲率。
        # 目标：softplus(c_raw) == curvature  =>  c_raw = log(e^curvature - 1)
        # math.expm1(x) = e^x - 1，数值稳定。
        # ====================================================================
        c_raw_init = math.log(math.expm1(float(curvature)))
        self.c = nn.Parameter(torch.tensor([c_raw_init], dtype=torch.float32))
        self.alpha = alpha
        self.beta = beta

        self.phi = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.depth_predictors = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())
        self.fusion_layer = nn.Linear(input_dim + hidden_dim, input_dim)
        self.gate_weight = nn.Linear(input_dim, input_dim)

        self.logit_scale = nn.Parameter(torch.tensor([2.6592]))
        self.loss_weight_contrastive = 0.7
        self.loss_weight_entailment = 0.3

    def forward(self, z_E):
        curr_c = torch.nn.functional.softplus(self.c)
        R = 1.0 / torch.sqrt(curr_c + 1e-6)
        u_v = self.phi(z_E)
        d_v = self.depth_predictors(u_v) # Sigmoid

        # 3. 计算 fusion 并【立即截断】
        fusion_input = torch.cat([z_E, u_v], dim=-1)
        z_tilde_out = self.fusion_layer(fusion_input)
        z_tilde_E = torch.relu(z_tilde_out)

        # 4. 计算 gate 并【立即截断】
        gate_out = self.gate_weight(z_tilde_E)
        m_v = torch.sigmoid(gate_out)

        # 5. 组合
        z_star_v = m_v * z_E + (1 - m_v) * z_tilde_E

        with torch.cuda.amp.autocast(enabled=False):
            # 1. 转 FP32
            z_star_v_32 = z_star_v.to(torch.float32)
            d_v_32 = d_v.to(torch.float32)

            # 2. 计算模长与缩放
            target_norm = (self.alpha + (self.beta-self.alpha) * d_v_32) * R
            z_star_norm = torch.norm(z_star_v_32, p=2, dim=-1, keepdim=True).clamp(min=1e-5)
            z_hat_E = (target_norm / z_star_norm) * z_star_v_32

            # 3. 模长上限截断 (Norm Capping)
            # 防止后期发散
            max_safe_norm = 15.0
            current_norm = torch.norm(z_hat_E, p=2, dim=-1, keepdim=True)
            scale_factor = torch.clamp(max_safe_norm / (current_norm + 1e-5), max=1.0)
            z_hat_E = z_hat_E * scale_factor
            # 4. 双曲映射
            z_H = exp_map0(z_hat_E, curv=curr_c)

            # 6. 逆映射
            z_E_last_32 = log_map0(z_H, curv=curr_c)
            output = z_E_last_32.to(z_E.dtype)

        return output, z_H
