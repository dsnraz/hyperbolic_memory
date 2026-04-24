import torch
from torch import nn
import torch.nn.functional as F
from . import lorentz as L

class LorentzianCLIPContrastive(nn.Module):
    def __init__(self, temperature=1.):
        super().__init__()
        self.temperature = temperature
        self.labels = None
        self.last_local_batch_size = None

    def forward(self, image_feats, text_feats, curv, unique_logit_scale=None, image_logit_scale=None, text_logit_scale=None, validation=False):
        # 1. 确定缩放因子
        if unique_logit_scale is not None:
            image_logit_scale = unique_logit_scale
            text_logit_scale = unique_logit_scale
        elif image_logit_scale is None or text_logit_scale is None:
            raise ValueError('At least (unique_logit_scale) or (image_logit_scale and text_logit_scale) must be provided')

        local_batch_size = text_feats.shape[0]
        device = image_feats.device

        # 2. 标签管理（增加 device 校验，防止多卡出错）
        if self.labels is None or local_batch_size != self.last_local_batch_size or self.labels.device != device:
            self.labels = torch.arange(local_batch_size, device=device, dtype=torch.long)
            self.last_local_batch_size = local_batch_size

        # 3. 计算双曲距离矩阵 (B, B)
        # 注意：这里调用你修改后的 L.pairwise_dist
        dist_matrix = L.pairwise_dist(image_feats, text_feats, curv)
        # if self.training:
        #     print(f"DEBUG: dist_matrix mean={dist_matrix.mean().item():.4f}, max={dist_matrix.max().item():.4f}")
        
        # 转换成 logits (取负号，距离越小相似度越高)
        image_logits = -dist_matrix
        text_logits = -dist_matrix.t() 

        # 4. 缩放并计算交叉熵
        loss_i2t = nn.functional.cross_entropy(image_logit_scale * image_logits, self.labels)
        loss_t2i = nn.functional.cross_entropy(text_logit_scale * text_logits, self.labels)
        
        contrastive_loss = 0.5 * (loss_i2t + loss_t2i)
        
        return {
            'loss': contrastive_loss, 
            'values': {
                'image': image_logits, 
                'text': text_logits,
                'text_logit_scale': text_logit_scale,
                'image_logit_scale': image_logit_scale,
                'curv': curv
            },
        }

def entailmentLoss_A(text_feats, image_feats, _curv, aperture_scale=1.0):
    # Hyperbolic entailment loss: text should entail matching image.
    _angle = L.oxy_angle1(text_feats, image_feats, _curv)
    _aperture = L.half_aperture1(text_feats, _curv)
    entailment_loss = torch.clamp(_angle - (aperture_scale * _aperture), min=0).mean()

    return entailment_loss



class AcceptModalityGapLoss(nn.Module):
    def __init__(self, lambda_c=0.1, p=0.7, q=0.3):
        """
        根据论文参数初始化：
        lambda_c: 质心正则化权重 (trade-off hyperparameter)
        p: 图像质心的目标半径 (Target distance for images)
        q: 文本质心的目标半径 (Target distance for text)
        """
        super().__init__()
        self.lambda_c = lambda_c
        self.p = p
        self.q = q
        self.labels = None
        self.last_local_batch_size = None
        self.eps = 1e-6

    def forward(self, image_feats, text_feats, curv, unique_logit_scale=None, 
                image_logit_scale=None, text_logit_scale=None, validation=False):
        
        R = 1.0 / torch.sqrt(curv + 1e-6) # 避免除零
        R = R.reshape([])
        
        # 1. 缩放因子处理（对齐 LorentzianCLIPContrastive 逻辑）
        if image_logit_scale is None and text_logit_scale is None:
            if unique_logit_scale is None:
                raise ValueError('At least unique_logit_scale or (image_logit_scale and text_logit_scale) must be provided')
            image_logit_scale = unique_logit_scale
            text_logit_scale = unique_logit_scale
        
        local_batch_size = text_feats.shape[0]
        device = image_feats.device

        # 2. 标签缓存管理（增加 device 校验，防止多卡出错）
        if self.labels is None or local_batch_size != self.last_local_batch_size or self.labels.device != device:
            self.labels = torch.arange(local_batch_size, device=device, dtype=torch.long)
            self.last_local_batch_size = local_batch_size

        # --- 3. 计算角度对比损失 L_angle (论文 Eq. 11) ---
        alpha = L.pairwise_exterior_angle(text_feats, image_feats, curv, self.eps)
        beta = torch.pi - alpha # 补角

        # 构造 Logits 矩阵（双向对比损失）
        logits_alpha_i2t = -alpha.T
        logits_alpha_t2i = -alpha
        
        # 论文提到显式最大化 beta (即 L(beta))
        logits_beta_i2t = beta.T
        logits_beta_t2i = beta

        # 按照 Eq. 11 计算四个交叉熵损失的均值
        loss_angle = 0.25 * (
            F.cross_entropy(image_logit_scale * logits_alpha_i2t, self.labels) +
            F.cross_entropy(text_logit_scale * logits_alpha_t2i, self.labels) +
            F.cross_entropy(image_logit_scale * logits_beta_i2t, self.labels) +
            F.cross_entropy(text_logit_scale * logits_beta_t2i, self.labels)
        )

        # --- 4. 计算质心正则化 L_centroid (论文 Eq. 12) ---
        img_centroid = L.lorentz_midpoint(image_feats.unsqueeze(0), curv, keep_dim=False) # 结果 (D,)
        txt_centroid = L.lorentz_midpoint(text_feats.unsqueeze(0), curv, keep_dim=False) # 结果 (D,)

        # 计算质心到原点的测地线距离
        origin = torch.zeros_like(img_centroid)
        dist_img = L.pairwise_dist(img_centroid.unsqueeze(0), origin.unsqueeze(0), curv).squeeze() # 结果标量
        dist_txt = L.pairwise_dist(txt_centroid.unsqueeze(0), origin.unsqueeze(0), curv).squeeze() # 结果标量

        # 强制质心分别接近 p (图像) 和 q (文本)
        loss_centroid = torch.abs(dist_img - self.p*R) + torch.abs(dist_txt - self.q*R)

        # --- 5. 汇总总损失 (论文 Eq. 13) ---
        total_loss = loss_angle + self.lambda_c * loss_centroid

        return {
            'total_loss': total_loss,
            'angle_loss': loss_angle,
            'centroid_loss': loss_centroid,
            'values': {
                'image': -alpha.T, 
                'text': -alpha,
                'text_logit_scale': text_logit_scale,
                'image_logit_scale': image_logit_scale,
                'curv': curv,
                'l_angle': loss_angle.detach(),
                'l_centroid': loss_centroid.detach()
            },
        }