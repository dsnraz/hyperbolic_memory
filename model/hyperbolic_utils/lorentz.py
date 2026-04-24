# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
洛伦兹模型双曲几何常用操作的实现。

该模型将 `d` 维双曲空间表示为 `(d+1)` 维欧氏空间中双叶双曲面的上半部分。

函数命名约定:
    - `_vectors` 后缀: 输入为 2D 张量 [Batch, Hidden_Dim]
    - `_sequences` 后缀: 输入为 3D 张量 [Batch, Sequence_Length, Hidden_Dim]

此处实现的所有函数仅输入/输出空间分量（Space Components），
并在内部根据双曲面约束计算时间分量（Time Component）：
    `x_time = torch.sqrt(1 / curv + torch.norm(x_space) ** 2)`
"""
from __future__ import annotations

import math
import torch
from torch import Tensor


# ============================================================================
# 内积计算 (Lorentz Inner Product)
# ============================================================================

def pairwise_inner_vectors(x: Tensor, y: Tensor, curv: float | Tensor = 1.0) -> Tensor:
    """
    计算 2D 向量集合之间的成对洛伦兹内积矩阵。
    
    输入格式: [Batch, Hidden_Dim]
    输出格式: [Batch_x, Batch_y] 的内积矩阵
    
    公式: <x, y>_L = x_space · y_space - x_time * y_time
    
    参数:
        x: 形状为 (B_x, D) 的张量，表示第一组向量。
        y: 形状为 (B_y, D) 的张量，表示第二组向量。
        curv: 曲率参数，默认为 1.0。
    
    返回:
        形状为 (B_x, B_y) 的内积矩阵。
    """
    x_time = torch.sqrt(1 / curv + torch.sum(x**2, dim=-1, keepdim=True))
    y_time = torch.sqrt(1 / curv + torch.sum(y**2, dim=-1, keepdim=True))
    xyl = x @ y.T - x_time @ y_time.T
    return xyl


def pairwise_inner_sequences(x: Tensor, y: Tensor, curv: float = 1.0) -> Tensor:
    """
    计算 3D 序列数据之间的成对洛伦兹内积矩阵。
    
    输入格式: [Batch, Sequence, Hidden_Dim] (squeeze 到 2D 后计算)
    输出格式: [Batch_x, Batch_y] 的内积矩阵
    
    参数:
        x: 形状为 (B, S, D) 或 (B, D) 的张量。
        y: 形状为 (B, S, D) 或 (B, D) 的张量。
        curv: 曲率参数，默认为 1.0。
    
    返回:
        形状为 (B_x, B_y) 的内积矩阵。
    """
    if x.dim() == 3: x = x.squeeze(1)
    if y.dim() == 3: y = y.squeeze(1)
    
    x_time = torch.sqrt(1 / curv + torch.sum(x**2, dim=-1, keepdim=True))
    y_time = torch.sqrt(1 / curv + torch.sum(y**2, dim=-1, keepdim=True))
    
    spatial_inner = x @ y.transpose(-2, -1)
    time_inner = x_time @ y_time.transpose(-2, -1)
    
    xyl = spatial_inner - time_inner
    return xyl


def elementwise_inner_vectors(x: Tensor, y: Tensor, curv: float | Tensor = 1.0) -> Tensor:
    """
    计算 2D 向量集合之间的逐元素洛伦兹内积。
    
    输入格式: [Batch, Hidden_Dim]
    输出格式: [Batch] (最后一维被约减)
    
    对应位置的向量 x[i] 与 y[i] 计算内积。
    
    参数:
        x: 形状为 (B, D) 的张量。
        y: 形状为 (B, D) 的张量。
        curv: 曲率参数，默认为 1.0。
    
    返回:
        形状为 (B,) 的内积向量。
    """
    x_time = torch.sqrt(1 / curv + torch.sum(x**2, dim=-1))
    y_time = torch.sqrt(1 / curv + torch.sum(y**2, dim=-1))
    xyl = torch.sum(x * y, dim=-1) - x_time * y_time
    return xyl


def elementwise_inner_sequences(x: Tensor, y: Tensor, curv: float | Tensor = 1.0) -> Tensor:
    """
    计算 3D 序列数据之间的逐元素洛伦兹内积。
    
    输入格式: [Batch, Sequence, Hidden_Dim]
    输出格式: [Batch, Sequence] (最后一维被约减)
    
    对应位置的向量 x[b, s] 与 y[b, s] 计算内积。
    
    参数:
        x: 形状为 (B, S, D) 的张量。
        y: 形状为 (B, S, D) 的张量。
        curv: 曲率参数，默认为 1.0。
    
    返回:
        形状为 (B, S) 的内积矩阵。
    """
    x_time = torch.sqrt(1 / curv + torch.sum(x**2, dim=-1))
    y_time = torch.sqrt(1 / curv + torch.sum(y**2, dim=-1))
    xyl = torch.sum(x * y, dim=-1) - x_time * y_time
    return xyl


# ============================================================================
# 测地线距离计算 (Geodesic Distance)
# ============================================================================

def pairwise_dist_vectors(
    x: Tensor, y: Tensor, curv: float | Tensor = 1.0, eps: float = 1e-5
) -> Tensor:
    """
    计算 2D 向量集合之间的成对测地线距离矩阵。
    
    输入格式: [Batch, Hidden_Dim]
    输出格式: [Batch_x, Batch_y] 的距离矩阵
    
    公式: d(x, y) = acosh(-curv * <x, y>_L) / sqrt(curv)
    
    参数:
        x: 形状为 (B_x, D) 的张量。
        y: 形状为 (B_y, D) 的张量。
        curv: 曲率参数，默认为 1.0。
        eps: 数值稳定性参数，防止 acosh 输入小于 1。
    
    返回:
        形状为 (B_x, B_y) 的距离矩阵。
    """
    c_xyl = -curv * pairwise_inner_vectors(x, y, curv)
    _distance = torch.acosh(torch.clamp(c_xyl, min=1 + eps))
    return _distance / curv**0.5


def pairwise_dist_sequences(
    x: Tensor, y: Tensor, curv: float | Tensor = 1.0, eps: float = 1e-5
) -> Tensor:
    """
    计算 3D 序列数据之间的成对测地线距离矩阵。
    
    输入格式: [Batch, Sequence, Hidden_Dim] (squeeze 到 2D 后计算)
    输出格式: [Batch_x, Batch_y] 的距离矩阵
    
    参数:
        x: 形状为 (B, S, D) 或 (B, D) 的张量。
        y: 形状为 (B, S, D) 或 (B, D) 的张量。
        curv: 曲率参数，默认为 1.0。
        eps: 数值稳定性参数。
    
    返回:
        形状为 (B_x, B_y) 的距离矩阵。
    """
    inner_val = pairwise_inner_sequences(x, y, curv)
    c_xyl = -curv * inner_val
    _distance = torch.acosh(torch.clamp(c_xyl, min=1 + eps))
    return _distance / (curv**0.5)


def elementwise_dist_vectors(
    x: Tensor, y: Tensor, curv: float | Tensor = 1.0, eps: float = 1e-5
) -> Tensor:
    """
    计算 2D 向量集合之间的逐元素测地线距离。
    
    输入格式: [Batch, Hidden_Dim]
    输出格式: [Batch]
    
    参数:
        x: 形状为 (B, D) 的张量。
        y: 形状为 (B, D) 的张量。
        curv: 曲率参数，默认为 1.0。
        eps: 数值稳定性参数。
    
    返回:
        形状为 (B,) 的距离向量。
    """
    c_xyl = -curv * elementwise_inner_vectors(x, y, curv)
    _distance = torch.acosh(torch.clamp(c_xyl, min=1 + eps))
    return _distance / curv**0.5


def elementwise_dist_sequences(
    x: Tensor, y: Tensor, curv: float | Tensor = 1.0, eps: float = 1e-5
) -> Tensor:
    """
    计算 3D 序列数据之间的逐元素测地线距离。
    
    输入格式: [Batch, Sequence, Hidden_Dim]
    输出格式: [Batch, Sequence]
    
    参数:
        x: 形状为 (B, S, D) 的张量。
        y: 形状为 (B, S, D) 的张量。
        curv: 曲率参数，默认为 1.0。
        eps: 数值稳定性参数。
    
    返回:
        形状为 (B, S) 的距离矩阵。
    """
    c_xyl = -curv * elementwise_inner_sequences(x, y, curv)
    _distance = torch.acosh(torch.clamp(c_xyl, min=1 + eps))
    return _distance / curv**0.5


# ============================================================================
# 双曲余弦定理 (Hyperbolic law of cosines, constant curvature)
# ============================================================================


def hyperbolic_law_of_cosines_cos(
    a: Tensor,
    b: Tensor,
    c: Tensor,
) -> Tensor:
    """
    双曲三角形：三边**测地**长为 ``a, b, c``（与 ``pairwise_dist_vectors`` 等使用的距离
    单位一致，即该模型下的弧长）。设所求**内角**的顶点为 b 与 c 的**公共**端点，**对边**为
    ``a``，则

    .. math:: \\cos \\theta = \\frac{\\cosh a - \\cosh b\\,\\cosh c}{\\sinh b\\,\\sinh c} \\,.

    ``a, b, c`` 需同形状、逐元素一一对应。当 ``b`` 或 ``c`` 的 ``|sinh|`` 过小时分母
    会数值失稳，请改用 ``hyperbolic_law_of_cosines_angle`` 做退化与 NaN 处理。
    """
    if a.shape != b.shape or a.shape != c.shape:
        raise ValueError("a, b, c 必须同形状。")
    return (torch.cosh(a) - torch.cosh(b) * torch.cosh(c)) / (torch.sinh(b) * torch.sinh(c))


def hyperbolic_law_of_cosines_angle(
    a: Tensor,
    b: Tensor,
    c: Tensor,
    eps: float = 1e-12,
) -> Tensor:
    """
    与 ``hyperbolic_law_of_cosines_cos`` 同几何约定，返回内角 :math:`\\theta` 的**弧度**。

    当 :math:`|\\sinh b|<\\texttt{eps}` 或 :math:`|\\sinh c|<\\texttt{eps}`、或任一边长
    非有限、或 :math:`\\cos\\theta` 非有限时，该位置为 ``nan``。否则在 ``arccos`` 前
    对 ``cos`` 做 ``[-1+1e-6, 1-1e-6]`` 钳制，避免仅由舍入引起的越界。
    """
    if a.shape != b.shape or a.shape != c.shape:
        raise ValueError("a, b, c 必须同形状。")
    shb, shc = torch.sinh(b), torch.sinh(c)
    good = (shb.abs() >= eps) & (shc.abs() >= eps) & torch.isfinite(a) & torch.isfinite(
        b
    ) & torch.isfinite(c)
    cos_o = (torch.cosh(a) - torch.cosh(b) * torch.cosh(c)) / (shb * shc)
    good = good & torch.isfinite(cos_o)
    cos_o = torch.clamp(cos_o, -1.0 + 1e-6, 1.0 - 1e-6)
    ang = torch.acos(cos_o)
    return torch.where(good, ang, torch.full_like(ang, float("nan")))


# ============================================================================
# 指数映射与对数映射 (Exponential & Logarithmic Maps)
# ============================================================================

def exp_map0_vectors(x: Tensor, curv: float | Tensor = 1.0, eps: float = 1e-5) -> Tensor:
    """
    指数映射 (2D 版本): 将欧氏切向量投影到双曲面上。
    
    输入格式: [Batch, Hidden_Dim]
    输出格式: [Batch, Hidden_Dim]
    
    这些向量被解释为双曲面顶点切空间中的速度向量。
    
    参数:
        x: 形状为 (B, D) 的张量，表示切向量。
        curv: 曲率参数，默认为 1.0。
        eps: 避免除零的小浮点数。
    
    返回:
        形状为 (B, D) 的张量，映射到双曲面上的空间分量。
    """
    if torch.isnan(x).any() or torch.isinf(x).any():
        print("NaN or Inf detected in input to exp_map0")

    x_norm = torch.norm(x, dim=-1, keepdim=True)
    rc_xnorm = curv**0.5 * x_norm

    sinh_input = torch.clamp(rc_xnorm, min=eps, max=math.asinh(2**15))
    rc_xnorm_clamped = torch.clamp(rc_xnorm, min=eps)

    _output = torch.sinh(sinh_input) * x / rc_xnorm_clamped

    if torch.isnan(_output).any() or torch.isinf(_output).any():
        print("NaN or Inf detected in output of exp_map0")

    return _output


def exp_map0_sequences(x: Tensor, curv: float | Tensor = 1.0, eps: float = 1e-5) -> Tensor:
    """
    指数映射 (3D 版本): 将欧氏切向量投影到双曲面上。
    
    输入格式: [Batch, Sequence, Hidden_Dim]
    输出格式: [Batch, Sequence, Hidden_Dim]
    
    这些向量被解释为双曲面顶点切空间中的速度向量。
    
    参数:
        x: 形状为 (B, S, D) 的张量，表示切向量。
        curv: 曲率参数，默认为 1.0。
        eps: 避免除零的小浮点数。
    
    返回:
        形状为 (B, S, D) 的张量，映射到双曲面上的空间分量。
    """
    if torch.isnan(x).any() or torch.isinf(x).any():
        print("exp_map0 输入中检测到 NaN 或 Inf")
        print(x)

    x_norm = torch.norm(x, dim=-1, keepdim=True)
    rc_xnorm = curv**0.5 * x_norm

    sinh_input = torch.clamp(rc_xnorm, min=eps, max=math.asinh(2**15))
    rc_xnorm_clamped = torch.clamp(rc_xnorm, min=eps)

    _output = torch.sinh(sinh_input) * x / rc_xnorm_clamped

    if torch.isnan(_output).any() or torch.isinf(_output).any():
        print("exp_map0 输出中检测到 NaN 或 Inf")

    return _output


def log_map0_vectors(x: Tensor, curv: float | Tensor = 1.0, eps: float = 1e-5) -> Tensor:
    """
    对数映射 (2D 版本): 将双曲面上的点映射回欧氏切空间。
    
    输入格式: [Batch, Hidden_Dim]
    输出格式: [Batch, Hidden_Dim]
    
    这是指数映射的逆过程。
    
    参数:
        x: 形状为 (B, D) 的张量，表示双曲面上的点（空间分量）。
        curv: 曲率参数，默认为 1.0。
        eps: 避免除零的小浮点数。
    
    返回:
        形状为 (B, D) 的张量，表示切空间中的欧氏向量。
    """
    rc_x_time = torch.sqrt(1 + curv * torch.sum(x**2, dim=-1, keepdim=True))
    _distance0 = torch.acosh(torch.clamp(rc_x_time, min=1 + eps))

    rc_xnorm = curv**0.5 * torch.norm(x, dim=-1, keepdim=True)
    _output = _distance0 * x / torch.clamp(rc_xnorm, min=eps)
    return _output


def log_map0_sequences(x: Tensor, curv: float | Tensor = 1.0, eps: float = 1e-5) -> Tensor:
    """
    对数映射 (3D 版本): 将双曲面上的点映射回欧氏切空间。
    
    输入格式: [Batch, Sequence, Hidden_Dim]
    输出格式: [Batch, Sequence, Hidden_Dim]
    
    这是指数映射的逆过程。
    
    参数:
        x: 形状为 (B, S, D) 的张量，表示双曲面上的点（空间分量）。
        curv: 曲率参数，默认为 1.0。
        eps: 避免除零的小浮点数。
    
    返回:
        形状为 (B, S, D) 的张量，表示切空间中的欧氏向量。
    """
    rc_x_time = torch.sqrt(1 + curv * torch.sum(x**2, dim=-1, keepdim=True))
    _distance0 = torch.acosh(torch.clamp(rc_x_time, min=1 + eps))

    rc_xnorm = curv**0.5 * torch.norm(x, dim=-1, keepdim=True)
    _output = _distance0 * x / torch.clamp(rc_xnorm, min=eps)
    return _output


# ============================================================================
# 蕴涵锥与角度计算 (Entailment Cone & Angles)
# ============================================================================

def half_aperture_vectors(
    x: Tensor, curv: float | Tensor = 1.0, min_radius: float = 0.1, eps: float = 1e-5
) -> Tensor:
    """
    计算 2D 向量形成的蕴涵锥的半孔径角。
    
    输入格式: [Batch, Hidden_Dim]
    输出格式: [Batch]
    
    参数:
        x: 形状为 (B, D) 的张量。
        curv: 曲率参数，默认为 1.0。
        min_radius: 双曲面顶点周围的小邻域半径，在此范围内孔径未定义。
        eps: 数值稳定性参数。
    
    返回:
        形状为 (B,) 的张量，半孔径角，值域为 (0, pi/2)。
    """
    asin_input = 2 * min_radius / (torch.norm(x, dim=-1) * curv**0.5 + eps)
    _half_aperture = torch.asin(torch.clamp(asin_input, min=-1 + eps, max=1 - eps))
    return _half_aperture


def half_aperture_sequences(
    x: Tensor, curv: float | Tensor = 1.0, min_radius: float = 0.1, eps: float = 1e-5
) -> Tensor:
    """
    计算 3D 序列数据形成的蕴涵锥的半孔径角。
    
    输入格式: [Batch, Sequence, Hidden_Dim]
    输出格式: [Batch, Sequence]
    
    参数:
        x: 形状为 (B, S, D) 的张量。
        curv: 曲率参数，默认为 1.0。
        min_radius: 双曲面顶点周围的小邻域半径。
        eps: 数值稳定性参数。
    
    返回:
        形状为 (B, S) 的张量，半孔径角，值域为 (0, pi/2)。
    """
    x_norm = torch.norm(x, dim=-1)
    asin_input = 2 * min_radius / (x_norm * curv**0.5 + eps)
    _half_aperture = torch.asin(torch.clamp(asin_input, min=-1 + eps, max=1 - eps))
    return _half_aperture


def cone_vertex_exterior_angle_vectors(
    x: Tensor, y: Tensor, curv: float | Tensor = 1.0, eps: float = 1e-5
) -> Tensor:
    """
    计算 2D 向量在双曲三角形 Oxy 中顶点 x 处的外角。
    
    输入格式: [Batch, Hidden_Dim]
    输出格式: [Batch]
    
    O 是双曲面的原点，此函数逐元素计算 x[i] 与 y[i] 之间的外角。
    
    参数:
        x: 形状为 (B, D) 的张量，锥体顶点向量。
        y: 形状为 (B, D) 的张量，目标向量。
        curv: 曲率参数，默认为 1.0。
        eps: 数值稳定性参数。
    
    返回:
        形状为 (B,) 的张量，外角，值域为 (0, pi)。
    """
    x_time = torch.sqrt(1 / curv + torch.sum(x**2, dim=-1))
    y_time = torch.sqrt(1 / curv + torch.sum(y**2, dim=-1))

    c_xyl = curv * (torch.sum(x * y, dim=-1) - x_time * y_time)

    acos_numer = y_time + c_xyl * x_time
    acos_denom = torch.sqrt(torch.clamp(c_xyl**2 - 1, min=eps))

    acos_input = acos_numer / (torch.norm(x, dim=-1) * acos_denom + eps)
    _angle = torch.acos(torch.clamp(acos_input, min=-1 + eps, max=1 - eps))

    return _angle


def cone_vertex_exterior_angle_sequences(
    x: Tensor, y: Tensor, curv: float | Tensor = 1.0, eps: float = 1e-5
) -> Tensor:
    """
    计算 3D 序列数据在双曲三角形 Oxy 中顶点 x 处的外角。
    
    输入格式: [Batch, Sequence, Hidden_Dim]
    输出格式: [Batch, Sequence]
    
    O 是双曲面的原点，此函数逐元素计算 x[b, s] 与 y[b, s] 之间的外角。
    
    参数:
        x: 形状为 (B, S, D) 的张量，锥体顶点向量。
        y: 形状为 (B, S, D) 的张量，目标向量。
        curv: 曲率参数，默认为 1.0。
        eps: 数值稳定性参数。
    
    返回:
        形状为 (B, S) 的张量，外角，值域为 (0, pi)。
    """
    x_time = torch.sqrt(1 / curv + torch.sum(x**2, dim=-1))
    y_time = torch.sqrt(1 / curv + torch.sum(y**2, dim=-1))

    c_xyl = curv * (torch.sum(x * y, dim=-1) - x_time * y_time)

    acos_numer = y_time + c_xyl * x_time
    acos_denom = torch.sqrt(torch.clamp(c_xyl**2 - 1, min=eps))

    x_norm = torch.norm(x, dim=-1)
    acos_input = acos_numer / (x_norm * acos_denom + eps)
    _angle = torch.acos(torch.clamp(acos_input, min=-1 + eps, max=1 - eps))

    return _angle


def pairwise_exterior_angle_vectors(
    x: Tensor, y: Tensor, curv: float | Tensor = 1.0, eps: float = 1e-5
) -> Tensor:
    """
    计算两组 2D 向量之间的双曲锥体外角矩阵。
    
    输入格式: [Batch, Hidden_Dim]
    输出格式: [Batch_x, Batch_y] 的角度矩阵
    
    注意不对称性：x 通常代表 Text (锥体顶点)，y 代表 Image。
    
    参数:
        x: 形状为 (B_x, D) 的张量，锥体顶点向量集合。
        y: 形状为 (B_y, D) 的张量，目标向量集合。
        curv: 曲率参数，默认为 1.0。
        eps: 数值稳定性参数。
    
    返回:
        形状为 (B_x, B_y) 的角度矩阵。angle[i, j] 表示 x[i] 到 y[j] 的外角。
    """
    if x.dim() == 3: x = x.squeeze(1)
    if y.dim() == 3: y = y.squeeze(1)

    # 1. 计算时间分量
    x_time = torch.sqrt(1 / curv + torch.sum(x**2, dim=-1, keepdim=True))  # (B, 1)
    y_time = torch.sqrt(1 / curv + torch.sum(y**2, dim=-1, keepdim=True))  # (B, 1)

    # 2. 计算洛伦兹内积矩阵 (B, B)
    inner_val = pairwise_inner_vectors(x, y, curv)
    c_xyl = curv * inner_val

    # 3. 准备分子: y_time + x_time * c * <x, y>_H
    y_time_t = y_time.transpose(0, 1)  # 转置为 (1, B)
    acos_numer = y_time_t + c_xyl * x_time  # 广播: (1, B) + (B, B) * (B, 1) -> (B, B)

    # 4. 准备分母: ||x_space|| * sqrt((c * <x, y>_H)^2 - 1)
    x_norm = torch.norm(x, dim=-1, keepdim=True).clamp(min=eps)  # (B, 1)
    acos_denom = torch.sqrt(torch.clamp(c_xyl**2 - 1.0, min=eps))  # (B, B)

    # 5. 计算最终角度
    acos_input = acos_numer / (x_norm * acos_denom + eps)
    _angle = torch.acos(torch.clamp(acos_input, min=-1 + eps, max=1 - eps))

    return _angle


# ============================================================================
# 中点计算 (Midpoint)
# ============================================================================

def lorentz_midpoint_sequences(
    x: Tensor, curv: float | Tensor = 1.0, eps: float = 1e-5, keep_dim: bool = False
) -> Tensor:
    """
    计算 3D 序列数据在双曲面上的 Fréchet 中点。
    
    输入格式: [Batch, Sequence, Hidden_Dim]
    输出格式: [Batch, Hidden_Dim] (keep_dim=False) 或 [Batch, 1, Hidden_Dim] (keep_dim=True)
    
    沿序列维度聚合，计算每个 batch 的中点。
    
    参数:
        x: 形状为 (B, S, D) 的张量。
        curv: 曲率参数，默认为 1.0。
        eps: 数值稳定性参数。
        keep_dim: 是否保留序列维度。
    
    返回:
        形状为 (B, D) 或 (B, 1, D) 的张量，双曲面上的中点。
    """
    # 计算空间分量的和
    x_space_sum = torch.sum(x**2, dim=-1, keepdim=True)

    # 计算时间分量
    x_time = torch.sqrt(1.0 / curv + x_space_sum)

    # 拼接时间分量和空间分量
    x_total = torch.cat([x_time, x], dim=-1)

    # 按序列维度求和
    sum_x = torch.sum(x_total, dim=1, keepdim=True)
    sum_x_space = sum_x[:, :, 1:]
    sum_x_time = sum_x[:, :, 0:1]

    # 计算内积用于归一化
    inner_product = -(sum_x_time**2) + torch.sum(sum_x_space**2, dim=-1, keepdim=True)

    # 计算归一化系数
    alpha = 1.0 / torch.sqrt(curv * torch.abs(inner_product) + eps)

    # 归一化得到中点的空间分量
    midpoint_space = alpha * sum_x_space

    if not keep_dim:
        midpoint_space = midpoint_space.squeeze(1)

    return midpoint_space


# ============================================================================
# 兼容性别名 (向后兼容旧命名)
# ============================================================================

# 以下别名保持向后兼容，不建议在新代码中使用

pairwise_inner1 = pairwise_inner_vectors
pairwise_dist1 = pairwise_dist_vectors
elementwise_inner1 = elementwise_inner_vectors
elementwise_dist1 = elementwise_dist_vectors
exp_map01 = exp_map0_vectors
log_map01 = log_map0_vectors
half_aperture1 = half_aperture_vectors
oxy_angle1 = cone_vertex_exterior_angle_vectors

pairwise_inner = pairwise_inner_sequences
pairwise_dist = pairwise_dist_sequences
elementwise_inner = elementwise_inner_sequences
elementwise_dist = elementwise_dist_sequences
exp_map0 = exp_map0_sequences
log_map0 = log_map0_sequences
half_aperture = half_aperture_sequences
oxy_angle = cone_vertex_exterior_angle_sequences


if __name__ == "__main__":
    # 测试代码
    x = torch.randn(2, 1, 4)
    y = torch.randn(2, 1, 4)
    print("输入 x:", x)

    # 测试新命名函数
    dist = pairwise_dist_sequences(x, y)
    print("成对距离 (3D):", dist)
    print("距离形状:", dist.shape)

    # 测试兼容性别名
    dist_compat = pairwise_dist(x, y)
    print("兼容性别名结果:", dist_compat)