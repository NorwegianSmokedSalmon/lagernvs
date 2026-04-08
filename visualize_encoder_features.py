# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
VGGT Encoder 特征可视化脚本。

本脚本提取 LagerNVS 模型中 VGGT Encoder（即 Reconstructor 模块）在解码（Renderer）之前
输出的中间层特征，并通过多种方式进行可视化：
  1. VGGT Aggregator 原始输出特征（最后一层，dim=2048）
  2. 经过 geo_feature_connector 投影后的特征（dim=768），即送入 Decoder 前的最终特征
  3. 使用 PCA 降维后的伪彩色图
  4. 各通道的均值热力图

输入格式与 minimal_inference.py 完全一致。

Usage:
  python visualize_encoder_features.py --images path/to/img1.png path/to/img2.png
  python visualize_encoder_features.py --images images/input_000000.png images/input_000001.png --output_dir feature_vis
"""

import argparse
import math
import os
import sys

# Pre-parse --gpu to set CUDA_VISIBLE_DEVICES before PyTorch initializes CUDA
if "--gpu" in sys.argv:
    gpu_idx = sys.argv.index("--gpu")
    if gpu_idx + 1 < len(sys.argv):
        os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[gpu_idx + 1]

import einops
import matplotlib
matplotlib.use("Agg")  # 使用非交互式后端，无需 GUI
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from models.encoder_decoder import EncDec_VitB8
from vggt.utils.load_fn import load_and_preprocess_images


# =============================================================================
# 核心：带钩子的 Reconstructor 前向传播，捕获中间层特征
# =============================================================================
class EncoderFeatureExtractor:
    """
    对 LagerNVS 的 EncDec 模型中的 Reconstructor（VGGT Encoder）进行特征提取。
    
    提取的特征层级：
    ├── Layer 1: VGGT Aggregator 最后一对 (frame, global) block 输出 
    │            → 拼接后的 tokens，维度 2*embed_dim = 2048
    │            → 包含 camera_token + register_tokens + patch_tokens
    │
    ├── Layer 2: 裁剪掉特殊 tokens 后的纯 image patch tokens
    │            → 维度仍为 2048，仅保留与图像 patch 对应的 tokens
    │
    ├── Layer 3: geo_feature_connector 全连接层投影后
    │            → 维度降至 renderer_hidden_size = 768
    │
    └── Layer 4: geo_feature_norm (LayerNorm) 归一化后
                 → 最终送入 Decoder (Renderer) 的特征
    """

    def __init__(self, model):
        self.model = model
        self.reconstructor = model.reconstructor
        self.features = {}
        self._hooks = []

    def _register_hooks(self):
        """注册前向钩子以捕获各层输出。"""

        # 钩子1: 捕获 VGGT 主干网络的原始输出
        def hook_vggt_output(module, input, output):
            self.features["vggt_raw_output"] = output.detach().clone()

        # 钩子2: 捕获 geo_feature_connector 投影后的输出
        def hook_geo_connector(module, input, output):
            self.features["geo_connector_output"] = output.detach().clone()

        # 钩子3: 捕获 geo_feature_norm (LayerNorm) 归一化后的输出
        def hook_geo_norm(module, input, output):
            self.features["geo_norm_output"] = output.detach().clone()

        h1 = self.reconstructor.vggt.register_forward_hook(hook_vggt_output)
        h2 = self.reconstructor.geo_feature_connector.register_forward_hook(hook_geo_connector)
        h3 = self.reconstructor.geo_feature_norm.register_forward_hook(hook_geo_norm)
        self._hooks = [h1, h2, h3]

    def _remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []

    @torch.no_grad()
    def extract(self, images, cam_token, dtype):
        """
        执行一次编码过程，同时捕获各层特征。

        Args:
            images: (B, V, 3, H, W) 输入图像
            cam_token: (B, V, 11) 相机 token
            dtype: 计算精度

        Returns:
            dict: 包含各层特征的字典
        """
        self.features = {}
        self._register_hooks()

        try:
            with torch.amp.autocast(device_type="cuda", dtype=dtype):
                # 调用 Reconstructor 的前向传播
                rec_tokens = self.reconstructor(images, cam_token)
                self.features["final_rec_tokens"] = rec_tokens.detach().clone()
        finally:
            self._remove_hooks()

        return self.features


# =============================================================================
# 可视化工具函数
# =============================================================================
def tokens_to_spatial(tokens, h_patches, w_patches):
    """
    将展平的 token 序列重塑为空间 2D 特征图。
    
    Args:
        tokens: (B, V, num_patches, C) 或 (B*V, num_patches, C)
        h_patches: patch 网格的高度方向数量
        w_patches: patch 网格的宽度方向数量
    
    Returns:
        (B*V, C, h_patches, w_patches) 的空间特征图
    """
    if tokens.dim() == 4:
        b, v, p, c = tokens.shape
        tokens = tokens.reshape(b * v, p, c)
    bv, p, c = tokens.shape
    assert p == h_patches * w_patches, f"Patch count mismatch: {p} != {h_patches}*{w_patches}"
    feat_map = tokens.permute(0, 2, 1).reshape(bv, c, h_patches, w_patches)
    return feat_map


def pca_colorize(feat_map, n_components=3):
    """
    使用 PCA 将高维特征图降维到 3 个分量，作为 RGB 伪彩色图。

    Args:
        feat_map: (N, C, H, W) 的特征图
        n_components: PCA 保留的主成分数（默认 3 对应 RGB）

    Returns:
        (N, 3, H, W) 的 [0,1] 范围伪彩色图
    """
    N, C, H, W = feat_map.shape
    # 展平所有空间位置: (N*H*W, C)
    pixels = feat_map.permute(0, 2, 3, 1).reshape(-1, C).float()

    # 中心化
    mean = pixels.mean(dim=0, keepdim=True)
    pixels_centered = pixels - mean

    # SVD 提取前3个主成分
    # 使用 torch.linalg.svd，只取前 n_components
    U, S, Vh = torch.linalg.svd(pixels_centered, full_matrices=False)
    components = Vh[:n_components]  # (3, C)

    # 投影到主成分空间
    projected = pixels_centered @ components.T  # (N*H*W, 3)

    # 归一化到 [0, 1]
    for i in range(n_components):
        col = projected[:, i]
        vmin, vmax = col.min(), col.max()
        if vmax - vmin > 1e-8:
            projected[:, i] = (col - vmin) / (vmax - vmin)
        else:
            projected[:, i] = 0.5

    # 重塑回空间维度
    colorized = projected.reshape(N, H, W, n_components).permute(0, 3, 1, 2)
    return colorized


def channel_mean_heatmap(feat_map):
    """
    计算特征图在通道维度上的均值，得到单通道热力图。
    
    Args:
        feat_map: (N, C, H, W)
    
    Returns:
        (N, H, W) 的热力图
    """
    return feat_map.float().mean(dim=1)


def channel_norm_heatmap(feat_map):
    """
    计算特征图在通道维度上的 L2 范数，得到特征强度图。
    
    Args:
        feat_map: (N, C, H, W)
    
    Returns:
        (N, H, W) 的热力图
    """
    return feat_map.float().norm(dim=1)


def save_feature_grid(feature_maps_dict, original_images, output_path,
                      h_patches, w_patches, image_size_hw):
    """
    创建一个大型特征可视化网格图并保存。

    每行对应一个输入视图，每列对应一种可视化：
    [原图 | VGGT原始特征PCA | 投影后特征PCA | 通道均值热力图 | 通道范数热力图]

    Args:
        feature_maps_dict: 包含各层特征的字典
        original_images: (B, V, 3, H, W) 原始输入图像
        output_path: 保存路径
        h_patches: patch 网格高度
        w_patches: patch 网格宽度
        image_size_hw: (H, W) 原始图片尺寸
    """
    b, v, _, H, W = original_images.shape

    # --- 准备各层特征的空间形式 ---
    # 1) VGGT Aggregator 的最终输出（裁剪特殊 tokens 后的 image patches）
    vggt_raw = feature_maps_dict["vggt_raw_output"]  # (B, V, P_total, 2C)
    patch_start_idx = feature_maps_dict["patch_start_idx"]
    vggt_patch_tokens = vggt_raw[:, :, patch_start_idx:, :]  # 只保留 patch tokens
    vggt_spatial = tokens_to_spatial(vggt_patch_tokens, h_patches, w_patches)

    # 2) geo_feature_connector 投影后的特征
    geo_conn = feature_maps_dict["geo_connector_output"]  # (B, V, num_patches, 768)
    geo_conn_spatial = tokens_to_spatial(geo_conn, h_patches, w_patches)

    # 3) 最终 LayerNorm 后的特征（送入 Decoder 前的最终特征）
    geo_norm = feature_maps_dict["geo_norm_output"]  # (B, V, num_patches, 768)
    geo_norm_spatial = tokens_to_spatial(geo_norm, h_patches, w_patches)

    # --- PCA 伪彩色 ---
    pca_vggt = pca_colorize(vggt_spatial)
    pca_geo_conn = pca_colorize(geo_conn_spatial)
    pca_geo_norm = pca_colorize(geo_norm_spatial)

    # --- 热力图 ---
    heatmap_mean_vggt = channel_mean_heatmap(vggt_spatial)
    heatmap_norm_vggt = channel_norm_heatmap(vggt_spatial)
    heatmap_mean_norm = channel_mean_heatmap(geo_norm_spatial)
    heatmap_norm_norm = channel_norm_heatmap(geo_norm_spatial)

    # --- 绘制大图 ---
    num_views = b * v
    num_cols = 7  # 原图 + PCA×3 + 热力图均值 + 热力图范数(vggt) + 热力图范数(final)
    fig, axes = plt.subplots(
        num_views, num_cols, 
        figsize=(num_cols * 4, num_views * 4),
        dpi=120
    )
    if num_views == 1:
        axes = axes[np.newaxis, :]

    col_titles = [
        "Original Input",
        "VGGT Output\nPCA Colorized (2048d)",
        "After Connector\nPCA Colorized (768d)",
        "After LayerNorm\nPCA Colorized (768d)",
        "VGGT Features\nChannel Mean Heatmap",
        "VGGT Features\nL2 Norm Heatmap",
        "Final Features\nL2 Norm Heatmap",
    ]

    for view_idx in range(num_views):
        # 原图
        orig_img = original_images.reshape(num_views, 3, H, W)[view_idx]
        orig_np = orig_img.permute(1, 2, 0).cpu().numpy()
        orig_np = np.clip(orig_np, 0, 1)
        axes[view_idx, 0].imshow(orig_np)
        axes[view_idx, 0].set_ylabel(f"View {view_idx}", fontsize=14, fontweight="bold")

        # PCA - VGGT 原始
        pca_v = pca_vggt[view_idx].permute(1, 2, 0).cpu().numpy()
        # 上采样到原图尺寸显示
        pca_v_up = F.interpolate(
            pca_vggt[view_idx:view_idx+1], size=(H, W), mode="bilinear", align_corners=False
        )[0].permute(1, 2, 0).cpu().numpy()
        axes[view_idx, 1].imshow(np.clip(pca_v_up, 0, 1))

        # PCA - Connector 投影后
        pca_c = F.interpolate(
            pca_geo_conn[view_idx:view_idx+1], size=(H, W), mode="bilinear", align_corners=False
        )[0].permute(1, 2, 0).cpu().numpy()
        axes[view_idx, 2].imshow(np.clip(pca_c, 0, 1))

        # PCA - LayerNorm 后
        pca_n = F.interpolate(
            pca_geo_norm[view_idx:view_idx+1], size=(H, W), mode="bilinear", align_corners=False
        )[0].permute(1, 2, 0).cpu().numpy()
        axes[view_idx, 3].imshow(np.clip(pca_n, 0, 1))

        # 热力图 - VGGT 通道均值
        hm_mean = F.interpolate(
            heatmap_mean_vggt[view_idx:view_idx+1].unsqueeze(0), size=(H, W), mode="bilinear", align_corners=False
        )[0, 0].cpu().numpy()
        im4 = axes[view_idx, 4].imshow(hm_mean, cmap="viridis")
        plt.colorbar(im4, ax=axes[view_idx, 4], fraction=0.046, pad=0.04)

        # 热力图 - VGGT L2范数
        hm_norm_v = F.interpolate(
            heatmap_norm_vggt[view_idx:view_idx+1].unsqueeze(0), size=(H, W), mode="bilinear", align_corners=False
        )[0, 0].cpu().numpy()
        im5 = axes[view_idx, 5].imshow(hm_norm_v, cmap="inferno")
        plt.colorbar(im5, ax=axes[view_idx, 5], fraction=0.046, pad=0.04)

        # 热力图 - 最终特征 L2范数
        hm_norm_n = F.interpolate(
            heatmap_norm_norm[view_idx:view_idx+1].unsqueeze(0), size=(H, W), mode="bilinear", align_corners=False
        )[0, 0].cpu().numpy()
        im6 = axes[view_idx, 6].imshow(hm_norm_n, cmap="magma")
        plt.colorbar(im6, ax=axes[view_idx, 6], fraction=0.046, pad=0.04)

    # 设置列标题
    for col_idx, title in enumerate(col_titles):
        axes[0, col_idx].set_title(title, fontsize=12, fontweight="bold", pad=10)

    # 去掉所有子图的坐标轴刻度
    for ax_row in axes:
        for ax in ax_row:
            ax.set_xticks([])
            ax.set_yticks([])

    plt.suptitle(
        "LagerNVS VGGT Encoder Feature Visualization\n"
        "(Intermediate Encoder Outputs before Decoder/Renderer)",
        fontsize=16, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=120)
    plt.close()
    print(f"[✓] 特征网格图已保存到: {output_path}")


def save_individual_feature_maps(feature_maps_dict, original_images, output_dir,
                                  h_patches, w_patches, image_size_hw):
    """
    为每个视图单独保存高分辨率特征图。

    Args:
        feature_maps_dict, original_images, output_dir: 同上
        h_patches, w_patches: patch 网格尺寸
        image_size_hw: (H, W)
    """
    b, v, _, H, W = original_images.shape
    num_views = b * v
    patch_start_idx = feature_maps_dict["patch_start_idx"]

    # VGGT 原始 patch tokens
    vggt_raw = feature_maps_dict["vggt_raw_output"]
    vggt_patches = vggt_raw[:, :, patch_start_idx:, :]
    vggt_spatial = tokens_to_spatial(vggt_patches, h_patches, w_patches)

    # 最终特征
    geo_norm = feature_maps_dict["geo_norm_output"]
    geo_norm_spatial = tokens_to_spatial(geo_norm, h_patches, w_patches)

    # PCA
    pca_vggt = pca_colorize(vggt_spatial)
    pca_final = pca_colorize(geo_norm_spatial)

    for view_idx in range(num_views):
        view_dir = os.path.join(output_dir, f"view_{view_idx:02d}")
        os.makedirs(view_dir, exist_ok=True)

        # 保存原图
        orig = original_images.reshape(num_views, 3, H, W)[view_idx]
        orig_np = (orig.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        plt.imsave(os.path.join(view_dir, "original.png"), orig_np)

        # PCA - VGGT
        pca_v_up = F.interpolate(
            pca_vggt[view_idx:view_idx+1], size=(H, W), mode="bilinear", align_corners=False
        )[0].permute(1, 2, 0).cpu().numpy()
        plt.imsave(os.path.join(view_dir, "pca_vggt_2048d.png"), np.clip(pca_v_up, 0, 1))

        # PCA - Final
        pca_f_up = F.interpolate(
            pca_final[view_idx:view_idx+1], size=(H, W), mode="bilinear", align_corners=False
        )[0].permute(1, 2, 0).cpu().numpy()
        plt.imsave(os.path.join(view_dir, "pca_final_768d.png"), np.clip(pca_f_up, 0, 1))

        # 热力图 - L2 范数
        hm = channel_norm_heatmap(geo_norm_spatial[view_idx:view_idx+1])[0]
        hm_up = F.interpolate(
            hm.unsqueeze(0).unsqueeze(0), size=(H, W), mode="bilinear", align_corners=False
        )[0, 0].cpu().numpy()

        fig, ax = plt.subplots(1, 1, figsize=(8, 8 * H / W), dpi=150)
        im = ax.imshow(hm_up, cmap="inferno")
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(f"View {view_idx} - Final Encoded Feature L2 Norm", fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(view_dir, "heatmap_l2norm_final.png"), dpi=150, bbox_inches="tight")
        plt.close()

        # 保存叠加图（热力图叠加在原图上）
        fig, ax = plt.subplots(1, 1, figsize=(8, 8 * H / W), dpi=150)
        ax.imshow(orig_np)
        ax.imshow(hm_up, cmap="jet", alpha=0.45)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"View {view_idx} - Feature Heatmap Overlay", fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(view_dir, "overlay_heatmap.png"), dpi=150, bbox_inches="tight")
        plt.close()

    print(f"[✓] 逐视图特征图已保存到: {output_dir}/view_XX/")


def save_feature_stats(feature_maps_dict, output_path):
    """打印并保存各层特征的统计信息。"""
    stats_lines = []
    stats_lines.append("=" * 70)
    stats_lines.append("LagerNVS VGGT Encoder 特征统计信息")
    stats_lines.append("=" * 70)

    layers = [
        ("VGGT Aggregator 原始输出", "vggt_raw_output"),
        ("geo_feature_connector 投影后", "geo_connector_output"),
        ("geo_feature_norm 归一化后", "geo_norm_output"),
        ("最终 Reconstructor 输出", "final_rec_tokens"),
    ]

    for name, key in layers:
        if key in feature_maps_dict:
            feat = feature_maps_dict[key].float()
            line = (
                f"\n[{name}]\n"
                f"  形状 (Shape): {list(feat.shape)}\n"
                f"  数据类型: {feat.dtype}\n"
                f"  均值 (Mean): {feat.mean().item():.6f}\n"
                f"  标准差 (Std): {feat.std().item():.6f}\n"
                f"  最小值 (Min): {feat.min().item():.6f}\n"
                f"  最大值 (Max): {feat.max().item():.6f}\n"
                f"  L2范数均值: {feat.norm(dim=-1).mean().item():.4f}\n"
            )
            stats_lines.append(line)

    stats_text = "\n".join(stats_lines)
    print(stats_text)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(stats_text)
    print(f"[✓] 特征统计已保存到: {output_path}")


# =============================================================================
# 主函数
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="LagerNVS VGGT Encoder 特征可视化"
    )
    parser.add_argument(
        "--images", nargs="+", required=True,
        help="输入图片路径（与 minimal_inference.py 相同）",
    )
    parser.add_argument(
        "--output_dir", type=str, default="encoder_feature_vis",
        help="可视化结果保存目录（默认: encoder_feature_vis）",
    )
    parser.add_argument(
        "--model_repo", type=str, default="facebook/lagernvs_general_512",
        help="HuggingFace 模型仓库 ID",
    )
    parser.add_argument(
        "--attention_type", type=str, default="bidirectional_cross_attention",
        choices=["bidirectional_cross_attention", "full_attention"],
        help="渲染器注意力类型",
    )
    parser.add_argument(
        "--target_size", type=int, default=512,
        help="目标图片尺寸（默认: 512）",
    )
    parser.add_argument(
        "--mode", type=str, default="resize",
        choices=["resize", "square_crop"],
        help="图像预处理模式",
    )
    parser.add_argument(
        "--gpu", type=str, default="0",
        help="GPU ID（sets CUDA_VISIBLE_DEVICES）",
    )
    args = parser.parse_args()

    # =========================================================================
    # 第一步：设备与精度设置
    # =========================================================================
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = (
        torch.bfloat16
        if device == "cuda" and torch.cuda.get_device_capability()[0] >= 8
        else torch.float16
    )
    print(f"Device: {device}, dtype: {dtype}")

    os.makedirs(args.output_dir, exist_ok=True)

    # =========================================================================
    # 第二步：加载并预处理输入图像
    # =========================================================================
    image_names = args.images
    num_cond_views = len(image_names)

    images = load_and_preprocess_images(
        image_names, mode=args.mode, target_size=args.target_size, patch_size=8
    )
    images = images.to(device).unsqueeze(0)  # (1, V, 3, H, W)
    image_size_hw = (images.shape[-2], images.shape[-1])
    print(f"加载了 {num_cond_views} 张图像, 形状: {images.shape}")

    # =========================================================================
    # 第三步：加载 LagerNVS 主模型
    # =========================================================================
    print(f"正在加载模型: {args.model_repo} ...")
    model = EncDec_VitB8(
        pretrained_vggt=False,
        attention_to_features_type=args.attention_type,
    )
    ckpt_path = hf_hub_download(args.model_repo, filename="model.pt")
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu")["model"])
    model.to(device)
    model.eval()
    print(f"模型已加载: {sum(p.numel() for p in model.parameters()):,} 参数")

    # =========================================================================
    # 第四步：构造 cam_token（与推理时一致的全零 token）
    # =========================================================================
    # 在推理管线中, cam_token 包含 scale 归一化信息。
    # 这里我们只关注编码器特征, 使用全零 cam_token 即可
    cam_tokens = torch.zeros(1, num_cond_views, 11, device=device)

    # =========================================================================
    # 第五步：提取 Encoder 各层特征
    # =========================================================================
    print("正在提取 VGGT Encoder 中间层特征...")
    extractor = EncoderFeatureExtractor(model)
    features = extractor.extract(images, cam_tokens, dtype)

    # 补充 patch_start_idx 信息
    features["patch_start_idx"] = model.reconstructor.vggt.aggregator.patch_start_idx

    # 计算 VGGT 内部使用的 patch 网格尺寸
    # Reconstructor 会将图像 resize 到 518px 长边, patch_size=14
    _, _, _, h_in, w_in = images.shape
    vggt_imsize = 518
    vggt_patch_size = 14
    if h_in > w_in:
        tgt_h = vggt_imsize
        tgt_w = (int(tgt_h * w_in / h_in) // vggt_patch_size) * vggt_patch_size
    else:
        tgt_w = vggt_imsize
        tgt_h = (int(tgt_w * h_in / w_in) // vggt_patch_size) * vggt_patch_size
    h_patches = tgt_h // vggt_patch_size
    w_patches = tgt_w // vggt_patch_size
    print(f"VGGT 内部分辨率: {tgt_h}x{tgt_w}, patch 网格: {h_patches}x{w_patches}")
    print(f"patch_start_idx = {features['patch_start_idx']} (camera=1 + register=4)")

    # =========================================================================
    # 第六步：打印特征统计信息
    # =========================================================================
    stats_path = os.path.join(args.output_dir, "feature_stats.txt")
    save_feature_stats(features, stats_path)

    # =========================================================================
    # 第七步：保存特征网格总览图
    # =========================================================================
    grid_path = os.path.join(args.output_dir, "feature_grid.png")
    print(f"正在绘制特征网格总览图...")
    save_feature_grid(
        features, images, grid_path,
        h_patches, w_patches, image_size_hw
    )

    # =========================================================================
    # 第八步：保存逐视图高分辨率特征图
    # =========================================================================
    print(f"正在保存逐视图特征图...")
    save_individual_feature_maps(
        features, images, args.output_dir,
        h_patches, w_patches, image_size_hw
    )

    # =========================================================================
    # 第九步：保存原始特征张量 (.pt 文件) 以供后续研究
    # =========================================================================
    tensor_save_path = os.path.join(args.output_dir, "raw_features.pt")
    save_dict = {}
    for key, val in features.items():
        if isinstance(val, torch.Tensor):
            save_dict[key] = val.cpu()
        else:
            save_dict[key] = val
    torch.save(save_dict, tensor_save_path)
    print(f"[✓] 原始特征张量已保存到: {tensor_save_path}")

    print("\n" + "=" * 60)
    print("所有可视化已完成！输出目录:")
    print(f"  {os.path.abspath(args.output_dir)}/")
    print("  ├── feature_grid.png          (总览网格图)")
    print("  ├── feature_stats.txt         (统计信息)")
    print("  ├── raw_features.pt           (原始特征张量)")
    for i in range(num_cond_views):
        print(f"  ├── view_{i:02d}/")
        print(f"  │   ├── original.png")
        print(f"  │   ├── pca_vggt_2048d.png    (VGGT 原始特征 PCA)")
        print(f"  │   ├── pca_final_768d.png    (最终编码特征 PCA)")
        print(f"  │   ├── heatmap_l2norm_final.png (L2范数热力图)")
        print(f"  │   └── overlay_heatmap.png   (热力图叠加)")
    print("=" * 60)


if __name__ == "__main__":
    main()
