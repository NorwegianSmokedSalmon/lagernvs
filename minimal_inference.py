# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Minimal end-to-end inference script for LagerNVS.

This script demonstrates the full pipeline:
  1. Load input images
  2. Create a target camera trajectory (using VGGT for pose estimation)
  3. Download and load the LagerNVS checkpoint from HuggingFace
  4. Render novel views
  5. Save output as an MP4 video

Prerequisites:
  - GPU with CUDA support (bfloat16 on Ampere+ GPUs, float16 otherwise)
  - HuggingFace token with access to the gated model repo.
    Set via: export HF_TOKEN=hf_your_token_here
    See README.md "Model Access" section for details.
  - Internet access for downloading VGGT (~4GB) and the LagerNVS checkpoint.
    On Meta devvms, prefix the command with `with-proxy`.

Usage:
  python minimal_inference.py --images path/to/img1.png path/to/img2.png
  python minimal_inference.py --images images/input_000000.png images/input_000001.png

Available checkpoints (set via --model_repo and --attention_type):
  General (512px):  facebook/lagernvs_general_512   attention=bidirectional_cross_attention
  Re10k (256px):    facebook/lagernvs_re10k_2v_256  attention=full_attention
  DL3DV (256px):    facebook/lagernvs_dl3dv_2-6_v_256 attention=bidirectional_cross_attention
"""

import argparse
import os
import sys

# Pre-parse --gpu to set CUDA_VISIBLE_DEVICES before PyTorch initializes CUDA
if "--gpu" in sys.argv:
    gpu_idx = sys.argv.index("--gpu")
    if gpu_idx + 1 < len(sys.argv):
        os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[gpu_idx + 1]

import torch
from eval.export import save_video
from huggingface_hub import hf_hub_download
from models.encoder_decoder import EncDec_VitB8
from vggt.utils.load_fn import load_and_preprocess_images
from vis import create_target_camera_path, render_chunked


def main():
    parser = argparse.ArgumentParser(description="LagerNVS minimal inference")
    parser.add_argument(
        "--images",
        nargs="+",
        required=True,
        help="Paths to 1 or more input images",
    )
    parser.add_argument(
        "--video_length",
        type=int,
        default=100,
        help="Number of frames to render (default: 100)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output_video.mp4",
        help="Output video path (default: output_video.mp4)",
    )
    parser.add_argument(
        "--model_repo",
        type=str,
        default="facebook/lagernvs_general_512",
        help="HuggingFace repo ID for the checkpoint",
    )
    parser.add_argument(
        "--attention_type",
        type=str,
        default="bidirectional_cross_attention",
        choices=["bidirectional_cross_attention", "full_attention"],
        help=(
            "Attention type for the renderer. "
            "Use 'full_attention' for Re10k model, "
            "'bidirectional_cross_attention' for General/DL3DV models."
        ),
    )
    parser.add_argument(
        "--target_size",
        type=int,
        default=512,
        help="Target size in pixels (default: 512)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="resize",
        choices=["resize", "square_crop"],
        help=(
            "Image preprocessing mode. "
            "'resize' preserves aspect ratio with longer side = target_size (General model). "
            "'square_crop' center-crops to square then resizes to target_size (256 models)."
        ),
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default="0",
        help="GPU ID to use (sets CUDA_VISIBLE_DEVICES)",
    )
    args = parser.parse_args()

    # -------------------------------------------------------------------------
    # 第一步：设备 (Device) 与数据类型 (dtype) 的设置
    # -------------------------------------------------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 如果是 Ampere 架构（例如 RTX 30/40系列，算力>=8.0）及以上的显卡，这里会自动使用支持更好的 bfloat16 数据类型。
    # 否则退化到普通的 float16，以保证显存可以支撑。
    dtype = (
        torch.bfloat16
        if device == "cuda" and torch.cuda.get_device_capability()[0] >= 8
        else torch.float16
    )
    print(f"Device: {device}, dtype: {dtype}")

    # -------------------------------------------------------------------------
    # 第二步：加载并对输入图像进行预处理
    # -------------------------------------------------------------------------
    # 使用 load_and_preprocess_images 函数来预处理输入的源图像：
    # "resize" 模式：保持原图宽高比，将最长边缩放到 target_size（适合通用 512 模型）。
    # "square_crop" 模式：先进行中心方形裁剪，再强制缩放到 target_size 宽高（适合 256 模型）。
    # 返回的张量尺寸预设为 (图片数量, 3通道, H, W)。
    image_names = args.images
    num_cond_views = len(image_names)

    images = load_and_preprocess_images(
        image_names, mode=args.mode, target_size=args.target_size, patch_size=8
    )
    # 给这批图片增加一个 Batch 的维度，形状变为：(1, 视图数量, 3, H, W)
    images = images.to(device).unsqueeze(0)
    image_size_hw = (images.shape[-2], images.shape[-1])
    print(f"Loaded {num_cond_views} images, shape: {images.shape}")

    # -------------------------------------------------------------------------
    # 第三步：创建目标相机的运动轨迹 (Trajectory)
    # -------------------------------------------------------------------------
    # 核心调用：create_target_camera_path 函数。
    # 这里的逻辑是：先使用 VGGT 算法（第一次运行会自动下载约4GB的预训练权重）去猜测这两张/多张输入图片在三维空间中大约的相对相机位姿。
    # 然后，在这些猜测的相机位姿之间，插值出一条非常平滑的 B-spline（B样条）摄像机曲线运动轨迹。
    # 如果只输入了一张图（单视图），它就会生成一段相机缓缓向前推进推镜头的目标运动轨迹。
    # 
    # 返回值说明：
    #   rays (射线): 包含当前条件视图和要生成的一百个帧对应的 3D 普吕克坐标 (Plucker ray coords)，格式：(1, num_cond_views + video_length, 6, H, W)
    #   cam_tokens: 向渲染模型传递尺度 (Scale) 归一化信息用的相机条件特征，形状：(1, num_cond_views + video_length, 11)
    print("Creating target camera path (downloads VGGT on first run)...")
    rays, cam_tokens = create_target_camera_path(
        image_names,
        args.video_length,
        num_cond_views,
        image_size_hw,
        device,
        dtype,
        mode=args.mode,
    )
    print(f"Rays shape: {rays.shape}, cam_tokens shape: {cam_tokens.shape}")

    # -------------------------------------------------------------------------
    # 第四步：加载主模型 (LagerNVS) 的权重与运行架构
    # -------------------------------------------------------------------------
    # EncDec_VitB8 = 对应的是带 ViT-B/8 配制的编解码器 (Encoder-Decoder) 体系：
    #   - 编码器 (Encoder)：负责基于提取出原图的几何特征信息。
    #   - 解码器 (Decoder)：这里的解码器是一个有 12 层 Transformer 的大规模渲染器 (patch_size=8, hidden_size=768)，利用前面的提取结果去真正完成 3D NVS 渲染。
    #
    # attention_to_features_type 参数决定了渲染器如何对待和融合传入的图像提取特征：
    #   "bidirectional_cross_attention" — 用于 General (通用) 和 DL3DV 的模型
    #   "full_attention"                — 用于 Re10k 专用模型
    print(f"Loading model from {args.model_repo}...")
    model = EncDec_VitB8(
        pretrained_vggt=False,  # 因为整个下载到的检查点已经打包包含了预训练权重，因此这里设为 false
        attention_to_features_type=args.attention_type,
    )

    # 这一步负责从 HuggingFace Hub 下载并加载模型的权重文件 (model.pt)，大小为 4.41 GB。
    # 需要在机器环境中设置好 HF 的鉴权 Token 且通过门控权限申请。
    ckpt_path = hf_hub_download(args.model_repo, filename="model.pt")
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu")["model"])
    model.to(device)
    model.eval()  # 转为推理模式，固化权重的随机Dropout之类操作
    print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")

    # -------------------------------------------------------------------------
    # 第五步：在新视角轨迹上执行 3D 大模型渲染
    # -------------------------------------------------------------------------
    # render_chunked : 这是一个分块渲染策略函数。如果我们一口气把 100 帧的需求扔进带自注意力的庞大 Transformer 模型里，显存会原地崩掉。
    # 因此，它实际上是把目标帧分割为每批 16 帧去分别通过大模型慢慢渲染。然后自动拼接起来。
    # 内部自带了 PyTorch 的 torch.amp.autocast 以开启自动混合精度计算（节省显存）。
    #
    # 接收参数：输入原图像 cond_images, 计算出的 3D 射线 rays, 相机标签 cam_tokens
    # 最终输出的 video_out 维度将会是 (B, 视频总帧数, 3通道RGB, H高, W宽)
    print(f"Rendering {args.video_length} frames...")
    with torch.no_grad():
        with torch.amp.autocast(device_type="cuda", dtype=dtype):
            video_out = render_chunked(
                model,
                (images, rays, cam_tokens),
                num_cond_views=num_cond_views,
            )
    print(f"Output video shape: {video_out.shape}")

    # -------------------------------------------------------------------------
    # 第六步：保存最后生成的视频
    # -------------------------------------------------------------------------
    # 借助于工具函数将 (T, C, H, W) 这种序列张量转存为用户可正常播放的 MP4 动画文件
    save_video(video_out[0], args.output)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
