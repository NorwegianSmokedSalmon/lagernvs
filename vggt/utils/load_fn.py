# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the VGGT license found at
# https://github.com/facebookresearch/vggt/blob/main/LICENSE.txt

import torch
from PIL import Image
from torchvision import transforms as TF


def load_and_preprocess_images(
    image_path_list, mode="square_crop", target_size=512, patch_size=8
):
    """
    加载并对图像进行预处理，使其符合模型输入的要求。

    参数:
        image_path_list (list): 图像文件路径的列表。
        mode (str): 预处理模式。
            - "square_crop": 先将原始图像中心裁剪为最大的正方形，然后再统一缩放拉伸到 target_size x target_size。适合 256 模型。
            - "resize": 保持图像原始的宽高比例，将较长的一边强行缩放至等于 target_size。
              较短的一边会通过四舍五入对齐到 patch_size 的整数倍。如果短边被缩减到小于
              0.5 * target_size，则会抛出异常强行中止。适合 512 通用模型。
        target_size (int): 目标图像尺寸的像素值 (默认: 512)
        patch_size (int): 补丁像素大小。在 "resize" 模式下，保证短边像素数能够被该值整除 (默认: 8)

    返回:
        torch.Tensor: 预处理后拼接成批次的图像张量，形状为 (N, 3, H, W) [N代表数量, 3代表RGB通道, H高, W宽]

    抛出异常:
        ValueError: 当输入图像列表为空时；mode 不合法时；或者在 "resize" 模式下长宽比过于极端时。
    """
    if len(image_path_list) == 0:
        raise ValueError("At least 1 image is required")

    if mode not in ["square_crop", "resize"]:
        raise ValueError("Mode must be 'square_crop' or 'resize'")

    images = []
    shapes = set()
    to_tensor = TF.ToTensor()

    for image_path in image_path_list:
        with open(image_path, "rb") as f:
            img = Image.open(f)
            img.load()

        if img.mode == "RGBA":
            background = Image.new("RGBA", img.size, (255, 255, 255, 255))
            img = Image.alpha_composite(background, img)
        img = img.convert("RGB")

        width, height = img.size

        if mode == "square_crop":
            short_side = min(width, height)
            left = (width - short_side) // 2
            top = (height - short_side) // 2
            img = img.crop((left, top, left + short_side, top + short_side))
            img = img.resize((target_size, target_size), Image.Resampling.BICUBIC)
            img = to_tensor(img)
        else:  # mode == "resize"
            if width >= height:
                new_width = target_size
                new_height = (
                    round(height * (target_size / width) / patch_size) * patch_size
                )
            else:
                new_height = target_size
                new_width = (
                    round(width * (target_size / height) / patch_size) * patch_size
                )

            shorter_side = min(new_width, new_height)
            if shorter_side < 0.5 * target_size:
                raise ValueError(
                    f"Image aspect ratio too extreme: shorter side ({shorter_side}px) "
                    f"is less than 0.5 * target_size ({0.5 * target_size:.0f}px). "
                    f"Original size: {width}x{height}. "
                    f"Consider using mode='square_crop' instead."
                )

            img = img.resize((new_width, new_height), Image.Resampling.BICUBIC)
            img = to_tensor(img)

        shapes.add((img.shape[1], img.shape[2]))
        images.append(img)

    if len(shapes) > 1:
        raise ValueError(
            f"Input images have different shapes after preprocessing: {shapes}. "
            f"All images must have the same resolution. Please crop or resize "
            f"your input images so they share approximately the same intrinsic "
            f"parameters (resolution and field of view)."
        )

    images = torch.stack(images)

    if len(image_path_list) == 1:
        if images.dim() == 3:
            images = images.unsqueeze(0)

    return images
