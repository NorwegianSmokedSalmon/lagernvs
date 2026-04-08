# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.renderer import Renderer
from vggt.models.vggt import VGGT


# Main model file
# Consists of
# 1. Encoder (Reconstructor)
#    VGGT-based feature extraction (利用VGGT网络提取图像特征的编码器)
# 2. Decoder (Renderer)
#    Series of (Self-attn, X-attn, MLP) blocks (一系列包含自注意力、交叉注意力和MLP的解码器渲染块)


class EncoderDecoder(nn.Module):
    """
    编码-解码器 (Encoder-Decoder) 整体架构。
    负责将输入的图像和相机特征进行编码，并通过大模型渲染管线输出新视角的图像特征。
    """
    def __init__(
        self,
        depth,
        hidden_size,
        patch_size,
        num_heads,
        freeze_vggt=True,
        pretrained_vggt=True,
        attention_to_features_type="bidirectional_cross_attention",
        pretrained_patch_embed=False,
    ):
        super().__init__()
        self.reconstructor = Reconstructor(
            hidden_size,
            target_patch_size=patch_size,
            pretrained_vggt=pretrained_vggt,
            freeze_vggt=freeze_vggt,
            pretrained_patch_embed=pretrained_patch_embed,
        )
        self.renderer = Renderer(
            depth,
            hidden_size,
            patch_size,
            num_heads,
            attention_to_features_type=attention_to_features_type,
        )

    def forward(
        self,
        images,
        rays,
        cam_token,
        num_cond_views,
        timeit=False,
    ):
        """
        前向传播逻辑：
        1. 剥离出现有条件视角 (Input Views) 与需要预测的目标视角 (Target Views)。
        2. 将条件图像和条件相机Token送入重建器 (Reconstructor) 提取特征。
        3. 对提取特征展平并扩展至目标射线的维度。
        4. 通过渲染器 (Renderer) 结合目标普吕克射线 (Plucker Rays) 渲染出最终的新视角内容。
        """
        # 前 num_cond_views 是已知的条件输入
        input_images = images[:, :num_cond_views, ...]
        cam_token = cam_token[:, :num_cond_views]
        # 后面的射线全都是要新渲染的目标视角射线
        target_rays = rays[:, num_cond_views:]

        v_target = target_rays.shape[1]

        rec_tokens = self.reconstructor(input_images, cam_token)

        rec_tokens = einops.rearrange(rec_tokens, "b v_input p c -> b (v_input p) c")
        rec_tokens = einops.repeat(
            rec_tokens,
            "b np d -> (b v_target) np d",
            v_target=v_target,
        )

        if timeit:
            rendered_images, time_t = self.renderer(
                rec_tokens, target_rays, timeit=timeit
            )
        else:
            rendered_images = self.renderer(rec_tokens, target_rays, timeit=timeit)

        cond_and_rendered_images = torch.cat([input_images, rendered_images], dim=1)

        if timeit:
            return cond_and_rendered_images, time_t

        return cond_and_rendered_images


class Reconstructor(nn.Module):
    """
    重建器模块 (Reconstructor)。
    负责从具有相机先验的条件图像中提取具有泛化能力的3D几何相关的重建特征。
    底层使用视觉基础预训练大模型 (VGGT) 进行处理。
    """

    def __init__(
        self,
        renderer_hidden_size,
        target_patch_size,
        pretrained_vggt=True,
        freeze_vggt=False,
        pretrained_patch_embed=False,
    ):
        super().__init__()
        self.vggt = VGGT(pretrained_patch_embed=pretrained_patch_embed)
        self.freeze_vggt = freeze_vggt
        if pretrained_vggt:
            print("Loading encoder weights from pretrained VGGT")
            vggt_pretrained_state = torch.hub.load_state_dict_from_url(
                "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt",
                map_location="cpu",
            )
            self.vggt.load_state_dict(vggt_pretrained_state, strict=False)
        else:
            print("VGGT weights not used for the encoder")

        # camera token projector (always use 11-dim tokens with scale)
        self.camera_encoding_dim = 11
        vggt_hidden_dim = 1024
        self.vggt_patch_size = 14
        self.target_patch_size = target_patch_size
        self.camera_mlp = nn.Sequential(
            nn.Linear(self.camera_encoding_dim, vggt_hidden_dim, bias=True),
            nn.SiLU(),
            nn.Linear(vggt_hidden_dim, vggt_hidden_dim, bias=True),
        )

        # channel-dim adapter
        self.geo_feature_connector = nn.Linear(1024 * 2, renderer_hidden_size)
        self.geo_feature_norm = nn.LayerNorm(renderer_hidden_size, bias=False)

    def forward(self, input_images, cam_token):
        """
        前向传播：
        Inputs:
            images: (b, v_input, 3, h, w) 输入源图像序列。
            cam_token: (b, v_input, 9或11) 相机条件，如果在没有相机位姿时可为全零。
            
        流程说明:
        把原本的图片经过长边518px归一化与裁剪填充后送到 VGGT 主干中。
        同时对相机的尺度和平移/旋转特征用 MLP 进行初步特征升维映射，作为 conditioning。
        最后通过全连接层 geo_feature_connector 将两组特征适配对接给目标主模型的 Decoder。
        """
        # resize input images so that longer size is 518
        b, v_input, _, h, w = input_images.shape
        input_images = einops.rearrange(input_images, "b v c h w -> (b v) c h w")
        vggt_imsize = 518
        input_camera_cond = self.camera_mlp(cam_token).unsqueeze(2)

        # resize input images so that the side length is divisible by 14
        if h > w:
            tgt_h = vggt_imsize
            tgt_w = (int(tgt_h * w / h) // self.vggt_patch_size) * self.vggt_patch_size
        else:
            tgt_w = vggt_imsize
            tgt_h = (int(tgt_w * h / w) // self.vggt_patch_size) * self.vggt_patch_size
        input_images = F.interpolate(
            input_images, size=(tgt_h, tgt_w), mode="bilinear", antialias=True
        )
        input_images = einops.rearrange(
            input_images, "(b v) c h w -> b v c h w", b=b, v=v_input
        )
        # extract features for the conditioning images
        if self.freeze_vggt:
            with torch.no_grad():
                tokens_vggt_cond = self.vggt(input_images, input_camera_cond).detach()
        else:
            tokens_vggt_cond = self.vggt(input_images, input_camera_cond)

        tokens_vggt_image_cond = tokens_vggt_cond[
            :, :, self.vggt.aggregator.patch_start_idx :, :
        ]

        tokens_vggt_image_cond = self.geo_feature_connector(tokens_vggt_image_cond)
        tokens_vggt_image_cond = self.geo_feature_norm(tokens_vggt_image_cond)

        return tokens_vggt_image_cond


def EncDec_VitB8(**kwargs):
    """
    返回配备了 ViT-B/8 （12层 Transformer, hidden_size=768, patch_size=8）规模的 Encoder-Decoder 模型。
    该架构配置对应于项目中加载并使用的主要推演大模型。
    """
    return EncoderDecoder(
        depth=12, hidden_size=768, patch_size=8, num_heads=12, **kwargs
    )
