from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import logging
import os
from functools import partial

from torch import nn
import torchvision
import torch.optim
from einops import rearrange
import collections.abc as container_abcs
from itertools import repeat

from lib.config import config
from lib.config import update_config
from lib.models import build_model, QuickGELU, Block, LayerNorm


# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, container_abcs.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple


class PatchMerge(nn.Module):
    """
    Equal to Conv Embedding
    """

    def __init__(self,
                 patch_size=7,
                 in_chans=3,
                 embed_dim=64,
                 stride=4,
                 padding=2,
                 norm_layer=None):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size

        self.proj = nn.Conv2d(
            in_chans, embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=padding
        )
        self.norm = norm_layer(embed_dim) if norm_layer else None

    def forward(self, x):
        x = self.proj(x)

        B, C, H, W = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')
        if self.norm:
            x = self.norm(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)
        return x


class PatchExpand(nn.Module):
    """
    Patch Expand
    """

    def __init__(self,
                 patch_size=3,
                 in_chans=384,
                 expand_dim=192,
                 stride=2,
                 padding=1,
                 norm_layer=None, ):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size

        self.up_sample = nn.Upsample(scale_factor=(stride, stride), mode='bilinear')
        self.proj = nn.Conv2d(in_chans,
                              expand_dim,
                              kernel_size=patch_size,
                              stride=1,
                              padding=padding)
        self.norm = norm_layer(expand_dim) if norm_layer else None

    def forward(self, x):
        x1, x2 = x
        x1 = self.up_sample(x1)
        x = self.proj(torch.cat([x1, x2], dim=1))
        B, C, H, W = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')
        if self.norm:
            x = self.norm(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)
        return x


class VANet(nn.Module):
    def __init__(self,
                 in_chans=3,
                 act_layer=QuickGELU,
                 cfg='./experiments/imagenet/cvt/cvt-13-224x224.yaml',
                 weights='./weights/CvT-13-224x224-IN-1k.pth',
                 embed_dims=[64, 192, 384],
                 depths=[1, 2, 10],
                 mlp_ratios=[4, 4, 4],
                 num_heads=[1, 3, 6],
                 strides=[4, 2, 2],
                 proj_drop=0.1,
                 attn_drop=0.1,
                 drop_path=0.1,
                 norm_layer=partial(LayerNorm, eps=1e-5),
                 num_class=1,
                 **kwargs):
        super().__init__()
        update_config(config, cfg=cfg)
        self.CvT = build_model(config)

        if weights is not None:
            state_dict = torch.load(weights, map_location='cpu')
            self.CvT.load_state_dict(state_dict, strict=False)

        if in_chans == 1:
            self.CvT.stage0.patch_embed = PatchMerge(patch_size=2 * strides[0] - 1,
                                                     in_chans=1,
                                                     embed_dim=embed_dims[0],
                                                     stride=strides[0],
                                                     padding=strides[0] // 2,
                                                     norm_layer=norm_layer)
        # encoder stage0
        self.encoder_stage0 = self.CvT.stage0

        # encoder stage1
        self.encoder_stage1 = self.CvT.stage1

        # encoder stage2
        self.encoder_stage2_merge = self.CvT.stage2.patch_embed
        self.encoder_stage2_blk = self.CvT.stage2.blocks[:depths[2] // 2]
        for step, instance in enumerate(self.encoder_stage2_blk):
            if step % 2 == 1:
                instance.attn.conv_proj_k.conv.stride = 1
                instance.attn.conv_proj_v.conv.stride = 1

        # decoder stage0
        self.decoder_stage0_blk = self.CvT.stage2.blocks[depths[2] // 2:]
        self.decoder_stage1_expand = PatchExpand(patch_size=2 * strides[2] - 1,
                                                 in_chans=embed_dims[2] + embed_dims[1],
                                                 expand_dim=embed_dims[1],
                                                 stride=strides[2],
                                                 padding=strides[2] // 2,
                                                 norm_layer=norm_layer)

        # decoder stage1
        self.decoder_stage1_blk = nn.ModuleList([Block(embed_dims[1],
                                                       embed_dims[1],
                                                       num_heads=num_heads[1],
                                                       mlp_ratio=mlp_ratios[1],
                                                       drop=proj_drop,
                                                       attn_drop=attn_drop,
                                                       drop_path=drop_path,
                                                       act_layer=act_layer,
                                                       norm_layer=norm_layer,
                                                       with_cls_token=False,
                                                       stride_kv=2, )
                                                 for i in range(depths[1])], )

        # decoder stage2
        self.decoder_stage2_expand = PatchExpand(patch_size=2 * strides[1] - 1,
                                                 in_chans=embed_dims[1] + embed_dims[0],
                                                 expand_dim=embed_dims[0],
                                                 stride=strides[1],
                                                 padding=strides[1] // 2,
                                                 norm_layer=norm_layer, )
        self.decoder_stage2_blk = nn.ModuleList([Block(embed_dims[0],
                                                       embed_dims[0],
                                                       num_heads=num_heads[0],
                                                       mlp_ratio=mlp_ratios[0],
                                                       drop=proj_drop,
                                                       attn_drop=attn_drop,
                                                       drop_path=drop_path,
                                                       act_layer=act_layer,
                                                       norm_layer=norm_layer,
                                                       with_cls_token=False,
                                                       stride_kv=2, )
                                                 for i in range(depths[0])])

        # mask head(s)
        self.mask_head0 = nn.Conv2d(in_channels=embed_dims[2],
                                    out_channels=num_class,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1, )
        self.mask_head1 = nn.Conv2d(in_channels=embed_dims[2],
                                    out_channels=num_class,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1, )
        self.mask_head2 = nn.Conv2d(in_channels=embed_dims[1],
                                    out_channels=num_class,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1, )
        self.mask_head3 = nn.Conv2d(in_channels=embed_dims[0],
                                    out_channels=num_class,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1, )

    def forward(self, x, cue=None):
        f0 = self.encoder_stage0(x)

        f1 = self.encoder_stage1(f0)

        f = self.encoder_stage2_merge(f1)
        B, C, H, W = f.shape
        f = rearrange(f, 'b c h w -> b (h w) c')
        for i, blk in enumerate(self.encoder_stage2_blk):
            if i % 2 == 1:
                f = blk(f, H, W, cue, scale=2)
            else:
                f = blk(f, H, W)

        f = rearrange(f, 'b (h w) c -> b c h w', h=H, w=W)
        out0 = self.mask_head0(f)
        f = rearrange(f, 'b c h w -> b (h w) c')
        for i, blk in enumerate(self.decoder_stage0_blk):
            # if i % 2 == 1:
            f = blk(f, H, W, r=out0)
        # else:
        #  f = blk(f, H, W)

        f = rearrange(f, 'b (h w) c -> b c h w', h=H, w=W)
        out1 = self.mask_head1(f)

        f = self.decoder_stage1_expand([f, f1])
        B, C, H, W = f.shape
        f = rearrange(f, 'b c h w -> b (h w) c')
        for i, blk in enumerate(self.decoder_stage1_blk):
            # if i % 2 == 1:
            f = blk(f, H, W, r=out1)
        # else:
        #    f = blk(f, H, W)

        f = rearrange(f, 'b (h w) c -> b c h w', h=H, w=W)
        out2 = self.mask_head2(f)
        f = self.decoder_stage2_expand([f, f0])
        B, C, H, W = f.shape
        f = rearrange(f, 'b c h w -> b (h w) c')
        for i, blk in enumerate(self.decoder_stage2_blk):
            f = blk(f, H, W, r=out2)

        f = rearrange(f, 'b (h w) c -> b c h w', h=H, w=W)

        out3 = self.mask_head3(f)

        return out3, out2, out1, out0


if __name__ == "__main__":
    x = torch.randn(2, 3, 352, 352)
    cue = torch.randn(2, 1, 11, 11)
    model = VANet()
    model.load_state_dict(torch.load("./save_model/VANet/epoch_50.pth"))
    print("VANet's encoder-decoder have {} paramerters in total".format(sum(x.numel() for x in model.parameters())))
    out3, out2, out1, out0 = model(x, cue)
    print(out3.shape)