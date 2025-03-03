# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/main/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

import logging
import math
from functools import partial
from typing import Callable, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.utils.checkpoint
from dinov2.layers import MemEffAttention, Mlp
from dinov2.layers import NestedTensorBlock as Block
from dinov2.layers import PatchEmbed, SwiGLUFFNFused
from torch.nn.init import trunc_normal_
from dinov2.models.help import Merge_block, Model_level_Adapeter
from dinov2.models.help  import VitInputLevelAdapter as Input_level_Adapeter
logger = logging.getLogger("dinov2")


def named_apply(fn: Callable, module: nn.Module, name="", depth_first=True, include_root=False) -> nn.Module:
    if not depth_first and include_root:
        fn(module=module, name=name)
    for child_name, child_module in module.named_children():
        child_name = ".".join((name, child_name)) if name else child_name
        named_apply(fn=fn, module=child_module, name=child_name, depth_first=depth_first, include_root=True)
    if depth_first and include_root:
        fn(module=module, name=name)
    return module


class BlockChunk(nn.ModuleList):
    def forward(self, x):
        for b in self:
            x = b(x)
        return x

import torch.nn.functional as F

class DinoVisionTransformer(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=384,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        ffn_bias=True,
        proj_bias=True,
        drop_path_rate=0.0,
        drop_path_uniform=False,
        init_values=None,  # for layerscale: None or 0 => no layerscale
        embed_layer=PatchEmbed,
        act_layer=nn.GELU,
        block_fn=Block,
        ffn_layer="mlp",
        block_chunks=1,
        num_register_tokens=0,
        interpolate_antialias=False,
        interpolate_offset=0.1,
        # RAW adapter parameters
        w_lut=True,
        light_mode='normal',
        lut_dim=32,
        k_size=3,
        merge_ratio=1.0,
        model_adapter_path=None,
        input_level_adapter_path=None,
        fea_c_s = [384, 768, 1920],
        ada_c_s = [16, 32, 64],
        mid_c_s = [384, 576, 768],
        # fea_c_s = [768, 1536, 2304],
        # ada_c_s = [16, 32, 64],
        # mid_c_s = [768, 1152, 1536],
    ):
        super().__init__()
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.num_features = self.embed_dim = embed_dim
        self.num_tokens = 1
        self.n_blocks = depth
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.num_register_tokens = num_register_tokens
        self.interpolate_antialias = interpolate_antialias
        self.interpolate_offset = interpolate_offset

        # RAW adapter configuration
        self.w_lut = w_lut
        self.light_mode = light_mode
        self.lut_dim = lut_dim
        self.k_size = k_size
        self.merge_ratio = merge_ratio

        # Patch embedding
        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        # CLS token and positional embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        assert num_register_tokens >= 0
        self.register_tokens = (
            nn.Parameter(torch.zeros(1, num_register_tokens, embed_dim)) if num_register_tokens else None
        )

        if drop_path_uniform is True:
            dpr = [drop_path_rate] * depth
        else:
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        if ffn_layer == "mlp":
            logger.info("using MLP layer as FFN")
            ffn_layer = Mlp
        elif ffn_layer == "swiglufused" or ffn_layer == "swiglu":
            logger.info("using SwiGLU layer as FFN")
            ffn_layer = SwiGLUFFNFused
        elif ffn_layer == "identity":
            logger.info("using Identity layer as FFN")

            def f(*args, **kwargs):
                return nn.Identity()

            ffn_layer = f
        else:
            raise NotImplementedError
        
        # Transformer blocks
        blocks_list = [
            block_fn(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                ffn_bias=ffn_bias,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                init_values=init_values,
            )
            for i in range(depth)
        ]
        if block_chunks > 0:
            self.chunked_blocks = True
            chunked_blocks = []
            chunksize = depth // block_chunks
            for i in range(0, depth, chunksize):
                chunked_blocks.append([nn.Identity()] * i + blocks_list[i : i + chunksize])
            self.blocks = nn.ModuleList([BlockChunk(p) for p in chunked_blocks])
        else:
            self.chunked_blocks = False
            self.blocks = nn.ModuleList(blocks_list)

        # Final normalization and head
        self.norm = norm_layer(embed_dim)
        self.head = nn.Identity()

        # Mask token
        self.mask_token = nn.Parameter(torch.zeros(1, embed_dim))

        # Initialize RAW adapter
        if self.w_lut:
            self.pre_encoder = Input_level_Adapeter(mode=light_mode, lut_dim=lut_dim, k_size=k_size, w_lut=w_lut)
            # self.model_adapter = Model_level_Adapeter(in_c=in_chans, w_lut=w_lut)
            self.model_adapter = Model_level_Adapeter(in_c=3, in_dim=ada_c_s[0], w_lut=self.w_lut)
            if model_adapter_path is not None:
                print("Loading model adapter:", model_adapter_path)
                adapter_state = torch.load(model_adapter_path, map_location="cpu")
                self.model_adapter.load_state_dict(adapter_state, strict=False)
            if input_level_adapter_path is not None:
                print("Loading input-level adapter:", input_level_adapter_path)
                adapter_state = torch.load(input_level_adapter_path, map_location="cpu")
                self.pre_encoder.load_state_dict(adapter_state, strict=False)
            # Merge block for combining features
            # self.merge_blocks = nn.ModuleList([
            #     Merge_block(fea_c=embed_dim, ada_c=ada_dim, mid_c=embed_dim, return_ada=True),
            #     Merge_block(fea_c=embed_dim, ada_c=ada_dim, mid_c=embed_dim, return_ada=True),
            #     Merge_block(fea_c=embed_dim, ada_c=ada_dim, mid_c=embed_dim, return_ada=False),
            # ])

            self.merge_1 = Merge_block(fea_c=fea_c_s[0], ada_c=ada_c_s[0], mid_c=mid_c_s[0], return_ada=True)
            self.merge_2 = Merge_block(fea_c=fea_c_s[1], ada_c=ada_c_s[1], mid_c=mid_c_s[1], return_ada=True)
            self.merge_3 = Merge_block(fea_c=fea_c_s[2], ada_c=ada_c_s[2], mid_c=mid_c_s[2], return_ada=False)
            self.merge_blocks = [self.merge_1, self.merge_2, self.merge_3]

        # Initialize weights
        self.init_weights()
        print("DinoVisionTransformer initialized.")

    def init_weights(self):
        trunc_normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.cls_token, std=1e-6)
        if self.register_tokens is not None:
            nn.init.normal_(self.register_tokens, std=1e-6)
        named_apply(init_weights_vit_timm, self)

    def interpolate_pos_encoding(self, x, w, h):
        previous_dtype = x.dtype
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        pos_embed = self.pos_embed.float()
        class_pos_embed = pos_embed[:, 0]
        patch_pos_embed = pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size
        w0, h0 = w0 + 0.1, h0 + 0.1

        sqrt_N = math.sqrt(N)
        sx, sy = float(w0) / sqrt_N, float(h0) / sqrt_N
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(sqrt_N), int(sqrt_N), dim).permute(0, 3, 1, 2),
            scale_factor=(sx, sy),
            mode="bicubic",
            antialias=self.interpolate_antialias,
        )
        assert int(w0) == patch_pos_embed.shape[-2]
        assert int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1).to(previous_dtype)

    def prepare_tokens_with_masks(self, x, masks=None):
        B, nc, w, h = x.shape

        x_raw = self.pre_encoder(x)

        if self.w_lut:  # I1, I2, I3, I4
            ada = self.model_adapter([x_raw[0], x_raw[1], x_raw[2], x_raw[3]])
        else:  # I1, I2, I3
            ada = self.model_adapter([x_raw[0], x_raw[1], x_raw[2]])

        # print("X befor x_raw", x.shape, x_raw[-1].shape, ada.shape)
        x = x_raw[-1]

        # Patch embedding
        x = self.patch_embed(x)  # [B, num_patches, embed_dim]
        ada = ada.reshape(ada.shape[0], ada.shape[1], -1) 
        
        # print("X befor x_raw", x.shape, x_raw[-1].shape, ada.shape)

        batch_size, channels, features = ada.shape
       
        if masks is not None:
            x = torch.where(masks.unsqueeze(-1), self.mask_token.to(x.dtype).unsqueeze(0), x)
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = x + self.interpolate_pos_encoding(x, w, h)
        if self.register_tokens is not None:
            x = torch.cat(
                (x[:, :1], self.register_tokens.expand(x.shape[0], -1, -1), x[:, 1:]), dim=1
            )

        # print("X after patch embedding:", x.shape, ada.shape)
        target_seq_len = x.shape[1] 
        linear_proj = nn.Linear(features, target_seq_len).to(ada.device).to(ada.dtype)
        ada = linear_proj(ada).permute(0, 2, 1) 


        return x, ada
        # return x

    def forward_features(self, x, masks=None):
        
        if isinstance(x, list):
            return self.forward_features_list(x, masks)

        # print("Original x shape: ", x.shape)
        x, ada = self.prepare_tokens_with_masks(x, masks)
        # print("ADAPTER after prepare tokens: ", ada.shape, x.shape)
        # x = self.prepare_tokens_with_masks(x, masks)

        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if self.w_lut and ada is not None and i < len(self.merge_blocks):
                x, ada  = self.merge_blocks[i](x, ada, ratio=self.merge_ratio)


        x_norm = self.norm(x)
        return {
            "x_norm_clstoken": x_norm[:, 0],
            "x_norm_regtokens": x_norm[:, 1 : self.num_register_tokens + 1],
            "x_norm_patchtokens": x_norm[:, self.num_register_tokens + 1 :],
            "x_prenorm": x,
            "masks": masks,
        }


    def forward_features_list(self, x_list, masks_list):
        outputs = []
        for x, masks in zip(x_list, masks_list):
            x, ada = self.prepare_tokens_with_masks(x, masks)
            print("ADAPTER after prepare tokens: ", ada.shape)
            # x = self.prepare_tokens_with_masks(x, masks)
            for i, blk in enumerate(self.blocks):
                x = blk(x)
                # if self.w_lut and ada is not None:
                #     x, ada = self.merge_block(x, ada, ratio=self.merge_ratio)                
                if self.w_lut and ada is not None and i < len(self.merge_blocks):
                    x, ada = self.merge_blocks[i](x, ada, ratio=self.merge_ratio)

            x_norm = self.norm(x)
            outputs.append({
                "x_norm_clstoken": x_norm[:, 0],
                "x_norm_regtokens": x_norm[:, 1 : self.num_register_tokens + 1],
                "x_norm_patchtokens": x_norm[:, self.num_register_tokens + 1 :],
                "x_prenorm": x,
                "masks": masks,
            })
        return outputs


    def _get_intermediate_layers_not_chunked(self, x, n=1):
        x = self.prepare_tokens_with_masks(x)
        output, total_block_len = [], len(self.blocks)
        blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in blocks_to_take:
                output.append(x)
        assert len(output) == len(blocks_to_take), f"only {len(output)} / {len(blocks_to_take)} blocks found"
        return output

    def _get_intermediate_layers_chunked(self, x, n=1):
        x = self.prepare_tokens_with_masks(x)
        output, i, total_block_len = [], 0, len(self.blocks[-1])
        blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        for block_chunk in self.blocks:
            for blk in block_chunk[i:]:
                x = blk(x)
                if i in blocks_to_take:
                    output.append(x)
                i += 1
        assert len(output) == len(blocks_to_take), f"only {len(output)} / {len(blocks_to_take)} blocks found"
        return output

    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        n: Union[int, Sequence] = 1,
        reshape: bool = False,
        return_class_token: bool = False,
        norm=True,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]]]:
        if self.chunked_blocks:
            outputs = self._get_intermediate_layers_chunked(x, n)
        else:
            outputs = self._get_intermediate_layers_not_chunked(x, n)
        if norm:
            outputs = [self.norm(out) for out in outputs]
        class_tokens = [out[:, 0] for out in outputs]
        outputs = [out[:, 1:] for out in outputs]
        if reshape:
            B, _, w, h = x.shape
            outputs = [
                out.reshape(B, w // self.patch_size, h // self.patch_size, -1).permute(0, 3, 1, 2).contiguous()
                for out in outputs
            ]
        if return_class_token:
            return tuple(zip(outputs, class_tokens))
        return tuple(outputs)

    def forward(self, *args, is_training=False, **kwargs):
        ret = self.forward_features(*args, **kwargs)
        if is_training:
            return ret
        else:
            return self.head(ret["x_norm_clstoken"])


def init_weights_vit_timm(module: nn.Module, name: str = ""):
    """ViT weight initialization (as in timm) for reproducibility."""
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def vit_small(patch_size=16, num_register_tokens=0, **kwargs):
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=MemEffAttention),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model


def vit_base(patch_size=16, num_register_tokens=0, **kwargs):
    print("HERE AM III")
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=MemEffAttention),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model


def vit_large(patch_size=16, num_register_tokens=0, **kwargs):
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=MemEffAttention),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model


def vit_giant2(patch_size=16, num_register_tokens=0, **kwargs):
    """
    Close to ViT-giant, with embed-dim 1536 and 24 heads (embed-dim per head 64)
    """
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=1536,
        depth=40,
        num_heads=24,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=MemEffAttention),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model
