from functools import partial
import warnings
import math
import torch
import torch.nn as nn
from timm.models.registry import register_model
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import to_2tuple, drop_path, trunc_normal_
from torch import Tensor, Size
from typing import Union, List

from .fastitpn import (
    _cfg, _shape_t, DropPath, Mlp, ConvMlp, SwiGLU, ConvSwiGLU, ConvMlpBlock, PatchMerge, ConvPatchMerge, PatchEmbed,
    ConvPatchEmbed, RelativePositionBias, DecoupledRelativePositionBias, load_pretrained, Block, Attention
)

from .compression.evit import evit_cem

class EvitAttention(nn.Module):
    def __init__(
            self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., window_size=None,
            attn_head_dim=None, use_decoupled_rel_pos_bias=False, deepnorm=False, subln=False
    ):
        """ window_size: 相对位置编码的窗口大小，而不是窗口注意力，目前没用 """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.deepnorm = deepnorm
        self.subln = subln
        if self.deepnorm or self.subln:
            self.q_proj = nn.Linear(dim, all_head_dim, bias=False)
            self.k_proj = nn.Linear(dim, all_head_dim, bias=False)
            self.v_proj = nn.Linear(dim, all_head_dim, bias=False)
        else:
            self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)

        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        self.rel_pos_bias = None
        self.qk_float = True

        self.window_size = None
        self.relative_position_bias_table = None

        if window_size:
            if use_decoupled_rel_pos_bias:
                self.rel_pos_bias = DecoupledRelativePositionBias(window_size=window_size, num_heads=num_heads)
            else:
                self.window_size = window_size
                self.num_relative_distance = (2 * window_size[0] - 1) * (
                        2 * window_size[1] - 1) + 3  # (2*14-1) * (2*14-1) + 3
                self.relative_position_bias_table = nn.Parameter(
                    torch.zeros(self.num_relative_distance, num_heads))  # 2*Wh-1 * 2*Ww-1, nH
                # cls to token & token 2 cls & cls to cls

                # get pair-wise relative position index for each token inside the window
                coords_h = torch.arange(window_size[0])
                coords_w = torch.arange(window_size[1])
                coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
                coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
                relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
                relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
                relative_coords[:, :, 0] += window_size[0] - 1  # shift to start from 0
                relative_coords[:, :, 1] += window_size[1] - 1
                relative_coords[:, :, 0] *= 2 * window_size[1] - 1
                relative_position_index = \
                    torch.zeros(size=(window_size[0] * window_size[1] + 1,) * 2, dtype=relative_coords.dtype)
                relative_position_index[1:, 1:] = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
                relative_position_index[0, 0:] = self.num_relative_distance - 3
                relative_position_index[0:, 0] = self.num_relative_distance - 2
                relative_position_index[0, 0] = self.num_relative_distance - 1

                self.register_buffer("relative_position_index", relative_position_index)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, rel_pos_bias=None, attn_mask=None, return_attention=False):
        B, N, C = x.shape

        if self.deepnorm or self.subln:
            q = F.linear(input=x, weight=self.q_proj.weight, bias=self.q_bias)
            k = F.linear(input=x, weight=self.k_proj.weight, bias=None)
            v = F.linear(input=x, weight=self.v_proj.weight, bias=self.v_bias)

            q = q.reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)  # B, num_heads, N, C
            k = k.reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
            v = v.reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
        else:
            qkv_bias = None
            if self.q_bias is not None:
                qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
            qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
            qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)  # 3, B, num_heads, N, C
            q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        if self.qk_float:
            attn = (q.float() @ k.float().transpose(-2, -1))
        else:
            attn = (q @ k.transpose(-2, -1))

        if self.relative_position_bias_table is not None:
            relative_position_bias = \
                self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                    self.window_size[0] * self.window_size[1] + 1,
                    self.window_size[0] * self.window_size[1] + 1, -1)  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
            attn = attn + relative_position_bias.unsqueeze(0).type_as(attn)

        if self.rel_pos_bias is not None:
            attn = attn + self.rel_pos_bias().type_as(attn)

        if rel_pos_bias is not None:
            attn = attn + rel_pos_bias.type_as(attn)
        if attn_mask is not None:
            attn_mask = attn_mask.bool()
            attn = attn.masked_fill(~attn_mask[:, None, None, :], float("-inf"))
        attn = attn.softmax(dim=-1).type_as(x)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)

        if return_attention:
            return x, attn
        return x

class Evit_HiViTAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, norm_layer=nn.LayerNorm, window_size=None, attn_head_dim=None,
                 use_decoupled_rel_pos_bias=False,
                 depth=None,
                 postnorm=False,
                 deepnorm=False,
                 subln=False,
                 swiglu=False,
                 naiveswiglu=False,
                 keep_ratio_x=1.0, ###
                 ):
        super().__init__()

        with_attn = num_heads > 0

        self.norm1 = norm_layer(dim) if with_attn else None
        self.attn = EvitAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, window_size=window_size,
            use_decoupled_rel_pos_bias=use_decoupled_rel_pos_bias, attn_head_dim=attn_head_dim,
            deepnorm=deepnorm,
            subln=subln
        ) if with_attn else None

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        if swiglu:
            self.mlp = xops.SwiGLU(in_features=dim, hidden_features=mlp_hidden_dim)  # hidden_features: 2/3
        elif naiveswiglu:
            self.mlp = SwiGLU(in_features=dim, hidden_features=mlp_hidden_dim, subln=subln, norm_layer=norm_layer,)
        else:
            self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, subln=subln, norm_layer=norm_layer)

        if init_values is not None and init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True) if self.attn is not None else None
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

        self.deepnorm = deepnorm
        if self.deepnorm:
            self.alpha = math.pow(2.0 * depth, 0.25)

        self.postnorm = postnorm

        self.keep_ratio_x = keep_ratio_x  ###

    def forward(self, xz, rel_pos_bias=None, attn_mask=None, global_index_z=None, global_index_x=None, keep_ratio_x=None, extra_index_x=None, num_template=1):
        lens_z = global_index_z.shape[1]
        lens_x = global_index_x.shape[1]

        xz_attn, attn = self.attn(self.norm1(xz), rel_pos_bias=rel_pos_bias, attn_mask=attn_mask, return_attention=True)
        xz = xz + self.drop_path(self.gamma_1 * xz_attn)

        # ********** Note: 压缩模块里面都要处理token的分离和合并 **********


        removed_index_x = None
        if self.keep_ratio_x < 1 and (keep_ratio_x is None or keep_ratio_x < 1):
            keep_ratio_x = self.keep_ratio_x if keep_ratio_x is None else keep_ratio_x
            xz, global_index_x, removed_index_x, extra_index_x = evit_cem( xz, attn, lens_z, keep_ratio_x, global_index_x )

        xz = xz + self.drop_path(self.gamma_2 * self.mlp(self.norm2(xz)))
        return xz, global_index_z, global_index_x, removed_index_x, extra_index_x, attn

class Evit_Fast_iTPN(nn.Module):
    def __init__(self, search_size=224,template_size=112, patch_size=16, in_chans=3, embed_dim=512, depth_stage1=3, depth_stage2=3, depth=24,
                 num_heads=8, bridge_mlp_ratio=3., mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0.0, init_values=None, attn_head_dim=None, norm_layer=nn.LayerNorm,
                 patch_norm=False, num_classes=1000, use_mean_pooling=False,
                 init_scale=0.01,
                 cls_token=False,
                 grad_ckpt=False,
                 stop_grad_conv1=False,
                 use_abs_pos_emb=True,
                 use_rel_pos_bias=False,
                 use_shared_rel_pos_bias=False,
                 use_shared_decoupled_rel_pos_bias=False,
                 convmlp=False,
                 postnorm=False,
                 deepnorm=False,
                 subln=False,
                 swiglu=False,
                 naiveswiglu=False,
                 token_type_indicate=False,
                 evit_loc=None, ###
                 evit_keep_ratio=None, ###
                 **kwargs):
        super().__init__()
        self.search_size = search_size
        self.template_size = template_size
        self.token_type_indicate = token_type_indicate
        self.mlp_ratio = mlp_ratio
        self.grad_ckpt = grad_ckpt
        self.num_main_blocks = depth
        self.depth_stage1 = depth_stage1
        self.depth_stage2 = depth_stage2
        self.depth = depth
        self.patch_size = patch_size
        self.num_features = self.embed_dim = embed_dim
        self.convmlp = convmlp
        self.stop_grad_conv1 = stop_grad_conv1
        self.use_rel_pos_bias = use_rel_pos_bias
        self.use_shared_rel_pos_bias = use_shared_rel_pos_bias
        self.use_shared_decoupled_rel_pos_bias = use_shared_decoupled_rel_pos_bias
        self.use_decoupled_rel_pos_bias = False
        self.evit_loc = evit_loc

        mlvl_dims = {'4': embed_dim // 4, '8': embed_dim // 2, '16': embed_dim}
        # split image into non-overlapping patches
        if convmlp:
            self.patch_embed = ConvPatchEmbed(
                search_size=search_size,template_size=template_size, patch_size=patch_size, in_chans=in_chans, embed_dim=mlvl_dims['4'],
                stop_grad_conv1=stop_grad_conv1,
                norm_layer=norm_layer if patch_norm else None)
        else:
            self.patch_embed = PatchEmbed(
                img_size=search_size, patch_size=patch_size, in_chans=in_chans, embed_dim=mlvl_dims['4'],
                norm_layer=norm_layer if patch_norm else None)
        self.num_patches_search = self.patch_embed.num_patches_search
        self.num_patches_template = self.patch_embed.num_patches_template
        if cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        else:
            self.cls_token = None
        if use_abs_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches_search+self.num_patches_template, embed_dim))
        else:
            self.pos_embed = None
        # indicate for tracking
        if self.token_type_indicate:
            self.template_background_token = nn.Parameter(torch.zeros(embed_dim))
            self.template_foreground_token = nn.Parameter(torch.zeros(embed_dim))
            self.search_token = nn.Parameter(torch.zeros(embed_dim))


        self.pos_drop = nn.Dropout(p=drop_rate)

        if use_shared_rel_pos_bias:
            self.rel_pos_bias = RelativePositionBias(window_size=self.patch_embed.patch_shape, num_heads=num_heads)
        else:
            self.rel_pos_bias = None

        if use_shared_decoupled_rel_pos_bias:
            assert self.rel_pos_bias is None
            self.rel_pos_bias = DecoupledRelativePositionBias(window_size=self.patch_embed.patch_shape, num_heads=num_heads)

        self.subln = subln
        self.swiglu = swiglu
        self.naiveswiglu = naiveswiglu

        self.build_blocks(
            depths=[depth_stage1, depth_stage2, depth],
            dims=mlvl_dims,
            num_heads=num_heads,
            bridge_mlp_ratio=bridge_mlp_ratio,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            window_size=self.patch_embed.patch_shape if use_rel_pos_bias else None,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            init_values=init_values,
            attn_head_dim=attn_head_dim,
            postnorm=postnorm,
            deepnorm=deepnorm,
            subln=subln,
            swiglu=swiglu,
            naiveswiglu=naiveswiglu,
            convmlp=convmlp,
            evit_loc=evit_loc,
            evit_keep_ratio=evit_keep_ratio,
        )

        self.norm = nn.Identity() if use_mean_pooling else norm_layer(embed_dim)
        self.fc_norm = norm_layer(embed_dim) if use_mean_pooling else None
        self.head = nn.Identity()

        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=.02)
        if self.cls_token is not None:
            trunc_normal_(self.cls_token, std=.02)

        if isinstance(self.head, nn.Linear):
            trunc_normal_(self.head.weight, std=.02)

        self.apply(self._init_weights)

        if isinstance(self.head, nn.Linear):
            self.head.weight.data.mul_(init_scale)
            self.head.bias.data.mul_(init_scale)

    def build_blocks(self,
                     depths=[3, 3, 24],
                     dims={'4': 128 // 4, '8': 256, '16': 512},
                     num_heads=8,
                     bridge_mlp_ratio=3.,
                     mlp_ratio=4.0,
                     qkv_bias=True,
                     qk_scale=None,
                     window_size=None,
                     drop=0.,
                     attn_drop=0.,
                     drop_path_rate=0.,
                     norm_layer=nn.LayerNorm,
                     init_values=0.,
                     attn_head_dim=None,
                     postnorm=False,
                     deepnorm=False,
                     subln=False,
                     swiglu=False,
                     naiveswiglu=False,
                     convmlp=False,
                     evit_loc=None,
                     evit_keep_ratio=None,
                     ):
        dpr = iter(x.item() for x in torch.linspace(0, drop_path_rate, depths[0] + depths[1] + depths[2]))

        self.blocks = nn.ModuleList()

        if convmlp:
            self.blocks.extend([
                ConvMlpBlock(
                    dim=dims['4'],
                    mlp_ratio=bridge_mlp_ratio,
                    drop_path=next(dpr),
                    norm_layer=norm_layer,
                    init_values=0.,
                    depth=depths[-1],
                    postnorm=postnorm,
                    deepnorm=deepnorm,
                    subln=subln,
                    swiglu=False,
                    naiveswiglu=False,
                ) for _ in range(depths[0])
            ])
            self.blocks.append(ConvPatchMerge(dims['4'], norm_layer))
            self.blocks.extend([
                ConvMlpBlock(
                    dim=dims['8'],
                    mlp_ratio=bridge_mlp_ratio,
                    drop_path=next(dpr),
                    norm_layer=norm_layer,
                    init_values=0.,
                    depth=depths[-1],
                    postnorm=postnorm,
                    deepnorm=deepnorm,
                    subln=subln,
                    swiglu=False,
                    naiveswiglu=False,
                ) for _ in range(depths[1])
            ])
            self.blocks.append(ConvPatchMerge(dims['8'], norm_layer))
        else:
            self.blocks.extend([
                Block(
                    dim=dims['4'],
                    num_heads=0,
                    mlp_ratio=bridge_mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=next(dpr),
                    norm_layer=norm_layer,
                    init_values=init_values,
                    window_size=window_size,
                    depth=depths[-1],
                    postnorm=postnorm,
                    deepnorm=deepnorm,
                    subln=subln,
                    swiglu=swiglu,
                    naiveswiglu=naiveswiglu,
                ) for _ in range(depths[0])
            ])
            self.blocks.append(PatchMerge(dims['4'], norm_layer))
            self.blocks.extend([
                Block(
                    dim=dims['8'],
                    num_heads=0,
                    mlp_ratio=bridge_mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=next(dpr),
                    norm_layer=norm_layer,
                    init_values=init_values,
                    window_size=window_size,
                    depth=depths[-1],
                    postnorm=postnorm,
                    deepnorm=deepnorm,
                    subln=subln,
                    swiglu=swiglu,
                    naiveswiglu=naiveswiglu,
                ) for _ in range(depths[1])
            ])
            self.blocks.append(PatchMerge(dims['8'], norm_layer))

        ######### stage 3 ########
        evit_index = 0
        for i in range(depths[2]):
            evit_keep_ratio_i = 1.0
            if evit_loc is not None and i in evit_loc:
                evit_keep_ratio_i = evit_keep_ratio[evit_index]
                evit_index += 1

            self.blocks.extend([
                Evit_HiViTAttentionBlock(
                    dim=dims['16'],
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=next(dpr),
                    norm_layer=norm_layer,
                    init_values=init_values,
                    window_size=window_size,
                    attn_head_dim=attn_head_dim,
                    depth=depths[-1],
                    postnorm=postnorm,
                    deepnorm=deepnorm,
                    subln=subln,
                    swiglu=swiglu,
                    naiveswiglu=naiveswiglu,
                    keep_ratio_x=evit_keep_ratio_i,
                )
            ])

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        if self.cls_token is not None:
            return {'pos_embed', 'cls_token'}
        return {'pos_embed'}

    def get_classifer(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def create_mask(self, image, image_anno):
        """
            image: (B, C, H, W)
            image_anno: (B, 4)    4: (x,y,w,h)
        """
        height = image.size(2)
        width = image.size(3)

        # Extract bounding box coordinates
        x0 = (image_anno[:, 0] * width).unsqueeze(1)
        y0 = (image_anno[:, 1] * height).unsqueeze(1)
        w = (image_anno[:, 2] * width).unsqueeze(1)
        h = (image_anno[:, 3] * height).unsqueeze(1)

        # Generate pixel indices
        x_indices = torch.arange(width, device=image.device)
        y_indices = torch.arange(height, device=image.device)

        # Create masks for x and y coordinates within the bounding boxes
        x_mask = ((x_indices >= x0) & (x_indices < x0 + w)).float()
        y_mask = ((y_indices >= y0) & (y_indices < y0 + h)).float()

        # Combine x and y masks to get final mask
        mask = x_mask.unsqueeze(1) * y_mask.unsqueeze(2) # (b,h,w)

        return mask

    def prepare_tokens_with_masks(self, template_list, search_list, template_anno_list, text_src, task_index):
        B = search_list[0].size(0)

        num_template = len(template_list)
        num_search = len(search_list)

        z = torch.stack(template_list, dim=1)           # (b,n,c,h,w)
        z = z.view(-1, *z.size()[2:])                   # (bn,c,h,w)
        x = torch.stack(search_list, dim=1)             # (b,n,c,h,w)
        x = x.view(-1, *x.size()[2:])                   # (bn,c,h,w)
        z_anno = torch.stack(template_anno_list, dim=1) # (b,n,4)
        z_anno = z_anno.view(-1, *z_anno.size()[2:])    # (bn,4)

        # Soft Token Type Embedding
        if self.token_type_indicate:
            # generate a foreground mask to 计算前景和背景的权重，前景的权重本质上就是Patch当中属于前景像素的比例
            z_indicate_mask = self.create_mask(z, z_anno)   # (b, h, w) 根据模板的bbox来生成精确的前景背景mask
            z_indicate_mask = z_indicate_mask.unfold(1, self.patch_size, self.patch_size).unfold(2, self.patch_size, self.patch_size) # (b, nh, nw, ph, pw) match the patch embedding
            z_indicate_mask = z_indicate_mask.mean(dim=(3,4)).flatten(1) # (b, nh * nw) elements are in [0,1], float, near to 1 indicates near to foreground, near to 0 indicates near to background

        if self.token_type_indicate:
            # generate the indicate_embeddings for z
            template_background_token = self.template_background_token.unsqueeze(0).unsqueeze(1).expand(z_indicate_mask.size(0), z_indicate_mask.size(1), self.embed_dim)
            template_foreground_token = self.template_foreground_token.unsqueeze(0).unsqueeze(1).expand(z_indicate_mask.size(0), z_indicate_mask.size(1), self.embed_dim)
            weighted_foreground = template_foreground_token * z_indicate_mask.unsqueeze(-1)
            weighted_background = template_background_token * (1 - z_indicate_mask.unsqueeze(-1))
            z_indicate = weighted_foreground + weighted_background


        z = self.patch_embed(z) # (b, c, h, w) --> (b, -1, p_h*inner_p, p_w*inner_p)
        x = self.patch_embed(x) # (b, c, h, w) --> (b, -1, p_h*inner_p, p_w*inner_p)

        # forward stage1&2
        if not self.convmlp and self.stop_grad_conv1:
            x = x.detach() * 0.9 + x * 0.1

        for blk in self.blocks[:-self.num_main_blocks]:
            z = checkpoint.checkpoint(blk, z) if self.grad_ckpt else blk(z)  # bn,c,h,w
            x = checkpoint.checkpoint(blk, x) if self.grad_ckpt else blk(x)  # bn,c,h,w

        x = x.flatten(2).transpose(1, 2)    # bn,l,c: 196/576
        z = z.flatten(2).transpose(1, 2)    # bn,l,c: 49/144

        if self.pos_embed is not None:
            x = x + self.pos_embed[:, :self.num_patches_search, :]
            z = z + self.pos_embed[:, self.num_patches_search:, :]

        if self.token_type_indicate:
            # generate the indicate_embeddings for x
            x_indicate = self.search_token.unsqueeze(0).unsqueeze(1).expand(x.size(0), x.size(1), self.embed_dim)
            # add indicate_embeddings to z and x
            x = x + x_indicate
            z = z + z_indicate


        z = z.view(-1, num_template, z.size(-2), z.size(-1))  # b,n,l,c
        z = z.reshape(z.size(0), -1, z.size(-1))  # b,l,c
        x = x.view(-1, num_search, x.size(-2), x.size(-1))
        x = x.reshape(x.size(0), -1, x.size(-1))

        if text_src is not None:
            xz = torch.cat([x, z, text_src], dim=1)
        else:
            xz = torch.cat([x, z], dim=1)

        if self.cls_token is not None:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            xz = torch.cat([cls_tokens, xz], dim=1)

        return xz

    def forward_features(self, template_list, search_list, template_anno_list, text_src, task_index,
                         ce_template_mask=None, ce_keep_rate=None, return_last_attn=False):
        aux_dict = {"removed_indexes_x": None}
        xz = self.prepare_tokens_with_masks(template_list, search_list, template_anno_list, text_src, task_index)   # return xz = [cls_tokens, x, z, text_src]
        xz = self.pos_drop(xz)
        B, _, C = xz.shape

        # *********************** 我们剪枝的对象在这里 ************************
        # Stage 3: 只在Stage3使用了Attention, 这里的Attention是加强版的Attention
        lens_z = self.num_template * self.num_patches_template
        lens_x = self.num_frames * self.num_patches_search

        global_index_z = torch.linspace(0, lens_z - 1, lens_z).to(xz.device)
        global_index_z = global_index_z.repeat(B, 1)
        removed_indexes_z = []

        global_index_x = torch.linspace(0, lens_x - 1, lens_x).to(xz.device)
        global_index_x = global_index_x.repeat(B, 1)
        removed_indexes_x = []
        token_count_per_block = [xz.shape[1]]

        extra_index_x = None
        extra_indexes_x = []

        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        for i, blk in enumerate(self.blocks[-self.num_main_blocks:]):
            if self.grad_ckpt:
                xz, global_index_z, global_index_x, removed_index_x, extra_index_x, attn = checkpoint.checkpoint(
                    blk, xz, rel_pos_bias, global_index_z=global_index_z, global_index_x=global_index_x,
                    keep_ratio_x=ce_keep_rate, extra_index_x=extra_index_x
                )
            else:
                xz, global_index_z, global_index_x, removed_index_x, extra_index_x, attn = blk(
                    xz, rel_pos_bias, global_index_z=global_index_z, global_index_x=global_index_x,
                    keep_ratio_x=ce_keep_rate, extra_index_x=extra_index_x
                )

            if self.evit_loc is not None and i in self.evit_loc:
                removed_indexes_x.append(removed_index_x)
                extra_indexes_x.append(extra_index_x)
            token_count_per_block.append(xz.shape[1])

        xz = self.norm(xz)
        if self.fc_norm is not None:
            xz = self.fc_norm(xz)

        lens_z_new = global_index_z.shape[1]
        lens_x_new = global_index_x.shape[1]

        cls_token = xz[:, 0, :].unsqueeze(1)
        x_full = xz[:, 1:1+lens_x_new, :]
        z = xz[:, 1+lens_x_new:-1, :]
        text_token = xz[:, -1, :].unsqueeze(1)

        valid_mask = (global_index_x != -1)
        x = x_full[valid_mask].reshape(B, -1, x_full.shape[2])

        # if removed_indexes_x and removed_indexes_x[0] is not None:
        if removed_indexes_x and any(removed_idx_x is not None for removed_idx_x in removed_indexes_x):
            # removed_indexes_cat = torch.cat(removed_indexes_x, dim=1)
            valid_removed = [removed_idx_x for removed_idx_x in removed_indexes_x if removed_idx_x is not None]
            removed_indexes_cat = torch.cat(valid_removed, dim=1)
            # 填充
            pruned_lens_x = lens_x - lens_x_new
            pad_x = torch.zeros([B, pruned_lens_x, x.shape[2]], device=x.device)
            x = torch.cat([x, pad_x], dim=1)
            # 恢复原始的顺序
            global_index_x_valid = global_index_x[global_index_x != -1].reshape(B, -1)
            index_all = torch.cat([global_index_x_valid, removed_indexes_cat], dim=1)
            # index_all = torch.cat([global_index_x, removed_indexes_cat], dim=1)
            x = torch.zeros_like(x).scatter_(dim=1, index=index_all.unsqueeze(-1).expand(B, -1, C).to(torch.int64), src=x)
        xz = torch.cat([cls_token, x, z, text_token], dim=1)

        aux_dict = {
            "attn": attn,
            "removed_indexes_x": removed_indexes_x,  # used for visualization
            "token_count_per_block": token_count_per_block,
            "extra_indexes_x": extra_indexes_x,
        }

        return xz, aux_dict

    def forward(self, template_list, search_list, template_anno_list, text_src, task_index,
                ce_template_mask=None, ce_keep_rate=None):
        xz, aux_dict = self.forward_features(template_list, search_list, template_anno_list, text_src, task_index,
                                             ce_template_mask=ce_template_mask, ce_keep_rate=ce_keep_rate)

        out = [xz]
        return out, aux_dict


@register_model
def evit_fastitpnt(pretrained=False, pos_type="interpolate", pretrain_type="", patchembed_init="copy", **kwargs):
    model = Evit_Fast_iTPN(
        patch_size=16, in_chans=6, embed_dim=384, depth_stage1=1, depth_stage2=1, depth=12, num_heads=6, bridge_mlp_ratio=3.,
        mlp_ratio=3., qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),
        convmlp=True,
        naiveswiglu=True,
        subln=True,
        pos_type=pos_type,
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(pretrain_type, map_location="cpu")
        load_pretrained(model,checkpoint,pos_type,patchembed_init)
    return model


@register_model
def evit_fastitpns(pretrained=False, pos_type="interpolate", pretrain_type="", patchembed_init="copy", **kwargs):
    model = Evit_Fast_iTPN(
        patch_size=16, in_chans=6, embed_dim=384, depth_stage1=2, depth_stage2=2, depth=20, num_heads=6, bridge_mlp_ratio=3.,
        mlp_ratio=3., qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),
        convmlp=True,
        naiveswiglu=True,
        subln=True,
        pos_type=pos_type,
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(pretrain_type, map_location="cpu")
        load_pretrained(model,checkpoint,pos_type,patchembed_init)
    return model


@register_model
def evit_fastitpnb(pretrained=False, pos_type="interpolate", pretrain_type="", patchembed_init="copy", **kwargs):
    model = Evit_Fast_iTPN(
        patch_size=16, in_chans=6, embed_dim=512, depth_stage1=3, depth_stage2=3, depth=24, num_heads=8, bridge_mlp_ratio=3.,
        mlp_ratio=3., qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),
        convmlp=True,
        naiveswiglu=True,
        subln=True,
        pos_type = pos_type,
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(pretrain_type, map_location="cpu")
        load_pretrained(model,checkpoint,pos_type,patchembed_init)
    return model


@register_model
def evit_fastitpnl(pretrained=False, pos_type="interpolate", pretrain_type="", patchembed_init="copy", **kwargs):
    model = Evit_Fast_iTPN(
        patch_size=16, in_chans=6, embed_dim=768, depth_stage1=2, depth_stage2=2, depth=40, num_heads=12, bridge_mlp_ratio=3.,
        mlp_ratio=3., qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),
        convmlp=True,
        naiveswiglu=True,
        subln=True,
        pos_type="interpolate",
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(pretrain_type, map_location="cpu")
        load_pretrained(model,checkpoint,pos_type,patchembed_init)
    return model
