import math
import logging
from functools import partial
from collections import OrderedDict
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import to_2tuple

from lib.models.layers.patch_embed import PatchEmbed
from lib.models.ostrackcmp.vit import VisionTransformer
from lib.models.layers.attn import Attention
from timm.models.layers import Mlp, DropPath, trunc_normal_, lecun_normal_
from lib.models.compression.ce import candidate_elimination
from lib.models.compression.te import template_elimination

_logger = logging.getLogger(__name__)


class CETEBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 keep_ratio_x=1.0, keep_ratio_dz=1.0,
                 num_patches_template=None,
                 ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.keep_ratio_x = keep_ratio_x
        self.keep_ratio_dz = keep_ratio_dz
        self.num_patches_template = num_patches_template

    def forward(self, xz, global_index_z, global_index_x, ce_template_mask=None, keep_ratio_x=None, keep_ratio_dz=None, num_template=1):
        x_attn, attn = self.attn(self.norm1(xz), return_attention=True)
        xz = xz + self.drop_path(x_attn)
        lens_z = global_index_z.shape[1]
        lens_x = global_index_x.shape[1]

        removed_index_x = None
        removed_index_z = None

        if self.keep_ratio_x < 1 and (keep_ratio_x is None or keep_ratio_x < 1):
            keep_ratio_x = self.keep_ratio_x if keep_ratio_x is None else keep_ratio_x
            xz, global_index_x, removed_index_x = candidate_elimination(
                tokens=xz, attn=attn, lens_z=lens_z, keep_ratio=keep_ratio_x,
                global_index_x=global_index_x, box_mask_z=ce_template_mask, num_template=num_template,
                ori_lens_1z=self.num_patches_template,
            )

        if self.keep_ratio_dz < 1 and (keep_ratio_dz is None or keep_ratio_dz < 1):
            keep_ratio_dz = self.keep_ratio_dz if keep_ratio_dz is None else keep_ratio_dz
            xz, global_index_z, removed_index_z = template_elimination(
                tokens=xz, attn=attn, lens_z=lens_z, keep_ratio=keep_ratio_dz,
                global_index_z=global_index_z, box_mask_z=ce_template_mask, num_template=num_template,
                ori_lens_1z=self.num_patches_template,
            )

        xz = xz + self.drop_path(self.mlp(self.norm2(xz)))
        return xz, global_index_z, removed_index_z, global_index_x, removed_index_x, attn

class VisionTransformerCETE(VisionTransformer):
    """ Vision Transformer with candidate elimination (CE) module

    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929

    Includes distillation token & head support for `DeiT: Data-efficient Image Transformers`
        - https://arxiv.org/abs/2012.12877
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, weight_init='',
                 ce_loc=None,
                 ce_keep_ratio=None,
                 dte_loc=None,
                 dte_keep_ratio=None,
                 num_patches_template=None,
                 ):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
            weight_init: (str): weight init scheme
        """
        # super().__init__()
        super().__init__()
        if isinstance(img_size, tuple):
            self.img_size = img_size
        else:
            self.img_size = to_2tuple(img_size)
        self.patch_size = patch_size
        self.in_chans = in_chans

        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        self.num_patches_template = num_patches_template

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        blocks = []
        ce_index = 0
        dte_index = 0
        self.ce_loc = ce_loc
        self.dte_loc = dte_loc
        for i in range(depth):
            ce_keep_ratio_i = 1.0
            if ce_loc is not None and i in ce_loc:
                ce_keep_ratio_i = ce_keep_ratio[ce_index]
                ce_index += 1

            dte_keep_ratio_i = 1.0
            if dte_loc is not None and i in dte_loc:
                dte_keep_ratio_i = dte_keep_ratio[dte_index]
                dte_index += 1

            blocks.append(
                CETEBlock(
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                    attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer,
                    keep_ratio_x=ce_keep_ratio_i, keep_ratio_dz=dte_keep_ratio_i,
                    num_patches_template=num_patches_template,
                )
            )

        self.blocks = nn.Sequential(*blocks)
        self.norm = norm_layer(embed_dim)

        self.init_weights(weight_init)

    def prepare_tokens(self, template_list, search_list, template_anno_list):
        B = search_list[0].size(0)

        num_template = len(template_list)
        num_search = len(search_list)

        z = torch.stack(template_list, dim=1)  # (b,n,c,h,w)
        z = z.view(-1, *z.size()[2:])  # (bn,c,h,w)
        x = torch.stack(search_list, dim=1)  # (b,n,c,h,w)
        x = x.view(-1, *x.size()[2:])  # (bn,c,h,w)
        z_anno = torch.stack(template_anno_list, dim=1)  # (b,n,4)
        z_anno = z_anno.view(-1, *z_anno.size()[2:])  # (bn,4)

        z = self.patch_embed(z)     # (b*num_template, l, c)
        x = self.patch_embed(x)     # (b*num_search, l, c)

        if self.add_cls_token:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            cls_tokens = cls_tokens + self.cls_pos_embed

        z += self.pos_embed_z
        x += self.pos_embed_x

        if self.add_sep_seg:
            z += self.template_segment_pos_embed
            x += self.search_segment_pos_embed

        z = z.view(-1, num_template, z.size(-2), z.size(-1))  # (b,num_template,l,c)
        z = z.reshape(z.size(0), -1, z.size(-1))  # (b,num_template*l,c)

        xz = torch.cat((x, z), dim=1)
        if self.add_cls_token:
            xz = torch.cat([cls_tokens, xz], dim=1)

        return xz

    def forward_features(self, template_list, search_list, template_anno_list,
                         ce_template_mask=None, ce_keep_rate=None, dte_keep_rate=None,
                         return_last_attn=False,
                         ):
        xz = self.prepare_tokens(template_list, search_list, template_anno_list)
        xz = self.pos_drop(xz)
        B, _, C = xz.shape

        lens_z = self.pos_embed_z.shape[1] * len(template_list)
        lens_x = self.pos_embed_x.shape[1] * len(search_list)

        global_index_z = torch.linspace(0, lens_z - 1, lens_z).to(xz.device)
        global_index_z = global_index_z.repeat(B, 1)
        removed_indexes_z = []

        global_index_x = torch.linspace(0, lens_x - 1, lens_x).to(xz.device)
        global_index_x = global_index_x.repeat(B, 1)
        removed_indexes_x = []

        ce_template_mask = ce_template_mask[:, :self.num_patches_template]

        for i, blk in enumerate(self.blocks):
            xz, global_index_z, removed_index_z, global_index_x, removed_index_x, attn = blk(
                xz,
                global_index_z=global_index_z,
                global_index_x=global_index_x,
                ce_template_mask=ce_template_mask,
                keep_ratio_x=ce_keep_rate,
                keep_ratio_dz=dte_keep_rate,
                num_template=len(template_list),
            )

            if self.ce_loc is not None and i in self.ce_loc:
                removed_indexes_x.append(removed_index_x)

            if self.dte_loc is not None and i in self.dte_loc:
                removed_indexes_z.append(removed_index_z)

        xz = self.norm(xz)
        lens_x_new = global_index_x.shape[1]
        lens_z_new = global_index_z.shape[1]

        if self.add_cls_token:
            cls_token = xz[:, 0]
        z = xz[:, -lens_z_new:]
        x = xz[:, -lens_x_new-lens_z_new:-lens_z_new]

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
            index_all = torch.cat([global_index_x, removed_indexes_cat], dim=1)
            x = torch.zeros_like(x).scatter_(dim=1, index=index_all.unsqueeze(-1).expand(B, -1, C).to(torch.int64), src=x)

        if self.add_cls_token:
            xz = torch.cat([cls_token, x, z], dim=1)
        else:
            xz = torch.cat([x, z], dim=1)

        aux_dict = {
            "attn": attn,
            "removed_indexes_x": removed_indexes_x,  # used for visualization
        }

        return xz, aux_dict

    def forward(self, template_list, search_list, template_anno_list, **kwargs):
        ce_template_mask = kwargs.pop("ce_template_mask", None)
        ce_keep_rate = kwargs.pop("ce_keep_rate", None)
        dte_keep_rate = kwargs.pop("dte_keep_rate", None)
        x, aux_dict = self.forward_features(template_list, search_list, template_anno_list,
                                            ce_template_mask=ce_template_mask,
                                            ce_keep_rate=ce_keep_rate,
                                            dte_keep_rate=dte_keep_rate,
                                            )

        return x, aux_dict


def _create_vision_transformer(pretrained=False, **kwargs):
    model = VisionTransformerCETE(**kwargs)

    if pretrained:
        if 'npz' in pretrained:
            model.load_pretrained(pretrained, prefix='')
        else:
            checkpoint = torch.load(pretrained, map_location="cpu")
            missing_keys, unexpected_keys = model.load_state_dict(checkpoint["model"], strict=False)
            print('Load pretrained model from: ' + pretrained)

    return model


def vit_base_patch16_224_cete(pretrained=False, **kwargs):
    """ ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer(pretrained=pretrained, **model_kwargs)
    return model