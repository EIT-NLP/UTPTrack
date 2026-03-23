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
from lib.models.compression.ce import candidate_elimination_by_ate
from lib.models.compression.ate import dynamic_template_elimination, static_template_elimination

_logger = logging.getLogger(__name__)


class CEATETTABlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 keep_ratio_x=1.0, keep_ratio_dz=1.0, keep_ratio_sz=1.0,
                 num_patches_template=None, token_type_aware_pruing=None,
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
        self.keep_ratio_sz = keep_ratio_sz
        self.num_patches_template = num_patches_template
        self.token_type_aware_pruing = token_type_aware_pruing

    def forward(self, xz, global_index_sz, global_index_dz, global_index_x, ce_template_mask=None,
                keep_ratio_x=None, keep_ratio_sz=None, keep_ratio_dz=None, num_template=1, z_indicate_mask=None,):
        x_attn, attn = self.attn(self.norm1(xz), return_attention=True)
        xz = xz + self.drop_path(x_attn)

        lens_sz = global_index_sz.shape[1]
        lens_dz = global_index_dz.shape[1]
        # lens_z = lens_sz + lens_dz
        # lens_x = global_index_x.shape[1]

        removed_index_x = None
        removed_index_sz = None
        removed_index_dz = None

        if self.keep_ratio_x < 1 and (keep_ratio_x is None or keep_ratio_x < 1):
            keep_ratio_x = self.keep_ratio_x if keep_ratio_x is None else keep_ratio_x
            xz, global_index_x, removed_index_x = candidate_elimination_by_ate(
                tokens=xz, attn=attn, lens_sz=lens_sz, lens_dz=lens_dz, keep_ratio=keep_ratio_x,
                global_index_x=global_index_x, box_mask_z=ce_template_mask, num_template=num_template,
            )

        if self.keep_ratio_dz < 1 and (keep_ratio_dz is None or keep_ratio_dz < 1):
            keep_ratio_dz = self.keep_ratio_dz if keep_ratio_dz is None else keep_ratio_dz
            xz, global_index_dz, removed_index_dz = dynamic_template_elimination(
                tokens=xz, attn=attn, lens_sz=lens_sz, lens_dz=lens_dz, keep_ratio=keep_ratio_dz,
                global_index_dz=global_index_dz, box_mask_z=ce_template_mask, num_template=num_template,
            )
            # print("lens_dz")
            # print(f"Before Pruning: {lens_dz}")
            lens_dz = global_index_dz.shape[1]
            # print(f"After  Pruning: {lens_dz}")

        if self.keep_ratio_sz < 1 and (keep_ratio_sz is None or keep_ratio_sz < 1):
            keep_ratio_sz = self.keep_ratio_sz if keep_ratio_sz is None else keep_ratio_sz
            xz, global_index_sz, removed_index_sz, ce_template_mask, z_indicate_mask = static_template_elimination(
                tokens=xz, attn=attn, lens_sz=lens_sz, lens_dz=lens_dz, keep_ratio=keep_ratio_sz,
                global_index_sz=global_index_sz, box_mask_z=ce_template_mask, num_template=num_template,
                indicate_mask_sz=z_indicate_mask, token_type_aware_pruing=self.token_type_aware_pruing,
            )

        xz = xz + self.drop_path(self.mlp(self.norm2(xz)))
        return (xz, global_index_sz, global_index_dz, removed_index_sz, removed_index_dz, global_index_x, removed_index_x,
                attn, ce_template_mask, z_indicate_mask)

class VisionTransformerCEATETTA(VisionTransformer):
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
                 ste_loc=None,
                 ste_keep_ratio=None,
                 num_patches_template=None,
                 token_type_aware_pruing=None,
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

        self.token_type_aware_pruing = token_type_aware_pruing  ###
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        blocks = []
        ce_index = 0
        dte_index = 0
        ste_index = 0
        self.ce_loc = ce_loc
        self.dte_loc = dte_loc
        self.ste_loc = ste_loc
        for i in range(depth):
            ce_keep_ratio_i = 1.0
            if ce_loc is not None and i in ce_loc:
                ce_keep_ratio_i = ce_keep_ratio[ce_index]
                ce_index += 1

            dte_keep_ratio_i = 1.0
            if dte_loc is not None and i in dte_loc:
                dte_keep_ratio_i = dte_keep_ratio[dte_index]
                dte_index += 1

            ste_keep_ratio_i = 1.0
            if ste_loc is not None and i in ste_loc:
                ste_keep_ratio_i = ste_keep_ratio[ste_index]
                ste_index += 1

            blocks.append(
                CEATETTABlock(
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                    attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer,
                    keep_ratio_x=ce_keep_ratio_i, keep_ratio_dz=dte_keep_ratio_i, keep_ratio_sz=ste_keep_ratio_i,
                    num_patches_template=num_patches_template, token_type_aware_pruing=self.token_type_aware_pruing,
                )
            )

        self.blocks = nn.Sequential(*blocks)
        self.norm = norm_layer(embed_dim)

        self.init_weights(weight_init)

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

        if self.token_type_aware_pruing:
            # generate a foreground mask to 计算前景和背景的权重，前景的权重本质上就是Patch当中属于前景像素的比例
            z_indicate_mask = self.create_mask(z, z_anno)  # (b, h, w) 根据模板的bbox来生成精确的前景背景mask
            z_indicate_mask = z_indicate_mask.unfold(1, self.patch_size, self.patch_size).unfold(2, self.patch_size, self.patch_size)  # (b, nh, nw, ph, pw) match the patch embedding
            z_indicate_mask = z_indicate_mask.mean(dim=(3, 4)).flatten(1)  # (b, nh * nw) elements are in [0,1], float, near to 1 indicates near to foreground, near to 0 indicates near to background

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

        if self.token_type_aware_pruing:
            z_indicate_mask = z_indicate_mask.view(B, -1)
            return xz, z_indicate_mask
        return xz

    def forward_features(self, template_list, search_list, template_anno_list,
                         ce_template_mask=None, ce_keep_rate=None, dte_keep_rate=None, ste_keep_rate=None,
                         return_last_attn=False,
                         ):
        xz, z_indicate_mask = self.prepare_tokens(template_list, search_list, template_anno_list)

        if self.token_type_aware_pruing == 'FULL_FOREGROUND':
            z_indicate_mask = (z_indicate_mask == 1.0)
        elif self.token_type_aware_pruing == 'ALL_FOREGROUND':
            z_indicate_mask = (z_indicate_mask > 0.0)
        elif self.token_type_aware_pruing == 'SOFT_ALL_FOREGROUND':
            z_indicate_mask = z_indicate_mask   # 不处理,bonus直接就是这个mask
        else:
            raise NotImplementedError

        xz = self.pos_drop(xz)
        B, _, C = xz.shape

        lens_z = self.pos_embed_z.shape[1] * len(template_list)
        lens_x = self.pos_embed_x.shape[1] * len(search_list)

        global_index_z = torch.linspace(0, lens_z - 1, lens_z).to(xz.device)
        global_index_z = global_index_z.repeat(B, 1)
        global_index_sz = global_index_z[:, :self.num_patches_template]
        global_index_dz = global_index_z[:, self.num_patches_template:]
        removed_indexes_sz = []
        removed_indexes_dz = []


        global_index_x = torch.linspace(0, lens_x - 1, lens_x).to(xz.device)
        global_index_x = global_index_x.repeat(B, 1)
        removed_indexes_x = []

        ce_template_mask = ce_template_mask[:, :self.num_patches_template]
        z_indicate_mask = z_indicate_mask[:, :self.num_patches_template]
        for i, blk in enumerate(self.blocks):
            (xz, global_index_sz, global_index_dz, removed_index_sz, removed_index_dz,
             global_index_x, removed_index_x, attn, ce_template_mask, z_indicate_mask) = blk(
                xz,
                global_index_sz=global_index_sz,
                global_index_dz=global_index_dz,
                global_index_x=global_index_x,
                ce_template_mask=ce_template_mask,
                keep_ratio_x=ce_keep_rate,
                keep_ratio_sz=ste_keep_rate,
                keep_ratio_dz=dte_keep_rate,
                num_template=len(template_list),
                z_indicate_mask=z_indicate_mask,
            )

            if self.ce_loc is not None and i in self.ce_loc:
                removed_indexes_x.append(removed_index_x)

            if self.dte_loc is not None and i in self.dte_loc:
                removed_indexes_dz.append(removed_index_dz)

            if self.ste_loc is not None and i in self.ste_loc:
                removed_indexes_sz.append(removed_index_sz)

        # print("11111111111111111-------- removed index x --------------------------------------")
        # print(removed_index_x)
        # print("22222222222222222-------- removed index dz --------------------------------------")
        # print(removed_index_dz)
        # print("33333333333333333-------- removed index sz --------------------------------------")
        # print(removed_index_sz)
        
        xz = self.norm(xz)
        lens_x_new = global_index_x.shape[1]
        lens_dz_new = global_index_dz.shape[1]
        lens_sz_new = global_index_sz.shape[1]

        if self.add_cls_token:
            cls_token = xz[:, 0]
        z = xz[:, -lens_sz_new-lens_dz_new:]
        x = xz[:, -lens_x_new-lens_sz_new-lens_dz_new:-lens_sz_new-lens_dz_new]

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
        ste_keep_rate = kwargs.pop("ste_keep_rate", None)
        x, aux_dict = self.forward_features(template_list, search_list, template_anno_list,
                                            ce_template_mask=ce_template_mask,
                                            ce_keep_rate=ce_keep_rate,
                                            dte_keep_rate=dte_keep_rate,
                                            ste_keep_rate=ste_keep_rate,
                                            )

        return x, aux_dict


def _create_vision_transformer(pretrained=False, **kwargs):
    model = VisionTransformerCEATETTA(**kwargs)

    if pretrained:
        if 'npz' in pretrained:
            model.load_pretrained(pretrained, prefix='')
        else:
            checkpoint = torch.load(pretrained, map_location="cpu")
            missing_keys, unexpected_keys = model.load_state_dict(checkpoint["model"], strict=False)
            print('Load pretrained model from: ' + pretrained)

    return model


def vit_base_patch16_224_ceatetta(pretrained=False, **kwargs):
    """ ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer(pretrained=pretrained, **model_kwargs)
    return model