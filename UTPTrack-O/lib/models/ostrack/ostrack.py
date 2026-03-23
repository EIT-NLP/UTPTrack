"""
Basic OSTrack model.
"""

import os

import torch
from torch import nn
from torch.nn.modules.transformer import _get_clones

from lib.models.layers.head import build_box_head
from lib.models.ostrack.vit import vit_base_patch16_224
from lib.models.ostrack.vit_ce import vit_large_patch16_224_ce, vit_base_patch16_224_ce
from lib.models.ostrack.vit_tome import (
    vit_base_patch16_224_tome,
    vit_large_patch16_224_tome,
)
from lib.models.ostrack.vit_evit import (
    vit_base_patch16_224_evit,
    vit_large_patch16_224_evit,
)
from lib.models.ostrack.vit_dyvit import (
    vit_base_patch16_224_dyvit,
    vit_large_patch16_224_dyvit,
)
from lib.models.ostrack.vit_evovit import (
    vit_base_patch16_224_evovit,
    vit_large_patch16_224_evovit,
)
from lib.utils.box_ops import box_xyxy_to_cxcywh


class OSTrack(nn.Module):
    """This is the base class for OSTrack"""

    def __init__(
        self,
        transformer,
        box_head,
        aux_loss=False,
        head_type="CORNER",
        num_search=1,
        num_template=1,
        lens_per_search=None,
        lens_per_template=None,
    ):
        """Initializes the model.
        Parameters:
            transformer: torch module of the transformer architecture.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.backbone = transformer
        self.box_head = box_head

        self.aux_loss = aux_loss
        self.head_type = head_type
        if head_type == "CORNER" or head_type == "CENTER":
            self.feat_sz_s = int(box_head.feat_sz)
            self.feat_len_s = int(box_head.feat_sz**2)

        if self.aux_loss:
            self.box_head = _get_clones(self.box_head, 6)

        self.num_search = num_search
        self.num_template = num_template
        self.lens_per_search = lens_per_search
        self.lens_per_template = lens_per_template

    def forward(
        self,
        template_list=None,
        search_list=None,
        template_anno_list=None,
        ce_template_mask=None,
        ce_keep_rate=None,
        return_last_attn=False,
        epoch=None,
    ):  
        xz, aux_dict = self.backbone(
            template_list=template_list,
            search_list=search_list,
            template_anno_list=template_anno_list,
            ce_template_mask=ce_template_mask,
            ce_keep_rate=ce_keep_rate,
            return_last_attn=return_last_attn,
            epoch=epoch,
        )

        # Forward hea
        out = self.forward_head(xz, None)

        out.update(aux_dict)
        out["backbone_feature"] = xz
        return out

    def forward_head(self, xz, gt_score_map=None):
        # xz: [cls, x, z] or [x, z]
        # x: xz[:, -lens_x-lens_z:-lens_z]
        # z: xz[:, -lens_z:]
        lens_x = self.num_search * self.lens_per_search
        lens_z = self.num_template * self.lens_per_template
        feature = xz[:, -lens_x - lens_z : -lens_z]
        feature = (feature.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()
        bs, Nq, C, HW = feature.size()
        feature = feature.view(-1, C, self.feat_sz_s, self.feat_sz_s)

        if self.head_type == "CORNER":
            # run the corner head
            pred_box, score_map = self.box_head(feature, True)
            outputs_coord = box_xyxy_to_cxcywh(pred_box)
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {
                "pred_boxes": outputs_coord_new,
                "score_map": score_map,
            }
            return out

        elif self.head_type == "CENTER":
            # run the center head
            score_map_ctr, bbox, size_map, offset_map, score = self.box_head(
                feature, gt_score_map
            )
            # outputs_coord = box_xyxy_to_cxcywh(bbox)
            outputs_coord = bbox
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {
                "pred_boxes": outputs_coord_new,
                "score_map": score_map_ctr,
                "size_map": size_map,
                "offset_map": offset_map,
                "score": score,
            }
            return out
        else:
            raise NotImplementedError


def build_ostrack(cfg, training=True):
    current_dir = os.path.dirname(
        os.path.abspath(__file__)
    )  # This is your Project Root
    pretrained_path = os.path.join(current_dir, "../../../pretrained_networks")
    if (
        cfg.MODEL.PRETRAIN_FILE
        and ("OSTrack" not in cfg.MODEL.PRETRAIN_FILE)
        and training
    ):
        pretrained = os.path.join(pretrained_path, cfg.MODEL.PRETRAIN_FILE)
    else:
        pretrained = ""

    if cfg.MODEL.BACKBONE.TYPE == "vit_base_patch16_224":
        backbone = vit_base_patch16_224(
            pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE
        )
        hidden_dim = backbone.embed_dim
        patch_start_index = 1

    elif cfg.MODEL.BACKBONE.TYPE == "vit_base_patch16_224_ce":
        backbone = vit_base_patch16_224_ce(
            pretrained,
            drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
            ce_loc=cfg.MODEL.BACKBONE.CE_LOC,
            ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO,
            num_patches_template=int(cfg.DATA.TEMPLATE.SIZE / cfg.MODEL.BACKBONE.STRIDE)
            ** 2,
        )
        hidden_dim = backbone.embed_dim
        patch_start_index = 1

    elif cfg.MODEL.BACKBONE.TYPE == "vit_large_patch16_224_ce":
        backbone = vit_large_patch16_224_ce(
            pretrained,
            drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
            ce_loc=cfg.MODEL.BACKBONE.CE_LOC,
            ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO,
            num_patches_template=int(cfg.DATA.TEMPLATE.SIZE / cfg.MODEL.BACKBONE.STRIDE)
            ** 2,
        )

        hidden_dim = backbone.embed_dim
        patch_start_index = 1

    elif cfg.MODEL.BACKBONE.TYPE == "vit_base_patch16_224_tome":
        backbone = vit_base_patch16_224_tome(
            pretrained,
            drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
            r=cfg.MODEL.R,
        )

        hidden_dim = backbone.embed_dim
        patch_start_index = 1

    elif cfg.MODEL.BACKBONE.TYPE == "vit_large_patch16_224_tome":
        backbone = vit_large_patch16_224_tome(
            pretrained,
            drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
            r=cfg.MODEL.R,
        )

        hidden_dim = backbone.embed_dim
        patch_start_index = 1

    elif cfg.MODEL.BACKBONE.TYPE == "vit_base_patch16_224_evit":
        backbone = vit_base_patch16_224_evit(
            pretrained,
            drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
            evit_loc=cfg.MODEL.BACKBONE.CE_LOC,
            evit_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO,
        )

        hidden_dim = backbone.embed_dim
        patch_start_index = 1

    elif cfg.MODEL.BACKBONE.TYPE == "vit_large_patch16_224_evit":
        backbone = vit_large_patch16_224_evit(
            pretrained,
            drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
            evit_loc=cfg.MODEL.BACKBONE.CE_LOC,
            evit_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO,
        )

        hidden_dim = backbone.embed_dim
        patch_start_index = 1

    elif cfg.MODEL.BACKBONE.TYPE == "vit_base_patch16_224_dyvit":
        backbone = vit_base_patch16_224_dyvit(
            pretrained,
            drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
            pruning_loc=cfg.MODEL.BACKBONE.CE_LOC,
            keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO,
        )

        hidden_dim = backbone.embed_dim
        patch_start_index = 1

    elif cfg.MODEL.BACKBONE.TYPE == "vit_large_patch16_224_dyvit":
        backbone = vit_large_patch16_224_dyvit(
            pretrained,
            drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
            pruning_loc=cfg.MODEL.BACKBONE.CE_LOC,
            keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO,
        )

        hidden_dim = backbone.embed_dim
        patch_start_index = 1

    elif cfg.MODEL.BACKBONE.TYPE == "vit_base_patch16_224_evovit":
        backbone = vit_base_patch16_224_evovit(
            pretrained,
            drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
            ce_loc=cfg.MODEL.BACKBONE.CE_LOC,
            ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO,
            prune_ratio=cfg.MODEL.PRUNE_RATIO,
            trade_off=cfg.MODEL.TRADE_OFF,
        )

        hidden_dim = backbone.embed_dim
        patch_start_index = 1

    elif cfg.MODEL.BACKBONE.TYPE == "vit_large_patch16_224_evovit":
        backbone = vit_large_patch16_224_evovit(
            pretrained,
            drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
            ce_loc=cfg.MODEL.BACKBONE.CE_LOC,
            ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO,
            prune_ratio=cfg.MODEL.PRUNE_RATIO,
            trade_off=cfg.MODEL.TRADE_OFF,
        )

        hidden_dim = backbone.embed_dim
        patch_start_index = 1

    else:
        raise NotImplementedError

    backbone.finetune_track(cfg=cfg, patch_start_index=patch_start_index)

    box_head = build_box_head(cfg, hidden_dim)

    model = OSTrack(
        backbone,
        box_head,
        aux_loss=False,
        head_type=cfg.MODEL.HEAD.TYPE,
        num_search=cfg.DATA.SEARCH.NUMBER,
        num_template=cfg.DATA.TEMPLATE.NUMBER,
        lens_per_search=int(cfg.DATA.SEARCH.SIZE / cfg.MODEL.BACKBONE.STRIDE) ** 2,
        lens_per_template=int(cfg.DATA.TEMPLATE.SIZE / cfg.MODEL.BACKBONE.STRIDE) ** 2,
    )

    if "OSTrack" in cfg.MODEL.PRETRAIN_FILE and training:
        checkpoint = torch.load(cfg.MODEL.PRETRAIN_FILE, map_location="cpu")
        missing_keys, unexpected_keys = model.load_state_dict(
            checkpoint["net"], strict=False
        )
        print("Load pretrained model from: " + cfg.MODEL.PRETRAIN_FILE)

    return model
