"""
Encoder modules: we use ITPN for the encoder.
"""

from torch import nn
from lib.utils.misc import is_main_process
from lib.models.sutrackcmp import fastitpn as fastitpn_module
from lib.models.sutrackcmp import blank_fastitpn as blank_fastitpn_module
from lib.models.sutrackcmp import ce_fastitpn as ce_fastitpn_module
from lib.models.sutrackcmp import cete_fastitpn as cete_fastitpn_module
from lib.models.sutrackcmp import ceate_fastitpn as ceate_fastitpn_module
from lib.models.sutrackcmp import ceatetta_fastitpn as ceatetta_fastitpn_module
from lib.models.sutrackcmp import ceatettamma_fastitpn as ceatettamma_fastitpn_module


class EncoderBase(nn.Module):
    def __init__(self, encoder: nn.Module, train_encoder: bool, open_layers: list, num_channels: int):
        super().__init__()
        open_blocks = open_layers[2:]
        open_items = open_layers[0:2]
        for name, parameter in encoder.named_parameters():

            if not train_encoder:
                freeze = True
                for open_block in open_blocks:
                    if open_block in name:
                        freeze = False
                if name in open_items:
                    freeze = False
                if freeze == True:
                    parameter.requires_grad_(False)  # here should allow users to specify which layers to freeze !

        self.body = encoder
        self.num_channels = num_channels

    def forward(self, template_list, search_list, template_anno_list, text_src, task_index,
                ce_template_mask=None,
                ce_keep_rate=None,
                dte_keep_rate=None,
                ste_keep_rate=None,
                ):
        xs, aux_dict = self.body(template_list, search_list, template_anno_list, text_src, task_index,
                                 ce_template_mask=ce_template_mask,
                                 ce_keep_rate=ce_keep_rate,
                                 dte_keep_rate=dte_keep_rate,
                                 ste_keep_rate=ste_keep_rate)
        return xs, aux_dict


class Encoder(EncoderBase):
    """ViT encoder."""
    def __init__(self, name: str,
                 train_encoder: bool,
                 search_size: int,
                 template_size: int,
                 open_layers: list,
                 cfg=None):
        if "ce_fastitpn" in name.lower():
            encoder = getattr(ce_fastitpn_module, name)(
                pretrained=is_main_process(),
                search_size=search_size,
                template_size=template_size,
                drop_rate=0.0,
                drop_path_rate=0.1,
                attn_drop_rate=0.0,
                init_values=0.1,
                drop_block_rate=None,
                use_mean_pooling=True,
                grad_ckpt=False,
                cls_token=cfg.MODEL.ENCODER.CLASS_TOKEN,
                pos_type=cfg.MODEL.ENCODER.POS_TYPE,
                token_type_indicate=cfg.MODEL.ENCODER.TOKEN_TYPE_INDICATE,
                pretrain_type = cfg.MODEL.ENCODER.PRETRAIN_TYPE,
                patchembed_init = cfg.MODEL.ENCODER.PATCHEMBED_INIT,
                ce_loc=cfg.MODEL.ENCODER.CE_LOC,
                ce_keep_ratio = cfg.MODEL.ENCODER.CE_KEEP_RATIO,
            )
            if "itpnb" in name:
                num_channels = 512
            elif "itpnl" in name:
                num_channels = 768
            elif "itpnt" in name:
                num_channels = 384
            elif "itpns" in name:
                num_channels = 384
            else:
                num_channels = 512
        elif "cete_fastitpn" in name.lower():
            encoder = getattr(cete_fastitpn_module, name)(
                pretrained=is_main_process(),
                search_size=search_size,
                template_size=template_size,
                drop_rate=0.0,
                drop_path_rate=0.1,
                attn_drop_rate=0.0,
                init_values=0.1,
                drop_block_rate=None,
                use_mean_pooling=True,
                grad_ckpt=False,
                cls_token=cfg.MODEL.ENCODER.CLASS_TOKEN,
                pos_type=cfg.MODEL.ENCODER.POS_TYPE,
                token_type_indicate=cfg.MODEL.ENCODER.TOKEN_TYPE_INDICATE,
                pretrain_type = cfg.MODEL.ENCODER.PRETRAIN_TYPE,
                patchembed_init = cfg.MODEL.ENCODER.PATCHEMBED_INIT,
                ce_loc=cfg.MODEL.ENCODER.CE_LOC,
                ce_keep_ratio = cfg.MODEL.ENCODER.CE_KEEP_RATIO,
                dte_loc=cfg.MODEL.ENCODER.DTE_LOC,
                dte_keep_ratio = cfg.MODEL.ENCODER.DTE_KEEP_RATIO,
            )
            if "itpnb" in name:
                num_channels = 512
            elif "itpnl" in name:
                num_channels = 768
            elif "itpnt" in name:
                num_channels = 384
            elif "itpns" in name:
                num_channels = 384
            else:
                num_channels = 512
        elif "ceate_fastitpn" in name.lower():
            encoder = getattr(ceate_fastitpn_module, name)(
                pretrained=is_main_process(),
                search_size=search_size,
                template_size=template_size,
                drop_rate=0.0,
                drop_path_rate=0.1,
                attn_drop_rate=0.0,
                init_values=0.1,
                drop_block_rate=None,
                use_mean_pooling=True,
                grad_ckpt=False,
                cls_token=cfg.MODEL.ENCODER.CLASS_TOKEN,
                pos_type=cfg.MODEL.ENCODER.POS_TYPE,
                token_type_indicate=cfg.MODEL.ENCODER.TOKEN_TYPE_INDICATE,
                pretrain_type=cfg.MODEL.ENCODER.PRETRAIN_TYPE,
                patchembed_init=cfg.MODEL.ENCODER.PATCHEMBED_INIT,
                ce_loc=cfg.MODEL.ENCODER.CE_LOC,
                ce_keep_ratio=cfg.MODEL.ENCODER.CE_KEEP_RATIO,
                dte_loc=cfg.MODEL.ENCODER.DTE_LOC,
                dte_keep_ratio=cfg.MODEL.ENCODER.DTE_KEEP_RATIO,
                ste_loc=cfg.MODEL.ENCODER.STE_LOC,
                ste_keep_ratio=cfg.MODEL.ENCODER.STE_KEEP_RATIO,
            )
            if "itpnb" in name:
                num_channels = 512
            elif "itpnl" in name:
                num_channels = 768
            elif "itpnt" in name:
                num_channels = 384
            elif "itpns" in name:
                num_channels = 384
            else:
                num_channels = 512
        elif "ceatetta_fastitpn" in name.lower():
            encoder = getattr(ceatetta_fastitpn_module, name)(
                pretrained=is_main_process(),
                search_size=search_size,
                template_size=template_size,
                drop_rate=0.0,
                drop_path_rate=0.1,
                attn_drop_rate=0.0,
                init_values=0.1,
                drop_block_rate=None,
                use_mean_pooling=True,
                grad_ckpt=False,
                cls_token=cfg.MODEL.ENCODER.CLASS_TOKEN,
                pos_type=cfg.MODEL.ENCODER.POS_TYPE,
                token_type_indicate=cfg.MODEL.ENCODER.TOKEN_TYPE_INDICATE,
                pretrain_type=cfg.MODEL.ENCODER.PRETRAIN_TYPE,
                patchembed_init=cfg.MODEL.ENCODER.PATCHEMBED_INIT,
                ce_loc=cfg.MODEL.ENCODER.CE_LOC,
                ce_keep_ratio=cfg.MODEL.ENCODER.CE_KEEP_RATIO,
                dte_loc=cfg.MODEL.ENCODER.DTE_LOC,
                dte_keep_ratio=cfg.MODEL.ENCODER.DTE_KEEP_RATIO,
                ste_loc=cfg.MODEL.ENCODER.STE_LOC,
                ste_keep_ratio=cfg.MODEL.ENCODER.STE_KEEP_RATIO,
                token_type_aware_pruing=cfg.MODEL.ENCODER.TOKEN_TYPE_AWARE_PRUNING,
            )
            if "itpnb" in name:
                num_channels = 512
            elif "itpnl" in name:
                num_channels = 768
            elif "itpnt" in name:
                num_channels = 384
            elif "itpns" in name:
                num_channels = 384
            else:
                num_channels = 512
        elif "ceatettamma_fastitpn" in name.lower():
            encoder = getattr(ceatettamma_fastitpn_module, name)(
                pretrained=is_main_process(),
                search_size=search_size,
                template_size=template_size,
                drop_rate=0.0,
                drop_path_rate=0.1,
                attn_drop_rate=0.0,
                init_values=0.1,
                drop_block_rate=None,
                use_mean_pooling=True,
                grad_ckpt=False,
                cls_token=cfg.MODEL.ENCODER.CLASS_TOKEN,
                pos_type=cfg.MODEL.ENCODER.POS_TYPE,
                token_type_indicate=cfg.MODEL.ENCODER.TOKEN_TYPE_INDICATE,
                pretrain_type=cfg.MODEL.ENCODER.PRETRAIN_TYPE,
                patchembed_init=cfg.MODEL.ENCODER.PATCHEMBED_INIT,
                ce_loc=cfg.MODEL.ENCODER.CE_LOC,
                ce_keep_ratio=cfg.MODEL.ENCODER.CE_KEEP_RATIO,
                dte_loc=cfg.MODEL.ENCODER.DTE_LOC,
                dte_keep_ratio=cfg.MODEL.ENCODER.DTE_KEEP_RATIO,
                ste_loc=cfg.MODEL.ENCODER.STE_LOC,
                ste_keep_ratio=cfg.MODEL.ENCODER.STE_KEEP_RATIO,
                token_type_aware_pruing=cfg.MODEL.ENCODER.TOKEN_TYPE_AWARE_PRUNING,
                multimodal_aware_pruning=cfg.MODEL.ENCODER.MULTIMODAL_AWARE_PRUNING,
                multimodal_aware_pruning_target=cfg.MODEL.ENCODER.MULTIMODAL_AWARE_PRUNING_TARGET,
            )
            if "itpnb" in name:
                num_channels = 512
            elif "itpnl" in name:
                num_channels = 768
            elif "itpnt" in name:
                num_channels = 384
            elif "itpns" in name:
                num_channels = 384
            else:
                num_channels = 512
        elif "blank_fastitpn" in name.lower():
            encoder = getattr(blank_fastitpn_module, name)(
                pretrained=is_main_process(),
                search_size=search_size,
                template_size=template_size,
                drop_rate=0.0,
                drop_path_rate=0.1,
                attn_drop_rate=0.0,
                init_values=0.1,
                drop_block_rate=None,
                use_mean_pooling=True,
                grad_ckpt=False,
                cls_token=cfg.MODEL.ENCODER.CLASS_TOKEN,
                pos_type=cfg.MODEL.ENCODER.POS_TYPE,
                token_type_indicate=cfg.MODEL.ENCODER.TOKEN_TYPE_INDICATE,
                pretrain_type = cfg.MODEL.ENCODER.PRETRAIN_TYPE,
                patchembed_init = cfg.MODEL.ENCODER.PATCHEMBED_INIT
            )
            if "itpnb" in name:
                num_channels = 512
            elif "itpnl" in name:
                num_channels = 768
            elif "itpnt" in name:
                num_channels = 384
            elif "itpns" in name:
                num_channels = 384
            else:
                num_channels = 512
        elif "fastitpn" in name.lower():
            encoder = getattr(fastitpn_module, name)(
                pretrained=is_main_process(),
                search_size=search_size,
                template_size=template_size,
                drop_rate=0.0,
                drop_path_rate=0.1,
                attn_drop_rate=0.0,
                init_values=0.1,
                drop_block_rate=None,
                use_mean_pooling=True,
                grad_ckpt=False,
                cls_token=cfg.MODEL.ENCODER.CLASS_TOKEN,
                pos_type=cfg.MODEL.ENCODER.POS_TYPE,
                token_type_indicate=cfg.MODEL.ENCODER.TOKEN_TYPE_INDICATE,
                pretrain_type = cfg.MODEL.ENCODER.PRETRAIN_TYPE,
                patchembed_init = cfg.MODEL.ENCODER.PATCHEMBED_INIT
            )
            if "itpnb" in name:
                num_channels = 512
            elif "itpnl" in name:
                num_channels = 768
            elif "itpnt" in name:
                num_channels = 384
            elif "itpns" in name:
                num_channels = 384
            else:
                num_channels = 512
        else:
            raise ValueError()
        super().__init__(encoder, train_encoder, open_layers, num_channels)



def build_encoder(cfg):
    train_encoder = (cfg.TRAIN.ENCODER_MULTIPLIER > 0) and (cfg.TRAIN.FREEZE_ENCODER == False)
    encoder = Encoder(cfg.MODEL.ENCODER.TYPE, train_encoder,
                      cfg.DATA.SEARCH.SIZE,
                      cfg.DATA.TEMPLATE.SIZE,
                      cfg.TRAIN.ENCODER_OPEN, cfg)
    encoder.body.num_frames = cfg.DATA.SEARCH.NUMBER
    encoder.body.num_template = cfg.DATA.TEMPLATE.NUMBER
    return encoder
