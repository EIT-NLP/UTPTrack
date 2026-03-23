"""
SUTrack Model
"""
import torch
import math
from torch import nn
import torch.nn.functional as F
from .encoder import build_encoder
from .clip import build_textencoder
from .decoder import build_decoder
from .task_decoder import build_task_decoder
from lib.utils.box_ops import box_xyxy_to_cxcywh
from lib.utils.pos_embed import get_sinusoid_encoding_table, get_2d_sincos_pos_embed


class SUTRACK(nn.Module):
    """ This is the base class for SUTrack """
    def __init__(self, text_encoder, encoder, decoder, task_decoder,
                 num_frames=1, num_template=1,
                 decoder_type="CENTER", task_feature_type="average"):
        """ Initializes the model. """
        super().__init__()
        self.encoder = encoder
        self.text_encoder = text_encoder
        self.decoder_type = decoder_type

        self.class_token = False if (encoder.body.cls_token is None) else True
        self.task_feature_type = task_feature_type

        self.num_patch_x = self.encoder.body.num_patches_search
        self.num_patch_z = self.encoder.body.num_patches_template
        self.fx_sz = int(math.sqrt(self.num_patch_x))
        self.fz_sz = int(math.sqrt(self.num_patch_z))

        self.task_decoder = task_decoder
        self.decoder = decoder

        self.num_frames = num_frames
        self.num_template = num_template
        encoder.num_frames = num_frames
        encoder.num_template = num_template


    def forward(self, text_data=None,
                template_list=None, search_list=None, template_anno_list=None,
                text_src=None, task_index=None,
                feature=None, mode="encoder",
                ce_template_mask=None,
                ce_keep_rate=None,
                ):
        if mode == "text":
            return self.forward_textencoder(text_data)
        elif mode == "encoder":
            x, aux_dict = self.forward_encoder(template_list, search_list, template_anno_list, text_src, task_index,
                                               ce_template_mask=ce_template_mask, ce_keep_rate=ce_keep_rate)
            return x
        elif mode == "decoder":
            return self.forward_decoder(feature), self.forward_task_decoder(feature)
        else:
            raise ValueError

    def forward_textencoder(self, text_data):
        # Forward the encoder
        text_src = self.text_encoder(text_data)
        return text_src

    def forward_encoder(self, template_list, search_list, template_anno_list, text_src, task_index,
                        ce_template_mask=None, ce_keep_rate=None):
        # Forward the encoder
        xz, aux_dict  = self.encoder(template_list, search_list, template_anno_list, text_src, task_index,
                                     ce_template_mask=ce_template_mask, ce_keep_rate=ce_keep_rate)
        return xz, aux_dict

    def forward_decoder(self, feature, gt_score_map=None):

        feature = feature[0]    # xz
        if self.class_token:
            feature = feature[:,1:self.num_patch_x * self.num_frames+1] # x
        else:
            feature = feature[:,0:self.num_patch_x * self.num_frames]   # x (B, HW, C)

        bs, HW, C = feature.size()
        if self.decoder_type in ['CORNER', 'CENTER']:
            feature = feature.permute((0, 2, 1)).contiguous()
            feature = feature.view(bs, C, self.fx_sz, self.fx_sz)
        if self.decoder_type == "CORNER":
            # run the corner head
            pred_box, score_map = self.decoder(feature, True)
            outputs_coord = box_xyxy_to_cxcywh(pred_box)
            outputs_coord_new = outputs_coord.view(bs, 1, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map}
            return out
        elif self.decoder_type == "CENTER":
            # run the center head
            score_map_ctr, bbox, size_map, offset_map = self.decoder(feature, gt_score_map)
            outputs_coord = bbox
            outputs_coord_new = outputs_coord.view(bs, 1, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map_ctr,
                   'size_map': size_map,
                   'offset_map': offset_map}
            return out
        elif self.decoder_type == "MLP":
            # run the mlp head
            score_map, bbox, offset_map = self.decoder(feature, gt_score_map)
            outputs_coord = bbox
            outputs_coord_new = outputs_coord.view(bs, 1, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map,
                   'offset_map': offset_map}
            return out
        else:
            raise NotImplementedError

    def forward_task_decoder(self, feature):
        feature = feature[0]
        if self.task_feature_type == 'class':
            feature = feature[:, 0:1]
        elif self.task_feature_type == 'text':
            feature = feature[:, -1:]
        elif self.task_feature_type == 'average':
            feature = feature.mean(1).unsqueeze(1)
        else:
            raise NotImplementedError('task_feature_type must be choosen from class, text, and average')
        feature = self.task_decoder(feature)
        return feature

def build_sutrack(cfg):
    encoder = build_encoder(cfg)    # Vision Encoder: Fast-iTPN
    if cfg.DATA.MULTI_MODAL_LANGUAGE:
        text_encoder = build_textencoder(cfg, encoder)  # Text Encoder: CLIP
    else:
        text_encoder = None
    decoder = build_decoder(cfg, encoder)   # Decoder: CenterPredictor
    task_decoder = build_task_decoder(cfg, encoder) # Task Decoder: MLP
    model = SUTRACK(
        text_encoder,
        encoder,
        decoder,
        task_decoder,
        num_frames = cfg.DATA.SEARCH.NUMBER,
        num_template = cfg.DATA.TEMPLATE.NUMBER,
        decoder_type=cfg.MODEL.DECODER.TYPE,
        task_feature_type=cfg.MODEL.TASK_DECODER.FEATURE_TYPE
    )

    return model
