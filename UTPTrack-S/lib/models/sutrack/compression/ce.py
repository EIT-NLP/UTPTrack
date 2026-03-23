import math
import torch
import torch.nn as nn
import torch.nn.functional as F



def candidate_elimination(tokens: torch.Tensor, attn: torch.Tensor, lens_z: int, keep_ratio: float,
                          global_index_x: torch.Tensor, box_mask_z: torch.Tensor, num_template: int):

    lens_x = tokens.shape[1] - 2 - lens_z
    bs, hn, _, _ = attn.shape

    lens_keep = math.ceil(keep_ratio * lens_x)
    if lens_keep == lens_x:
        # print("lens_keep == lens_x")
        return tokens, global_index_x, None
    # else:
    #     print(f"Prune: {lens_x - lens_keep}")

    # 目前支支持1或者2，sutrack的默认设置
    assert num_template == 1 or num_template == 2, "num_template must be 1 or 2"

    if num_template == 2:
        lens_1z = lens_z // 2
        box_mask_z = box_mask_z[:, :lens_1z]
        attn_z = attn[:, :, 1+lens_x:1+lens_x+lens_1z, 1:1 + lens_x]
    else:
        # token: [cls, x, z, text]
        # attn: (bs, num_head, k, v)
        attn_z = attn[:, :, 1 + lens_x:-1, 1:1 + lens_x]

    # # token: [cls, x, z, text]
    # # attn: (bs, num_head, k, v)
    # attn_z = attn[:, :, 1+lens_x:-1, 1:1+lens_x]

    if box_mask_z is not None:
        box_mask_z = box_mask_z.unsqueeze(1).unsqueeze(-1).expand(-1, attn_z.shape[1], -1, attn_z.shape[-1])
        attn_z = attn_z[box_mask_z]
        attn_z = attn_z.view(bs, hn, -1, lens_x)
        attn_z = attn_z.mean(dim=2).mean(dim=1)  # B, H, L-z, L_x --> B, L_x
    else:
        attn_z = attn_z.mean(dim=2).mean(dim=1)  # B, H, L-z, L_x --> B, L_x

    # use sort instead of topk, due to the speed issue
    # https://github.com/pytorch/pytorch/issues/22812
    sorted_attn, indices = torch.sort(attn_z, dim=1, descending=True)

    topk_attn, topk_idx = sorted_attn[:, :lens_keep], indices[:, :lens_keep]
    non_topk_attn, non_topk_idx = sorted_attn[:, lens_keep:], indices[:, lens_keep:]

    keep_index_x = global_index_x.gather(dim=1, index=topk_idx)
    removed_index_x = global_index_x.gather(dim=1, index=non_topk_idx)

    # separate tokens: [cls, x, z, text] (bs, num_tokens, C)
    tokens_cls = tokens[:, 0].unsqueeze(1)
    tokens_x = tokens[:, 1:1+lens_x]
    tokens_z = tokens[:, 1+lens_x:-1]
    tokens_text = tokens[:, -1].unsqueeze(1)

    # obtain the attentive and inattentive tokens
    B, L, C = tokens_x.shape
    attentive_tokens_x = tokens_x.gather(dim=1, index=topk_idx.unsqueeze(-1).expand(B, -1, C))
    tokens_new = torch.cat([tokens_cls, attentive_tokens_x, tokens_z, tokens_text], dim=1)

    return tokens_new, keep_index_x, removed_index_x


def generate_bbox_mask(bbox_mask, bbox):
    b, h, w = bbox_mask.shape
    for i in range(b):
        bbox_i = bbox[i].cpu().tolist()
        bbox_mask[i, int(bbox_i[1]):int(bbox_i[1] + bbox_i[3] - 1), int(bbox_i[0]):int(bbox_i[0] + bbox_i[2] - 1)] = 1
    return bbox_mask


def generate_mask_cond(cfg, bs, device):
    template_size = cfg.DATA.TEMPLATE.SIZE
    stride = cfg.MODEL.ENCODER.STRIDE
    template_feat_size = template_size // stride

    # CTR_POINT 只用这一种
    if template_feat_size == 8:
        index = slice(3, 4)
    elif template_feat_size == 12:
        index = slice(5, 6)
    elif template_feat_size == 7:
        index = slice(3, 4)
    elif template_feat_size == 14:
        index = slice(6, 7)
    else:
        raise NotImplementedError
    box_mask_z = torch.zeros([bs, template_feat_size, template_feat_size], device=device)
    box_mask_z[:, index, index] = 1
    box_mask_z = box_mask_z.flatten(1).to(torch.bool)

    return box_mask_z


def adjust_keep_rate(epoch, warmup_epochs, total_epochs, ITERS_PER_EPOCH, base_keep_rate=0.5, max_keep_rate=1, iters=-1):
    if epoch < warmup_epochs:
        return 1
    if epoch >= total_epochs:
        return base_keep_rate
    if iters == -1:
        iters = epoch * ITERS_PER_EPOCH
    total_iters = ITERS_PER_EPOCH * (total_epochs - warmup_epochs)
    iters = iters - ITERS_PER_EPOCH * warmup_epochs
    keep_rate = base_keep_rate + (max_keep_rate - base_keep_rate) * (math.cos(iters / total_iters * math.pi) + 1) * 0.5

    return keep_rate
