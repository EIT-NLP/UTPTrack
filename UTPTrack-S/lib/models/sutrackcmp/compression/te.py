import math
import torch
import torch.nn as nn
import torch.nn.functional as F


""" For CETE """
def template_elimination(tokens: torch.Tensor, attn: torch.Tensor, lens_z: int, keep_ratio: float,
                         global_index_z: torch.Tensor, box_mask_z: torch.Tensor, num_template: int, ori_lens_1z):

    lens_x = tokens.shape[1] - 2 - lens_z
    bs, hn, _, _ = attn.shape

    # 目前支支持1或者2，sutrack的默认设置
    assert num_template == 2, "num_template must be 2 for TE"

    lens_zd = lens_z - ori_lens_1z
    lens_keep = math.ceil(keep_ratio * lens_zd)
    if lens_keep == lens_zd:
        # print("lens_keep == lens_zd")
        return tokens, global_index_z, None
    # else:
    #     print(f"Prune Z: {lens_zd - lens_keep}")

    # # token: [cls, x, z, text]
    # # attn: (bs, num_head, k, v)
    attn_zs = attn[:, :, 1+lens_x:1+lens_x+ori_lens_1z, 1+lens_x+ori_lens_1z:-1]
    # attn_zs = attn[:, :, -1-lens_z:-1-lens_zd, -1-lens_zd:-1] # 等价

    # print(f"box: {box_mask_z.sum()}")
    if box_mask_z is not None:
        box_mask_z = box_mask_z.unsqueeze(1).unsqueeze(-1).expand(-1, attn_zs.shape[1], -1, attn_zs.shape[-1])
        attn_zs = attn_zs[box_mask_z]
        attn_zs = attn_zs.view(bs, hn, -1, lens_zd)
        attn_zs = attn_zs.mean(dim=2).mean(dim=1)  # B, H, L-z, L_x --> B, L_zd
    else:
        attn_zs = attn_zs.mean(dim=2).mean(dim=1)  # B, H, L-z, L_x --> B, L_zd

    # use sort instead of topk, due to the speed issue
    # https://github.com/pytorch/pytorch/issues/22812
    sorted_attn, indices = torch.sort(attn_zs, dim=1, descending=True)

    topk_attn, topk_idx = sorted_attn[:, :lens_keep], indices[:, :lens_keep]
    non_topk_attn, non_topk_idx = sorted_attn[:, lens_keep:], indices[:, lens_keep:]

    keep_index_zd = global_index_z.gather(dim=1, index=topk_idx+ori_lens_1z)
    removed_index_z = global_index_z.gather(dim=1, index=non_topk_idx+ori_lens_1z)
    # 静态模板全部保留
    keep_index_zs = torch.arange(0, ori_lens_1z, dtype=torch.int64).to(tokens.device)
    keep_index_zs = keep_index_zs.repeat(bs, 1)
    keep_index_z = torch.cat([keep_index_zs, keep_index_zd], dim=1)

    # separate tokens: [cls, x, z, text] (bs, num_tokens, C)
    tokens_cls = tokens[:, 0].unsqueeze(1)
    tokens_x = tokens[:, 1:1+lens_x]
    tokens_zs = tokens[:, 1+lens_x:1+lens_x+ori_lens_1z]
    tokens_zd = tokens[:, 1+lens_x+ori_lens_1z:-1]
    tokens_text = tokens[:, -1].unsqueeze(1)

    # obtain the attentive and inattentive tokens
    B, L, C = tokens_zd.shape

    attentive_tokens_z = tokens_zd.gather(dim=1, index=topk_idx.unsqueeze(-1).expand(B, -1, C))

    tokens_new = torch.cat([tokens_cls, tokens_x, tokens_zs, attentive_tokens_z, tokens_text], dim=1)

    return tokens_new, keep_index_z, removed_index_z