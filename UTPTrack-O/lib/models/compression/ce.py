import torch
import math


""" For CE, CETE"""
def candidate_elimination(tokens: torch.Tensor, attn: torch.Tensor, lens_z: int, keep_ratio: float,
                          global_index_x: torch.Tensor, box_mask_z: torch.Tensor, num_template: int, ori_lens_1z):
    """
    Eliminate potential background candidates for computation reduction and noise cancellation.
    Args:
        attn (torch.Tensor): [B, num_heads, L_t + L_s, L_t + L_s], attention weights
        tokens (torch.Tensor):  [B, L_t + L_s, C], template and search region tokens
        lens_t (int): length of template
        keep_ratio (float): keep ratio of search region tokens (candidates)
        global_index (torch.Tensor): global index of search region tokens
        box_mask_z (torch.Tensor): template mask used to accumulate attention weights

    Returns:
        tokens_new (torch.Tensor): tokens after candidate elimination
        keep_index (torch.Tensor): indices of kept search region tokens
        removed_index (torch.Tensor): indices of removed search region tokens
    """
    lens_x = tokens.shape[1] - lens_z
    bs, hn, _, _ = attn.shape

    lens_keep = math.ceil(keep_ratio * lens_x)
    if lens_keep == lens_x:
        # print("lens_keep == lens_x")
        return tokens, global_index_x, None
    # else:
    #     print(f"Prune X: {lens_x - lens_keep}")

    assert num_template == 1 or num_template == 2, "num_template must be 1 or 2 for CE"

    # token: [x, z]
    # attn: (bs, num_head, k, v)
    if num_template == 2:
        attn_z = attn[:, :, lens_x:lens_x+ori_lens_1z, :lens_x]
    else:
        attn_z = attn[:, :, lens_x:-1, :lens_x]

    if box_mask_z is not None:
        box_mask_z = box_mask_z.unsqueeze(1).unsqueeze(-1).expand(-1, attn_z.shape[1], -1, attn_z.shape[-1])
        # attn_t = attn_t[:, :, box_mask_z, :]
        attn_z = attn_z[box_mask_z]
        attn_z = attn_z.view(bs, hn, -1, lens_x)
        attn_z = attn_z.mean(dim=2).mean(dim=1)  # B, H, L-T, L_s --> B, L_s
    else:
        attn_z = attn_z.mean(dim=2).mean(dim=1)  # B, H, L-T, L_s --> B, L_s

    # use sort instead of topk, due to the speed issue
    # https://github.com/pytorch/pytorch/issues/22812
    sorted_attn, indices = torch.sort(attn_z, dim=1, descending=True)

    topk_attn, topk_idx = sorted_attn[:, :lens_keep], indices[:, :lens_keep]
    non_topk_attn, non_topk_idx = sorted_attn[:, lens_keep:], indices[:, lens_keep:]

    keep_index_x = global_index_x.gather(dim=1, index=topk_idx)
    removed_index_x = global_index_x.gather(dim=1, index=non_topk_idx)

    # separate template and search tokens
    # [x, z]
    tokens_z = tokens[:, lens_x:]
    tokens_x = tokens[:, :lens_x]

    # obtain the attentive and inattentive tokens
    B, L, C = tokens_x.shape
    attentive_tokens_x = tokens_x.gather(dim=1, index=topk_idx.unsqueeze(-1).expand(B, -1, C))

    tokens_new = torch.cat([attentive_tokens_x, tokens_z], dim=1)

    return tokens_new, keep_index_x, removed_index_x

"""For CEATE"""
def candidate_elimination_by_ate(tokens: torch.Tensor, attn: torch.Tensor,
                                 lens_sz: int, lens_dz: int,
                                 keep_ratio: float,
                                 global_index_x: torch.Tensor,
                                 box_mask_z: torch.Tensor,
                                 num_template: int,
                                 ):
    lens_x = tokens.shape[1] - lens_sz - lens_dz
    bs, hn, _, _ = attn.shape

    lens_keep = math.ceil(keep_ratio * lens_x)
    if lens_keep == lens_x:
        # print("lens_keep == lens_x")
        return tokens, global_index_x, None
    # else:
    #     print(f"Prune X: {lens_x - lens_keep}")

    assert num_template == 2, "num_template must be 2 for CE_w/_ATE"

    # token: [x, z]
    # attn: (bs, num_head, k, v)
    attn_z = attn[:, :, lens_x:lens_x+lens_sz, :lens_x] # 这里越界了

    # print(f"box: {box_mask_z.sum()}")
    if box_mask_z is not None:
        box_mask_z = box_mask_z.unsqueeze(1).unsqueeze(-1).expand(-1, attn_z.shape[1], -1, attn_z.shape[-1])
        attn_z = attn_z[box_mask_z]
        attn_z = attn_z.view(bs, hn, -1, lens_x)
        attn_z = attn_z.mean(dim=2).mean(dim=1)  # B, H, L-T, L_s --> B, L_s
    else:
        attn_z = attn_z.mean(dim=2).mean(dim=1)  # B, H, L-T, L_s --> B, L_s

    # use sort instead of topk, due to the speed issue
    # https://github.com/pytorch/pytorch/issues/22812
    sorted_attn, indices = torch.sort(attn_z, dim=1, descending=True)

    topk_attn, topk_idx = sorted_attn[:, :lens_keep], indices[:, :lens_keep]
    non_topk_attn, non_topk_idx = sorted_attn[:, lens_keep:], indices[:, lens_keep:]

    keep_index_x = global_index_x.gather(dim=1, index=topk_idx)
    removed_index_x = global_index_x.gather(dim=1, index=non_topk_idx)

    # separate template and search tokens
    # [x, z]
    tokens_z = tokens[:, lens_x:]
    tokens_x = tokens[:, :lens_x]

    # obtain the attentive and inattentive tokens
    B, L, C = tokens_x.shape
    attentive_tokens_x = tokens_x.gather(dim=1, index=topk_idx.unsqueeze(-1).expand(B, -1, C))

    tokens_new = torch.cat([attentive_tokens_x, tokens_z], dim=1)

    return tokens_new, keep_index_x, removed_index_x
