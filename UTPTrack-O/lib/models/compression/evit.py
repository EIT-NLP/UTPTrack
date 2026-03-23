import math
import torch


def evit_cem(
    tokens: torch.Tensor,
    attn: torch.Tensor,
    lens_z: int,
    keep_ratio: float,
    global_index_x: torch.Tensor,
):

    lens_x = tokens.shape[1] - 1 - lens_z
    bs, hn, _, _ = attn.shape

    lens_keep = math.ceil(keep_ratio * lens_x)
    if lens_keep == lens_x:
        return tokens, global_index_x, None, None

    # attention from z tokens to x tokens
    # tokens: [cls_token, x, z]
    attn_z = attn[:, :, 1 + lens_x :, 1 : 1 + lens_x]
    attn_z = attn_z.mean(dim=2).mean(dim=1)  # B, L_z

    # sort attention to get top-k and non-top-k
    sorted_attn, indices = torch.sort(attn_z, dim=1, descending=True)
    topk_attn, topk_idx = sorted_attn[:, : lens_keep + 1], indices[:, : lens_keep + 1]
    non_topk_attn, non_topk_idx = (
        sorted_attn[:, lens_keep + 1 :],
        indices[:, lens_keep + 1 :],
    )

    # gather global indices for topk and removed x tokens
    keep_index_x = global_index_x.gather(dim=1, index=topk_idx)
    removed_index_x = global_index_x.gather(dim=1, index=non_topk_idx)

    # separate tokens: [cls, x, z] (bs, num_tokens, C)
    tokens_cls = tokens[:, 0].unsqueeze(1)
    tokens_x = tokens[:, 1 : 1 + lens_x]
    tokens_z = tokens[:, 1 + lens_x :]

    # gather kept x tokens
    B, L, C = tokens_x.shape
    attentive_tokens_x = tokens_x.gather(
        dim=1, index=topk_idx.unsqueeze(-1).expand(B, -1, C)
    )

    # compute extra_token by attention-weighted sum of removed tokens
    non_topk_attn = attn_z.gather(dim=1, index=non_topk_idx)
    non_topk_tokens = tokens_x.gather(
        dim=1, index=non_topk_idx.unsqueeze(-1).expand(B, -1, C)
    )
    extra_token = torch.sum(
        non_topk_tokens * non_topk_attn.unsqueeze(-1), dim=1, keepdim=True
    )

    tokens_topk = attentive_tokens_x[:, :lens_keep].clone()
    tokens_topk_extra = attentive_tokens_x[:, lens_keep:].clone()
    tokens_topk_extra[:, 0, :] = extra_token.squeeze(1)

    attentive_tokens_x = torch.cat([tokens_topk, tokens_topk_extra], dim=1)

    # concatenate new token sequence: cls + topk x + extra + z
    tokens_new = torch.cat([tokens_cls, attentive_tokens_x, tokens_z], dim=1)

    extra_index_x = keep_index_x[:, -1:].clone()

    # extra_token_index = (global_index_x.max(dim=1, keepdim=True)[0] + 1).long()
    # keep_index_x = torch.cat([keep_index_x, extra_token_index], dim=1)

    return tokens_new, keep_index_x, removed_index_x, extra_index_x
