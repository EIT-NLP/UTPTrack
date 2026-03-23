import math
import torch

def candidate_elimination(tokens: torch.Tensor, attn: torch.Tensor, lens_z: int, keep_ratio: float, global_index_x: torch.Tensor, ):

    lens_x = tokens.shape[1] - 1 - lens_z
    bs, hn, _, _ = attn.shape

    lens_keep = math.ceil(keep_ratio * lens_x)
    if lens_keep == lens_x:
        return tokens, global_index_x, None, None


    attn_z = attn[:, :, 1 + lens_x:, 1:1 + lens_x]
    attn_z = attn_z.mean(dim=2).mean(dim=1)  # B, H, L-z, L_x --> B, L_x

    sorted_attn, indices = torch.sort(attn_z, dim=1, descending=True)

    topk_attn, topk_idx = sorted_attn[:, :lens_keep], indices[:, :lens_keep]
    non_topk_attn, non_topk_idx = sorted_attn[:, lens_keep:], indices[:, lens_keep:]

    keep_index_x = global_index_x.gather(dim=1, index=topk_idx)
    removed_index_x = global_index_x.gather(dim=1, index=non_topk_idx)

    # separate tokens: [cls, x, z] (bs, num_tokens, C)
    tokens_cls = tokens[:, 0].unsqueeze(1)
    tokens_x = tokens[:, 1:1+lens_x]
    tokens_z = tokens[:, 1+lens_x:]

    # obtain the attentive and inattentive tokens
    B, L, C = tokens_x.shape
    attentive_tokens_x = tokens_x.gather(dim=1, index=topk_idx.unsqueeze(-1).expand(B, -1, C))
    removed_tokens_x = tokens_x.gather(dim=1, index=non_topk_idx.unsqueeze(-1).expand(B, -1, C))
    tokens_new = torch.cat([tokens_cls, attentive_tokens_x, tokens_z], dim=1)

    return tokens_new, keep_index_x, removed_index_x, removed_tokens_x