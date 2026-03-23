import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def dynamic_template_elimination(tokens: torch.Tensor, attn: torch.Tensor,
                                 lens_sz: int, lens_dz: int,
                                 keep_ratio: float,
                                 global_index_dz: torch.Tensor,
                                 box_mask_z: torch.Tensor,
                                 num_template: int,
                                 ):

    lens_x = tokens.shape[1] - 2 - lens_sz - lens_dz
    bs, hn, _, _ = attn.shape

    # 目前支支持1或者2，sutrack的默认设置
    assert num_template == 2, "num_template must be 2 for TE"

    lens_keep = math.ceil(keep_ratio * lens_dz)
    if lens_keep == lens_dz:
        # print("lens_keep == lens_dz")
        return tokens, global_index_dz, None
    # else:
    #     print(f"Prune DZ: {lens_dz - lens_keep}")

    # token: [cls, x, z, text]
    # attn: (bs, num_head, k, v)
    attn_z = attn[:, :, 1+lens_x:1+lens_x+lens_sz, 1+lens_x+lens_sz:-1]
    # attn_z = attn[:, :, -1-lens_dz-lens_sz:-1-lens_dz, -1-lens_dz:-1] # 等价

    # print(f"dynamic box: {box_mask_z.sum()}")
    if box_mask_z is not None:
        box_mask_z = box_mask_z.unsqueeze(1).unsqueeze(-1).expand(-1, attn_z.shape[1], -1, attn_z.shape[-1])
        attn_z = attn_z[box_mask_z]
        attn_z = attn_z.view(bs, hn, -1, lens_dz)
        attn_z = attn_z.mean(dim=2).mean(dim=1)  # B, H, L-z, L_x --> B, L_zd
    else:
        attn_z = attn_z.mean(dim=2).mean(dim=1)  # B, H, L-z, L_x --> B, L_zd

    # use sort instead of topk, due to the speed issue
    # https://github.com/pytorch/pytorch/issues/22812
    sorted_attn, indices = torch.sort(attn_z, dim=1, descending=True)

    topk_attn, topk_idx = sorted_attn[:, :lens_keep], indices[:, :lens_keep]
    non_topk_attn, non_topk_idx = sorted_attn[:, lens_keep:], indices[:, lens_keep:]

    keep_index_dz = global_index_dz.gather(dim=1, index=topk_idx)
    removed_index_dz = global_index_dz.gather(dim=1, index=non_topk_idx)

    # separate tokens: [cls, x, z, text] (bs, num_tokens, C)
    tokens_cls = tokens[:, 0].unsqueeze(1)
    tokens_x = tokens[:, 1:1+lens_x]
    tokens_sz = tokens[:, 1+lens_x:1+lens_x+lens_sz]
    tokens_dz = tokens[:, 1+lens_x+lens_sz:-1]
    tokens_text = tokens[:, -1].unsqueeze(1)

    # obtain the attentive and inattentive tokens
    B, L, C = tokens_dz.shape

    attentive_tokens_dz = tokens_dz.gather(dim=1, index=topk_idx.unsqueeze(-1).expand(B, -1, C))

    tokens_new = torch.cat([tokens_cls, tokens_x, tokens_sz, attentive_tokens_dz, tokens_text], dim=1)

    return tokens_new, keep_index_dz, removed_index_dz

def static_template_elimination(tokens: torch.Tensor, attn: torch.Tensor,
                                lens_sz: int, lens_dz: int,
                                keep_ratio: float,
                                global_index_sz: torch.Tensor,
                                box_mask_z: torch.Tensor,
                                num_template: int,
                                indicate_mask_sz=None,
                                token_type_aware_pruing=None,
                                ):

    lens_x = tokens.shape[1] - 2 - lens_sz - lens_dz
    bs, hn, _, _ = attn.shape

    # 目前支支持1或者2，sutrack的默认设置
    assert num_template == 2, "num_template must be 2 for TE"

    lens_keep = math.ceil(keep_ratio * lens_sz)
    if lens_keep == lens_sz:
        # print("lens_keep == lens_sz")
        return tokens, global_index_sz, None, box_mask_z, indicate_mask_sz
    # else:
    #     print(f"Prune SZ: {lens_sz - lens_keep}")

    # token: [cls, x, z, text]
    # attn: (bs, num_head, k, v)
    attn_z = attn[:, :, 1+lens_x:1+lens_x+lens_sz, 1+lens_x:1+lens_x+lens_sz]
    # attn_z = attn[:, :, -1-lens_dz-lens_sz:-1-lens_dz, -1-lens_dz-lens_sz:-1-lens_dz]   # 等价

    # print(f"static box: {box_mask_z.sum()}")
    if box_mask_z is not None:
        batch_indices, true_positions = torch.where(box_mask_z)  # 找到box_mask_z中的true的位置
        box_mask_z = box_mask_z.unsqueeze(1).unsqueeze(-1).expand(-1, attn_z.shape[1], -1, attn_z.shape[-1])
        attn_z = attn_z[box_mask_z]
        attn_z = attn_z.view(bs, hn, -1, lens_sz)
        attn_z = attn_z.mean(dim=2).mean(dim=1)  # B, H, L-z, L_x --> B, L_zd
        bonus_mask = torch.zeros_like(attn_z)
        bonus_mask[batch_indices, true_positions] = 1.0 # 给静态模板中心点的attn加上一个大值，防止被删掉
        attn_z = attn_z + bonus_mask
    else:
        attn_z = attn_z.mean(dim=2).mean(dim=1)  # B, H, L-z, L_x --> B, L_zd

    if indicate_mask_sz is not None:
        if token_type_aware_pruing in ['FULL_FOREGROUND', 'ALL_FOREGROUND']:
            batch_indices_sz, true_positions_sz = torch.where(indicate_mask_sz)
            indicate_bonus_mask = torch.zeros_like(attn_z)
            indicate_bonus_mask[batch_indices_sz, true_positions_sz] = 1.0
        elif token_type_aware_pruing in ['SOFT_ALL_FOREGROUND']:
            indicate_bonus_mask = indicate_mask_sz
        attn_z = attn_z + indicate_bonus_mask

    # use sort instead of topk, due to the speed issue
    # https://github.com/pytorch/pytorch/issues/22812
    sorted_attn, indices = torch.sort(attn_z, dim=1, descending=True)

    topk_attn, topk_idx = sorted_attn[:, :lens_keep], indices[:, :lens_keep]
    non_topk_attn, non_topk_idx = sorted_attn[:, lens_keep:], indices[:, lens_keep:]

    keep_index_sz = global_index_sz.gather(dim=1, index=topk_idx)
    removed_index_sz = global_index_sz.gather(dim=1, index=non_topk_idx)

    # separate tokens: [cls, x, z, text] (bs, num_tokens, C)
    tokens_cls = tokens[:, 0].unsqueeze(1)
    tokens_x = tokens[:, 1:1+lens_x]
    tokens_sz = tokens[:, 1+lens_x:1+lens_x+lens_sz]
    tokens_dz = tokens[:, 1+lens_x+lens_sz:-1]
    tokens_text = tokens[:, -1].unsqueeze(1)

    # obtain the attentive and inattentive tokens
    B, L, C = tokens_sz.shape

    attentive_tokens_sz = tokens_sz.gather(dim=1, index=topk_idx.unsqueeze(-1).expand(B, -1, C))

    if box_mask_z is not None:
        # box_mask_idx_sorted, _ =  torch.sort(topk_idx, dim=1)   # TODO: 7.31修改
        box_mask_z = box_mask_z[:, 0, :, 0]
        # box_mask_z_new = box_mask_z.gather(dim=1, index=box_mask_idx_sorted)
        box_mask_z_new = box_mask_z.gather(dim=1, index=topk_idx)
    else:
        box_mask_z_new = None

    if indicate_mask_sz is not None:
        # indicate_mask_idx_sorted, _ = torch.sort(topk_idx, dim=1)   # TODO: 7.31修改
        # indicate_mask_sz_new = indicate_mask_sz.gather(dim=1, index=indicate_mask_idx_sorted)
        indicate_mask_sz_new = indicate_mask_sz.gather(dim=1, index=topk_idx)
    else:
        indicate_mask_sz_new = None
    tokens_new = torch.cat([tokens_cls, tokens_x, attentive_tokens_sz, tokens_dz, tokens_text], dim=1)

    return tokens_new, keep_index_sz, removed_index_sz, box_mask_z_new, indicate_mask_sz_new

def dynamic_template_elimination_mma(tokens: torch.Tensor, attn: torch.Tensor,
                                 lens_sz: int, lens_dz: int,
                                 keep_ratio: float,
                                 global_index_dz: torch.Tensor,
                                 box_mask_z: torch.Tensor,
                                 num_template: int,
                                 task_index=None,
                                 mmaware_pruning_target = None,
                                 ):

    lens_x = tokens.shape[1] - 2 - lens_sz - lens_dz
    bs, hn, _, _ = attn.shape

    # 目前支支持1或者2，sutrack的默认设置
    assert num_template == 2, "num_template must be 2 for TE"

    lens_keep = math.ceil(keep_ratio * lens_dz)
    if lens_keep == lens_dz:
        # print("lens_keep == lens_dz")
        return tokens, global_index_dz, None
    # else:
    #     print(f"Prune DZ: {lens_dz - lens_keep}")

    # # # token: [cls, x, z, text]
    # # # attn: (bs, num_head, k, v)
    # # attn_z = attn[:, :, 1+lens_x:-1, 1:1+lens_x]
    # # attn_z = attn[:, :, 1+lens_x:1+lens_x+lens_sz, 1+lens_x+lens_sz:-1]   # 如果ce和te在同样位置，lens_x会错
    # attn_z = attn[:, :, -1-lens_dz-lens_sz:-1-lens_dz, -1-lens_dz:-1]       # 修正
    if task_index is not None and (task_index == 1).any() and 'D' in mmaware_pruning_target:  # 根据task_index动态选择注意力提取范围
        attn_template = attn[:, :, -1-lens_dz-lens_sz:-1-lens_dz, -1-lens_dz:-1]
        attn_text = attn[:, :, -1:, -1-lens_dz:-1]

        # 为不需要文本注意力的样本，将文本注意力置零
        text_weight = (task_index == 1).float().view(bs, 1, 1, 1)
        attn_text = attn_text * text_weight  # 只有task_index==1的样本保留文本注意力

        attn_z = torch.cat([attn_template, attn_text], dim=2)   # # (bs, H, lens_sz+1, lens_x)

        # 相应地扩展box_mask_z
        if box_mask_z is not None:  # (bs, lens_sz): True or False
            # 文本token总是有效的，所以创建全True的掩码
            text_mask = torch.ones(bs, 1, device=box_mask_z.device, dtype=box_mask_z.dtype)  # [bs, 1]
            box_mask_z = torch.cat([box_mask_z, text_mask], dim=1)  # [bs, lens_sz+1]
    else:
        attn_z = attn[:, :, -1-lens_dz-lens_sz:-1-lens_dz, -1-lens_dz:-1]

    # print(f"box: {box_mask_z.sum()}")
    if box_mask_z is not None:
        box_mask_z = box_mask_z.unsqueeze(1).unsqueeze(-1).expand(-1, attn_z.shape[1], -1, attn_z.shape[-1])
        attn_z = attn_z[box_mask_z]
        attn_z = attn_z.view(bs, hn, -1, lens_dz)
        attn_z = attn_z.mean(dim=2).mean(dim=1)  # B, H, L-z, L_x --> B, L_zd
    else:
        attn_z = attn_z.mean(dim=2).mean(dim=1)  # B, H, L-z, L_x --> B, L_zd

    # use sort instead of topk, due to the speed issue
    # https://github.com/pytorch/pytorch/issues/22812
    sorted_attn, indices = torch.sort(attn_z, dim=1, descending=True)

    topk_attn, topk_idx = sorted_attn[:, :lens_keep], indices[:, :lens_keep]
    non_topk_attn, non_topk_idx = sorted_attn[:, lens_keep:], indices[:, lens_keep:]

    keep_index_dz = global_index_dz.gather(dim=1, index=topk_idx)   # 这个似乎有问题
    removed_index_dz = global_index_dz.gather(dim=1, index=non_topk_idx)


    # separate tokens: [cls, x, z, text] (bs, num_tokens, C)
    tokens_cls = tokens[:, 0].unsqueeze(1)
    tokens_x = tokens[:, 1:1+lens_x]
    tokens_sz = tokens[:, 1+lens_x:1+lens_x+lens_sz]
    tokens_dz = tokens[:, 1+lens_x+lens_sz:-1]
    tokens_text = tokens[:, -1].unsqueeze(1)

    # obtain the attentive and inattentive tokens
    B, L, C = tokens_dz.shape

    attentive_tokens_dz = tokens_dz.gather(dim=1, index=topk_idx.unsqueeze(-1).expand(B, -1, C))

    tokens_new = torch.cat([tokens_cls, tokens_x, tokens_sz, attentive_tokens_dz, tokens_text], dim=1)

    return tokens_new, keep_index_dz, removed_index_dz


def static_template_elimination_mma(tokens: torch.Tensor, attn: torch.Tensor,
                                lens_sz: int, lens_dz: int,
                                keep_ratio: float,
                                global_index_sz: torch.Tensor,
                                box_mask_z: torch.Tensor,
                                num_template: int,
                                indicate_mask_sz=None,
                                token_type_aware_pruing=None,
                                task_index=None,
                                mmaware_pruning_target = None,
                                ):

    lens_x = tokens.shape[1] - 2 - lens_sz - lens_dz
    bs, hn, _, _ = attn.shape

    # 目前支支持1或者2，sutrack的默认设置
    assert num_template == 2, "num_template must be 2 for TE"

    lens_keep = math.ceil(keep_ratio * lens_sz)
    if lens_keep == lens_sz:
        # print("lens_keep == lens_sz")
        return tokens, global_index_sz, None, box_mask_z, indicate_mask_sz
    # else:
    #     print(f"Prune SZ: {lens_sz - lens_keep}")

    # token: [cls, x, z, text]
    # attn: (bs, num_head, k, v)
    if task_index is not None and (task_index == 1).any() and 'S' in mmaware_pruning_target:  # 根据task_index动态选择注意力提取范围
        attn_template = attn[:, :, 1+lens_x:1+lens_x+lens_sz, 1+lens_x:1+lens_x+lens_sz]
        attn_text = attn[:, :, -1:, 1+lens_x:1+lens_x+lens_sz]

        # 为不需要文本注意力的样本，将文本注意力置零
        text_weight = (task_index == 1).float().view(bs, 1, 1, 1)
        attn_text = attn_text * text_weight  # 只有task_index==1的样本保留文本注意力

        attn_z = torch.cat([attn_template, attn_text], dim=2)   # # (bs, H, lens_sz+1, lens_x)

        # 相应地扩展box_mask_z
        if box_mask_z is not None:  # (bs, lens_sz): True or False
            batch_indices, true_positions = torch.where(box_mask_z)  # 找到box_mask_z中的true的位置
            # 文本token总是有效的，所以创建全True的掩码
            text_mask = torch.ones(bs, 1, device=box_mask_z.device, dtype=box_mask_z.dtype)  # [bs, 1]
            box_mask_z = torch.cat([box_mask_z, text_mask], dim=1)  # [bs, lens_sz+1]
    else:
        attn_z = attn[:, :, 1+lens_x:1+lens_x+lens_sz, 1+lens_x:1+lens_x+lens_sz]
        # attn_z = attn[:, :, -1-lens_dz-lens_sz:-1-lens_dz, -1-lens_dz-lens_sz:-1-lens_dz]   # 等价

    # print(f"static box: {box_mask_z.sum()}")
    if box_mask_z is not None:
        if task_index is None or not (task_index == 1).any() or 'S' not in mmaware_pruning_target: # 另一个分支移到上面完成了
            batch_indices, true_positions = torch.where(box_mask_z)  # 找到box_mask_z中的true的位置
        box_mask_z = box_mask_z.unsqueeze(1).unsqueeze(-1).expand(-1, attn_z.shape[1], -1, attn_z.shape[-1])
        attn_z = attn_z[box_mask_z]
        attn_z = attn_z.view(bs, hn, -1, lens_sz)
        attn_z = attn_z.mean(dim=2).mean(dim=1)  # B, H, L-z, L_x --> B, L_zd
        bonus_mask = torch.zeros_like(attn_z)
        bonus_mask[batch_indices, true_positions] = 1.0 # 给静态模板中心点的attn加上一个大值，防止被删掉    # TODO:待定
        attn_z = attn_z + bonus_mask
    else:
        attn_z = attn_z.mean(dim=2).mean(dim=1)  # B, H, L-z, L_x --> B, L_zd

    if indicate_mask_sz is not None:
        if token_type_aware_pruing in ['FULL_FOREGROUND', 'ALL_FOREGROUND']:
            batch_indices_sz, true_positions_sz = torch.where(indicate_mask_sz)
            indicate_bonus_mask = torch.zeros_like(attn_z)
            indicate_bonus_mask[batch_indices_sz, true_positions_sz] = 1.0
        elif token_type_aware_pruing in ['SOFT_ALL_FOREGROUND']:
            indicate_bonus_mask = indicate_mask_sz
        attn_z = attn_z + indicate_bonus_mask

    # use sort instead of topk, due to the speed issue
    # https://github.com/pytorch/pytorch/issues/22812
    sorted_attn, indices = torch.sort(attn_z, dim=1, descending=True)

    topk_attn, topk_idx = sorted_attn[:, :lens_keep], indices[:, :lens_keep]
    non_topk_attn, non_topk_idx = sorted_attn[:, lens_keep:], indices[:, lens_keep:]

    keep_index_sz = global_index_sz.gather(dim=1, index=topk_idx)
    removed_index_sz = global_index_sz.gather(dim=1, index=non_topk_idx)

    # separate tokens: [cls, x, z, text] (bs, num_tokens, C)
    tokens_cls = tokens[:, 0].unsqueeze(1)
    tokens_x = tokens[:, 1:1+lens_x]
    tokens_sz = tokens[:, 1+lens_x:1+lens_x+lens_sz]
    tokens_dz = tokens[:, 1+lens_x+lens_sz:-1]
    tokens_text = tokens[:, -1].unsqueeze(1)

    # obtain the attentive and inattentive tokens
    B, L, C = tokens_sz.shape

    attentive_tokens_sz = tokens_sz.gather(dim=1, index=topk_idx.unsqueeze(-1).expand(B, -1, C))

    if box_mask_z is not None:
        # box_mask_idx_sorted, _ =  torch.sort(topk_idx, dim=1)   # TODO:7.31修改
        # box_mask_z = box_mask_z[:, 0, :, 0]
        # box_mask_z_new = box_mask_z.gather(dim=1, index=box_mask_idx_sorted)
        if task_index is not None and (task_index == 1).any() and 'S' in mmaware_pruning_target:  # 根据task_index动态选择注意力提取范围
            box_mask_z = box_mask_z[:, 0, :-1, 0]
            # box_mask_z_new = box_mask_z.gather(dim=1, index=box_mask_idx_sorted)
            box_mask_z_new = box_mask_z.gather(dim=1, index=topk_idx)
        else:
            box_mask_z = box_mask_z[:, 0, :, 0]
            # box_mask_z_new = box_mask_z.gather(dim=1, index=box_mask_idx_sorted)
            box_mask_z_new = box_mask_z.gather(dim=1, index=topk_idx)
    else:
        box_mask_z_new = None

    if indicate_mask_sz is not None:
        # indicate_mask_idx_sorted, _ = torch.sort(topk_idx, dim=1) # TODO:7.31修改
        # indicate_mask_sz_new = indicate_mask_sz.gather(dim=1, index=indicate_mask_idx_sorted)
        indicate_mask_sz_new = indicate_mask_sz.gather(dim=1, index=topk_idx)
    else:
        indicate_mask_sz_new = None
    tokens_new = torch.cat([tokens_cls, tokens_x, attentive_tokens_sz, tokens_dz, tokens_text], dim=1)

    return tokens_new, keep_index_sz, removed_index_sz, box_mask_z_new, indicate_mask_sz_new