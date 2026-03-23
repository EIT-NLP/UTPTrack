from typing import Tuple, List, Tuple, Union

import torch


def parse_r(num_layers: int, r: Union[List[int], Tuple[int, float], int]) -> List[int]:
    """
    Process a constant r or r schedule into a list for use internally.

    r can take the following forms:
     - int: A constant number of tokens per layer.
     - Tuple[int, float]: A pair of r, inflection.
       Inflection describes there the the reduction / layer should trend
       upward (+1), downward (-1), or stay constant (0). A value of (r, 0)
       is as providing a constant r. (r, -1) is what we describe in the paper
       as "decreasing schedule". Any value between -1 and +1 is accepted.
     - List[int]: A specific number of tokens per layer. For extreme granularity.
    """
    inflect = 0
    if isinstance(r, list):
        if len(r) < num_layers:
            r = r + [0] * (num_layers - len(r))
        return list(r)
    elif isinstance(r, tuple):
        r, inflect = r

    min_val = int(r * (1.0 - inflect))
    max_val = 2 * r - min_val
    step = (max_val - min_val) / (num_layers - 1)

    return [int(min_val + step * i) for i in range(num_layers)]


# 二分图软匹配算法,加权平均
def merge_wavg_bsm(
    xz: torch.Tensor,
    metric: torch.Tensor,
    global_index_x: torch.Tensor,
    r: int,
    lens_z: int,
    class_token: bool = False,
    size: torch.Tensor = None,
    mode: str = "wavg",
):

    lens_x = metric.shape[1] - lens_z
    protected = lens_z
    if class_token:
        protected += 1
        lens_x -= 1

    # We can only reduce by a maximum of 50% tokens
    t = metric.shape[1]
    r = min(r, (t - protected) // 2)

    if r <= 0:
        return xz, size

    if size is None:
        size = torch.ones_like(metric[..., 0, None])

    # 分离各个部分
    start_idx = 0
    cls_part = None
    cls_size = None
    if class_token:
        cls_part = xz[:, start_idx : start_idx + 1, :]
        cls_size = size[:, start_idx : start_idx + 1, :]
        start_idx += 1

    metric_x = metric[:, start_idx : start_idx + lens_x, :]
    x_part = xz[:, start_idx : start_idx + lens_x, :]
    x_size = size[:, start_idx : start_idx + lens_x, :]
    start_idx += lens_x

    z_part = xz[:, start_idx : start_idx + lens_z, :]
    z_size = size[:, start_idx : start_idx + lens_z, :]
    start_idx += lens_z

    with torch.no_grad():
        metric_x = metric_x / metric_x.norm(dim=-1, keepdim=True)

        # 将 tokens 分成两组：偶数位置[0,2,4]和奇数位置[1,3,5]，偶数令牌位置被合并到奇数令牌位置
        a, b = metric_x[..., ::2, :], metric_x[..., 1::2, :]
        # 计算相似度矩阵 (B, group_a, group_b)
        scores = a @ b.transpose(-1, -2)

        # 找到最相似的 token 对
        node_max, node_idx = scores.max(dim=-1)  # 每个偶数令牌的最佳奇数匹配
        edge_idx = node_max.argsort(dim=-1, descending=True)[
            ..., None
        ]  # # 按相似度排序

        # edge_idx, node_idx, 都是bsm中的idx
        unm_idx = edge_idx[..., r:, :]  # 未合并的token索引
        src_idx = edge_idx[..., :r, :]  # 源token索引（将被合并）
        dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)  # 目标token索引

    src_bsm_idx = (
        src_idx.squeeze(-1) * 2
    )  # 当前序列的偶数位置，偶数组会被合并的token的原始idx
    x_removed_bsm_idx, _ = torch.sort(src_bsm_idx, dim=1)
    x_removed_global_idx = global_index_x.gather(dim=1, index=x_removed_bsm_idx)

    def apply_weighted_merge(x_tensor, size_tensor):
        # 分成奇偶组
        src_x, dst_x = x_tensor[..., ::2, :], x_tensor[..., 1::2, :]
        src_size, dst_size = size_tensor[..., ::2, :], size_tensor[..., 1::2, :]

        n, t1, c = src_x.shape

        # 获取未合并的tokens
        unm_x = src_x.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c))
        unm_size = src_size.gather(dim=-2, index=unm_idx.expand(n, t1 - r, 1))
        unm_bsm_idx = unm_idx.squeeze(-1) * 2  # 偶数组不合并的token的原始idx

        # 获取要合并的源tokens
        src_to_merge_x = src_x.gather(dim=-2, index=src_idx.expand(n, r, c))
        src_to_merge_size = src_size.gather(dim=-2, index=src_idx.expand(n, r, 1))

        if mode == "wavg":  # 加权平均
            # 计算加权和：x * size
            weighted_src = src_to_merge_x * src_to_merge_size
            # 累加加权特征到目标位置
            dst_x = dst_x * dst_size  # 目标位置也要加权
            dst_x = dst_x.scatter_add(-2, dst_idx.expand(n, r, c), weighted_src)
            # 累加权重到目标位置
            dst_size = dst_size.scatter_add(
                -2, dst_idx.expand(n, r, 1), src_to_merge_size
            )
            # 计算加权平均：避免除零
            dst_x = dst_x / torch.clamp(dst_size, min=1e-6)
        else:
            raise ValueError(f"不支持的reduce模式: {mode}")

        # 记录奇数令牌的原始索引
        dst_bsm_idx = torch.arange(1, 2 * dst_x.shape[1], 2, device=dst_x.device)
        dst_bsm_idx = dst_bsm_idx.unsqueeze(0).expand(n, -1)

        # 合并
        x_keep_tokens = torch.cat([unm_x, dst_x], dim=1)
        x_keep_sizes = torch.cat([unm_size, dst_size], dim=1)
        x_keep_bsml_idx = torch.cat([unm_bsm_idx, dst_bsm_idx], dim=1)

        # 根据原始索引排序，恢复正确顺序
        sorted_keep_bsm_idx = torch.argsort(x_keep_bsml_idx, dim=1)

        merged_x = x_keep_tokens.gather(
            dim=1, index=sorted_keep_bsm_idx.unsqueeze(-1).expand(-1, -1, c)
        )
        merged_x_size = x_keep_sizes.gather(
            dim=1, index=sorted_keep_bsm_idx.unsqueeze(-1).expand(-1, -1, 1)
        )
        merged_x_idx = x_keep_bsml_idx.gather(
            dim=1, index=sorted_keep_bsm_idx
        )  # [B, M]

        return merged_x, merged_x_size, merged_x_idx

    keep_x_part, keep_x_size, x_keep_bsm_idx = apply_weighted_merge(x_part, x_size)

    x_keep_global_index = global_index_x.gather(dim=1, index=x_keep_bsm_idx)

    # 重新组合所有部分
    combined_parts = []
    combined_sizes = []
    if class_token:
        combined_parts.append(cls_part)
        combined_sizes.append(cls_size)

    combined_parts.append(keep_x_part)  # 合并后的x_tokens
    combined_sizes.append(keep_x_size)

    combined_parts.append(z_part)  # z_part（不变）
    combined_sizes.append(z_size)

    tokens_new = torch.cat(combined_parts, dim=1)
    tokens_new_size = torch.cat(combined_sizes, dim=1)

    return tokens_new, tokens_new_size, x_keep_global_index, x_removed_global_idx
