import numpy as np
import torch


############## used for visulize eliminated tokens #################
def get_keep_indices(decisions):
    keep_indices = []
    for i in range(3):
        if i == 0:
            keep_indices.append(decisions[i])
        else:
            keep_indices.append(keep_indices[-1][decisions[i]])
    return keep_indices


# def gen_masked_tokens(tokens, indices, alpha=0.2):
#     # indices = [i for i in range(196) if i not in indices]
#     indices = indices[0].astype(int)
#     tokens = tokens.copy()
#
#     # 检查数据稀疏性
#     one_ratio = (tokens == 255).sum() / tokens.size
#
#     if one_ratio > 0.8:  # 如果80%以上是0（很稀疏）
#         # 对于稀疏数据，使用更明显的mask策略
#         # 使用数据的非零最大值作为mask
#         non_one_values = tokens[tokens != 255]
#         if len(non_one_values) > 0:
#             mask_value = non_one_values.min()
#         else:
#             mask_value = 0  # 如果全是255，用黑色
#         tokens[indices] = mask_value
#     else:
#         # 原有策略
#         tokens[indices] = alpha * tokens[indices] + (1 - alpha) * 255
#
#     # tokens[indices] = alpha * tokens[indices] + (1 - alpha) * 255
#     return tokens
def gen_masked_tokens(tokens, indices, merged_indices=None, host_indices=None, alpha=0.2):
    indices = indices[0].astype(int)
    tokens = tokens.copy()

    # 检查数据稀疏性
    one_ratio = (tokens == 255).sum() / tokens.size

    if one_ratio > 0.8:  # 如果80%以上是0（很稀疏）
        # 对于稀疏数据，使用更明显的mask策略
        # 使用数据的非零最大值作为mask
        non_one_values = tokens[tokens != 255]
        if len(non_one_values) > 0:
            mask_value = non_one_values.min()
        else:
            mask_value = 0  # 如果全是255，用黑色
        tokens[indices] = mask_value
    else:
        # 原有策略
        tokens[indices] = alpha * tokens[indices] + (1 - alpha) * 255

    # 处理merged_indices（黄色遮罩）
    if merged_indices is not None and len(merged_indices) > 0:
        merged_idx = merged_indices[0].astype(int)
        # 黄色 [255, 255, 0]：保持R和G通道亮，B通道暗
        tokens[merged_idx, :, :, 0] = alpha * tokens[merged_idx, :, :, 0] + (1 - alpha) * 255  # R通道
        tokens[merged_idx, :, :, 1] = alpha * tokens[merged_idx, :, :, 1] + (1 - alpha) * 255  # G通道
        tokens[merged_idx, :, :, 2] = alpha * tokens[merged_idx, :, :, 2]  # B通道保持原值的alpha比例

    # 处理host_indices（绿色遮罩）
    if host_indices is not None and len(host_indices) > 0:
        host_idx = host_indices[0].astype(int)
        # 绿色 [0, 255, 0]：只有G通道亮
        tokens[host_idx, :, :, 0] = alpha * tokens[host_idx, :, :, 0]  # R通道保持原值的alpha比例
        tokens[host_idx, :, :, 1] = alpha * tokens[host_idx, :, :, 1] + (1 - alpha) * 255  # G通道
        tokens[host_idx, :, :, 2] = alpha * tokens[host_idx, :, :, 2]  # B通道保持原值的alpha比例

    return tokens


def recover_image(tokens, H, W, Hp, Wp, patch_size):
    # image: (C, 196, 16, 16)
    image = tokens.reshape(Hp, Wp, patch_size, patch_size, 3).swapaxes(1, 2).reshape(H, W, 3)
    return image


def pad_img(img):
    height, width, channels = img.shape
    im_bg = np.ones((height, width + 8, channels)) * 255
    im_bg[0:height, 0:width, :] = img
    return im_bg


def gen_visualization(image, mask_indices, patch_size=16, layout='grid', dte=False, image_dte=None,
                      merged_indices=None, host_indices=None, extra_indices=None,):
    """
    改进的可视化函数，支持剪枝和合并操作的可视化

    Args:
        image: 原始图像
        mask_indices: 被移除的token索引列表（每层一个数组）
        patch_size: patch大小
        layout: 布局方式 ('grid' 或 'horizontal')
        dte: 是否为多模态（深度+RGB）
        image_dte: 深度图像（如果是多模态）
        merged_indices: 被合并的token索引列表（可选）
        host_indices: 接收合并信息的主token索引列表（可选）
        show_merge: 是否显示合并效果

    Returns:
        可视化图像
    """
    # 向后兼容：如果没有提供合并相关参数，使用原始方法
    if merged_indices is None or host_indices is None:
        return _gen_visualization(image, mask_indices, patch_size, layout, dte, image_dte)
    elif extra_indices is not None:
        # EVIT
        return _gen_visualization_with_extra(image, patch_size, layout, dte, image_dte, merged_indices, host_indices)
    else:
        # 新方法：同时显示剪枝和合并效果
        return _gen_visualization_with_merge(image, mask_indices, patch_size, layout, dte, image_dte, merged_indices, host_indices)


def _gen_visualization(image, mask_indices, patch_size=16, layout='grid', dte=False, image_dte=None):
    # mask mask_indices need to cat
    num_stages = len(mask_indices)
    for i in range(1, num_stages):
        mask_indices[i] = np.concatenate([mask_indices[i - 1], mask_indices[i]], axis=1)

    # keep_indices = get_keep_indices(decisions)
    image = np.asarray(image)
    if dte is True:
        image_dte = np.asarray(image_dte)
    H, W, C = image.shape
    Hp, Wp = H // patch_size, W // patch_size
    image_tokens = image.reshape(Hp, patch_size, Wp, patch_size, 3).swapaxes(1, 2).reshape(Hp * Wp, patch_size, patch_size, 3)
    if dte is True:
        image_dte_tokens = image_dte.reshape(Hp, patch_size, Wp, patch_size, 3).swapaxes(1, 2).reshape(Hp * Wp, patch_size, patch_size, 3)

    stages_show = []
    stages = [
        recover_image(gen_masked_tokens(image_tokens, mask_indices[i]), H, W, Hp, Wp, patch_size)
        for i in range(num_stages)
    ]
    if dte is True:
        stages_dte = [
            recover_image(gen_masked_tokens(image_dte_tokens, mask_indices[i]), H, W, Hp, Wp, patch_size)
            for i in range(num_stages)
        ]

    for i in range(num_stages):
        stages_show.append(stages[i])
        if dte is True:
            stages_show.append(stages_dte[i])

    # imgs = [image] + stages
    imgs = stages_show

    if layout == 'grid' and len(imgs) > 4:
        # 如果图像数量超过4张，创建更大的网格
        viz = create_flexible_grid(imgs)
    else:
        # 默认水平排列（原始方式）
        imgs = [pad_img_grid(img) for img in imgs]
        viz = np.concatenate(imgs, axis=1)

    return viz

# def _gen_visualization_with_merge(image, pruned_indices, patch_size, layout, dte, image_dte, merged_indices, host_indices):
#     num_stages = len(pruned_indices)
#     for i in range(1, num_stages):
#         pruned_indices[i] = np.concatenate([pruned_indices[i - 1], pruned_indices[i]], axis=1)
#         merged_indices[i] = np.concatenate([merged_indices[i - 1], merged_indices[i]], axis=1)
#         host_indices[i] = np.concatenate([host_indices[i - 1], host_indices[i]], axis=1)
#
#
#     image = np.asarray(image)
#     if dte:
#         image_dte = np.asarray(image_dte)
#     H, W, C = image.shape
#     Hp, Wp = H // patch_size, W // patch_size
#     image_tokens = image.reshape(Hp, patch_size, Wp, patch_size, 3).swapaxes(1, 2).reshape(Hp * Wp, patch_size, patch_size, 3)
#     if dte is True:
#         image_dte_tokens = image_dte.reshape(Hp, patch_size, Wp, patch_size, 3).swapaxes(1, 2).reshape(Hp * Wp, patch_size, patch_size, 3)
#
#     stages_show = []
#     stages = [
#         recover_image(gen_masked_tokens(
#             image_tokens,
#             pruned_indices[i],
#             merged_indices=merged_indices[i],
#             host_indices=host_indices[i]
#         ), H, W, Hp, Wp, patch_size)
#         for i in range(num_stages)
#     ]
#     if dte is True:
#         stages_dte = [
#             recover_image(gen_masked_tokens(
#                 image_dte_tokens,
#                 pruned_indices[i],
#                 merged_indices=merged_indices[i],
#                 host_indices=host_indices[i]
#             ), H, W, Hp, Wp, patch_size)
#             for i in range(num_stages)
#         ]
#
#     for i in range(num_stages):
#         stages_show.append(stages[i])
#         if dte is True:
#             stages_show.append(stages_dte[i])
#
#     # imgs = [image] + stages
#     imgs = stages_show
#
#     if layout == 'grid' and len(imgs) > 4:
#         # 如果图像数量超过4张，创建更大的网格
#         viz = create_flexible_grid(imgs)
#     else:
#         # 默认水平排列（原始方式）
#         imgs = [pad_img_grid(img) for img in imgs]
#         viz = np.concatenate(imgs, axis=1)
#
#     return viz

def create_flexible_grid(images, cols=8):
    """创建灵活的网格布局，可以处理任意数量的图像"""
    n_images = len(images)
    rows = (n_images + cols - 1) // cols  # 向上取整

    # 如果图像数量不是完整的网格，用白色图像填充
    while len(images) < rows * cols:
        # 创建与第一张图像相同大小的白色图像
        white_img = np.ones_like(images[0]) * 255
        images.append(white_img)

    # 为每张图像添加padding
    padded_images = [pad_img_grid(img) for img in images]

    # 按行组织图像
    grid_rows = []
    for r in range(rows):
        row_images = padded_images[r * cols:(r + 1) * cols]
        row = np.concatenate(row_images, axis=1)
        grid_rows.append(row)

    # 垂直拼接所有行
    grid = np.concatenate(grid_rows, axis=0)

    return grid


def pad_img_grid(img, padding=8):
    """为网格布局添加padding"""
    height, width, channels = img.shape
    # 在右侧和底部添加padding
    im_bg = np.ones((height + padding, width + padding, channels)) * 255
    im_bg[0:height, 0:width, :] = img
    return im_bg

def multi_img_set_visualization(img_list):
    """ 将模板拼起来可视化 """
    num_img = len(img_list)

    im_list = []
    for i in range(num_img):
        im = np.array(img_list[i])
        # 添加空白的部分，方便可视化区分
        # empty_padding = np.zeros((im.shape[0],10,im.shape[2]))
        # im = np.concatenate((im, empty_padding),axis=1)

        im_list.append(im)

    viz = np.concatenate(im_list, axis=1)
    return viz

def _gen_visualization_with_extra(image, patch_size, layout, dte, image_dte, merged_indices, host_indices):
    """
    可视化合并过程（不包含剪枝）
    每一阶段只展示 merged 和 host token
    """
    num_stages = len(merged_indices)

    image = np.asarray(image)
    if dte:
        image_dte = np.asarray(image_dte)

    H, W, C = image.shape
    Hp, Wp = H // patch_size, W // patch_size

    # 切分为 patch tokens
    image_tokens = image.reshape(Hp, patch_size, Wp, patch_size, 3)\
                         .swapaxes(1, 2)\
                         .reshape(Hp * Wp, patch_size, patch_size, 3)

    if dte:
        image_dte_tokens = image_dte.reshape(Hp, patch_size, Wp, patch_size, 3)\
                                   .swapaxes(1, 2)\
                                   .reshape(Hp * Wp, patch_size, patch_size, 3)

    stages_show = []

    for i in range(num_stages):
        current_merged = merged_indices[i]
        current_host = host_indices[i]

        stage_image = recover_image(
            gen_evit_masked_tokens_by_stage(
                image_tokens,
                removed_indices=None,
                merged_indices=current_merged,
                host_indices=current_host
            ), H, W, Hp, Wp, patch_size
        )
        stages_show.append(stage_image)

        if dte:
            stage_image_dte = recover_image(
                gen_evit_masked_tokens_by_stage(
                    image_dte_tokens,
                    removed_indices=None,
                    merged_indices=current_merged,
                    host_indices=current_host
                ), H, W, Hp, Wp, patch_size
            )
            stages_show.append(stage_image_dte)

    # 拼图展示
    imgs = stages_show
    if layout == 'grid' and len(imgs) > 4:
        viz = create_flexible_grid(imgs)
    else:
        imgs = [pad_img_grid(img) for img in imgs]
        viz = np.concatenate(imgs, axis=1)

    return viz

def _gen_visualization_with_merge(image, pruned_indices, patch_size, layout, dte, image_dte, merged_indices, host_indices):
    """
    解决颜色覆盖问题的可视化函数
    策略：在每个阶段，按优先级显示token状态，避免历史操作的颜色覆盖
    """
    num_stages = len(pruned_indices)

    cumulative_removed = []

    for i in range(num_stages):
        if i == 0:  # 第一阶段：只有当前阶段的pruned tokens被移除
            cumulative_removed.append(pruned_indices[i])
        else:   # 后续阶段：累积之前所有被移除的token + 当前阶段的pruned和merged
            prev_removed = cumulative_removed[i - 1]
            prev_merged = merged_indices[i - 1]  # 上一阶段的merged也要移除
            current_pruned = pruned_indices[i]

            # 合并所有需要移除的索引
            all_removed = []
            if len(prev_removed) > 0 and len(prev_removed[0]) > 0:
                all_removed.append(prev_removed[0])
            if len(prev_merged) > 0 and len(prev_merged[0]) > 0:
                all_removed.append(prev_merged[0])
            if len(current_pruned) > 0 and len(current_pruned[0]) > 0:
                all_removed.append(current_pruned[0])

            if all_removed:
                cumulative_removed.append([np.concatenate(all_removed)])
            else:
                cumulative_removed.append([np.array([])])


    image = np.asarray(image)
    if dte:
        image_dte = np.asarray(image_dte)
    H, W, C = image.shape
    Hp, Wp = H // patch_size, W // patch_size
    image_tokens = image.reshape(Hp, patch_size, Wp, patch_size, 3).swapaxes(1, 2).reshape(Hp * Wp, patch_size, patch_size, 3)
    if dte is True:
        image_dte_tokens = image_dte.reshape(Hp, patch_size, Wp, patch_size, 3).swapaxes(1, 2).reshape(Hp * Wp, patch_size, patch_size, 3)

    stages_show = []

    # 为每个阶段生成可视化
    for i in range(num_stages):
        # 当前阶段的状态：
        # - removed: 累积的所有被移除的token
        # - merged: 仅当前阶段的merged token
        # - host: 仅当前阶段的host token
        current_removed = cumulative_removed[i]
        current_merged = merged_indices[i]
        current_host = host_indices[i]

        # RGB图像
        stage_image = recover_image(
            gen_masked_tokens_by_stage(
                image_tokens, current_removed, current_merged, current_host
            ), H, W, Hp, Wp, patch_size
        )
        stages_show.append(stage_image)

        # 深度图像（如果有）
        if dte is True:
            stage_image_dte = recover_image(
                gen_masked_tokens_by_stage(
                    image_dte_tokens, current_removed, current_merged, current_host
                ), H, W, Hp, Wp, patch_size
            )
            stages_show.append(stage_image_dte)

    # 布局处理
    imgs = stages_show
    if layout == 'grid' and len(imgs) > 4:
        viz = create_flexible_grid(imgs)
    else:
        imgs = [pad_img_grid(img) for img in imgs]
        viz = np.concatenate(imgs, axis=1)

    return viz

def gen_evit_masked_tokens_by_stage(tokens, removed_indices, merged_indices, host_indices, alpha=0.2):
    tokens = tokens.copy()

    # 检查数据稀疏性
    one_ratio = (tokens == 255).sum() / tokens.size

    # 1. 处理已移除的tokens（灰色/暗化）
    if removed_indices is not None and len(removed_indices) > 0 and len(removed_indices[0]) > 0:
    # if len(removed_indices) > 0 and len(removed_indices[0]) > 0:
        indices = removed_indices[0].astype(int)

        if one_ratio > 0.8:  # 如果80%以上是255（很稀疏）
            # 对于稀疏数据，使用更明显的mask策略
            non_one_values = tokens[tokens != 255]
            if len(non_one_values) > 0:
                mask_value = non_one_values.min()
            else:
                mask_value = 0  # 如果全是255，用黑色
            tokens[indices] = mask_value
        else:
            tokens[indices] = alpha * tokens[indices] + (1 - alpha) * 255

    # 2. 处理当前阶段被合并的tokens（黄色遮罩）
    if merged_indices is not None and len(merged_indices) > 0 and len(merged_indices[0]) > 0:
        merged_idx = merged_indices[0].astype(int)
        # 黄色 [255, 255, 0]：保持R和G通道亮，B通道暗
        tokens[merged_idx, :, :, 0] = alpha * tokens[merged_idx, :, :, 0] + (1 - alpha) * 255  # R通道
        tokens[merged_idx, :, :, 1] = alpha * tokens[merged_idx, :, :, 1] + (1 - alpha) * 255  # G通道
        tokens[merged_idx, :, :, 2] = alpha * tokens[merged_idx, :, :, 2]  # B通道保持原值的alpha比例

    # 3. 处理当前阶段的宿主tokens（绿色遮罩）
    if host_indices is not None and len(host_indices) > 0 and len(host_indices[0]) > 0:
        host_tensor = host_indices[0]
        if isinstance(host_tensor, torch.Tensor):
            host_idx = host_tensor.cpu().numpy().astype(int)
        else:
            host_idx = host_tensor.astype(int)
        # 绿色 [0, 255, 0]：只有G通道亮
        tokens[host_idx, :, :, 0] = alpha * tokens[host_idx, :, :, 0]  # R通道保持原值的alpha比例
        tokens[host_idx, :, :, 1] = alpha * tokens[host_idx, :, :, 1] + (1 - alpha) * 255  # G通道
        tokens[host_idx, :, :, 2] = alpha * tokens[host_idx, :, :, 2]  # B通道保持原值的alpha比例

    return tokens

def gen_masked_tokens_by_stage(tokens, removed_indices, merged_indices, host_indices, alpha=0.2):
    tokens = tokens.copy()

    # 检查数据稀疏性
    one_ratio = (tokens == 255).sum() / tokens.size

    # 1. 处理已移除的tokens（灰色/暗化）
    if len(removed_indices) > 0 and len(removed_indices[0]) > 0:
        indices = removed_indices[0].astype(int)

        if one_ratio > 0.8:  # 如果80%以上是255（很稀疏）
            # 对于稀疏数据，使用更明显的mask策略
            non_one_values = tokens[tokens != 255]
            if len(non_one_values) > 0:
                mask_value = non_one_values.min()
            else:
                mask_value = 0  # 如果全是255，用黑色
            tokens[indices] = mask_value
        else:
            tokens[indices] = alpha * tokens[indices] + (1 - alpha) * 255

    # 2. 处理当前阶段被合并的tokens（黄色遮罩）
    if merged_indices is not None and len(merged_indices) > 0 and len(merged_indices[0]) > 0:
        merged_idx = merged_indices[0].astype(int)
        # 黄色 [255, 255, 0]：保持R和G通道亮，B通道暗
        tokens[merged_idx, :, :, 0] = alpha * tokens[merged_idx, :, :, 0] + (1 - alpha) * 255  # R通道
        tokens[merged_idx, :, :, 1] = alpha * tokens[merged_idx, :, :, 1] + (1 - alpha) * 255  # G通道
        tokens[merged_idx, :, :, 2] = alpha * tokens[merged_idx, :, :, 2]  # B通道保持原值的alpha比例

    # 3. 处理当前阶段的宿主tokens（绿色遮罩）
    if host_indices is not None and len(host_indices) > 0 and len(host_indices[0]) > 0:
        host_idx = host_indices[0].astype(int)
        # 绿色 [0, 255, 0]：只有G通道亮
        tokens[host_idx, :, :, 0] = alpha * tokens[host_idx, :, :, 0]  # R通道保持原值的alpha比例
        tokens[host_idx, :, :, 1] = alpha * tokens[host_idx, :, :, 1] + (1 - alpha) * 255  # G通道
        tokens[host_idx, :, :, 2] = alpha * tokens[host_idx, :, :, 2]  # B通道保持原值的alpha比例

    return tokens