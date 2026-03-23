from ce import adjust_keep_rate
import matplotlib.pyplot as plt


total_epoch = 180
ce_start_epoch = 20
ce_warm_epoch = 80
ce_total_epoch = ce_start_epoch + ce_warm_epoch
base_keep_rate = 0.7

# 收集数据
epochs = []
keep_rates = []
for i in range(180):
    curr_keep_rate = adjust_keep_rate(i, ce_start_epoch, ce_total_epoch, ITERS_PER_EPOCH=1, base_keep_rate=base_keep_rate)
    epochs.append(i)
    keep_rates.append(curr_keep_rate)
    print(f"Epoch {i}: {curr_keep_rate:.4f}")

# 绘制曲线图
plt.figure(figsize=(12, 6))
plt.plot(epochs, keep_rates, linewidth=2, color='blue', label='Keep Rate')

# 添加关键节点的垂直线
plt.axvline(x=ce_start_epoch, color='red', linestyle='--', alpha=0.7, label=f'Start Epoch ({ce_start_epoch})')
plt.axvline(x=ce_total_epoch, color='green', linestyle='--', alpha=0.7, label=f'End Epoch ({ce_total_epoch})')

# 添加水平参考线
plt.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5, label='Max Keep Rate (1.0)')
plt.axhline(y=base_keep_rate, color='orange', linestyle=':', alpha=0.7, label=f'Base Keep Rate ({base_keep_rate})')

# 图表美化
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Keep Rate', fontsize=12)
plt.title('Keep Rate Adjustment Curve', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend()

# 设置y轴范围，留点边距
plt.ylim(base_keep_rate - 0.1, 1.1)

# 添加文本注释
plt.text(ce_start_epoch/2, 1.05, 'Warmup Phase\n(Keep Rate = 1.0)', ha='center', va='bottom', fontsize=10, bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7))

plt.text((ce_start_epoch + ce_total_epoch)/2, 0.85, 'Cosine Annealing\nPhase', ha='center', va='center', fontsize=10, bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7))

plt.text((ce_total_epoch + total_epoch)/2, base_keep_rate + 0.05, f'Stable Phase\n(Keep Rate = {base_keep_rate})', ha='center', va='bottom', fontsize=10, bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.7))

plt.tight_layout()
plt.show()

# 打印关键统计信息
print(f"\n=== 关键统计信息 ===")
print(f"总训练轮数: {total_epoch}")
print(f"预热阶段: Epoch 0-{ce_start_epoch-1} (Keep Rate = 1.0)")
print(f"余弦退火阶段: Epoch {ce_start_epoch}-{ce_total_epoch-1}")
print(f"稳定阶段: Epoch {ce_total_epoch}-{total_epoch-1} (Keep Rate = {base_keep_rate})")
print(f"最低保留率: {min(keep_rates):.4f}")
print(f"最高保留率: {max(keep_rates):.4f}")