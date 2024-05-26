from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt
import numpy as np

# 指定 events.out.tfevents 文件的路径
file_path = './checkpoint/events.out.tfevents.1716623598.n1'

# 创建 EventAccumulator 对象来读取事件文件
event_acc = EventAccumulator(file_path)
event_acc.Reload()

# 获取 BPP 和 PSNR 的事件数据
bpp_data = event_acc.Scalars('Train/bpp')
psnr_data = event_acc.Scalars('Train/psnr')

# 提取步数和对应的值
bpp_values = [scalar.value for scalar in bpp_data]
psnr_values = [scalar.value for scalar in psnr_data]

# 获取 PSNR 达到最高值时对应的 BPP 值和索引
best_psnr_index = psnr_values.index(max(psnr_values))
best_bpp_value = bpp_values[best_psnr_index]

# 将 BPP 和 PSNR 数据进行排序
sorted_indices = sorted(range(len(bpp_values)), key=lambda i: bpp_values[i])
sorted_bpp_values = [bpp_values[i] for i in sorted_indices]
sorted_psnr_values = [psnr_values[i] for i in sorted_indices]

# 指定 BPP 的范围和选取的点数
num_points = 20
bpp_min = min(sorted_bpp_values)
bpp_max = max(sorted_bpp_values)

# 在 BPP 范围内均匀选取 num_points 个点
selected_bpp_values = np.linspace(bpp_min, bpp_max, num_points)

# 找到每个选取点对应的最接近的 BPP 值和对应的 PSNR 值
closest_indices = [np.abs(np.array(sorted_bpp_values) - bpp_value).argmin() for bpp_value in selected_bpp_values]
selected_bpp_values = [sorted_bpp_values[i] for i in closest_indices]
selected_psnr_values = [sorted_psnr_values[i] for i in closest_indices]

# 可视化数据
plt.plot(selected_bpp_values, selected_psnr_values, marker='o')
for i, (bpp, psnr) in enumerate(zip(selected_bpp_values, selected_psnr_values)):
    plt.text(bpp, psnr, f'({i+1})')
plt.xlabel('BPP')
plt.ylabel('PSNR')
plt.title('BPP vs PSNR')
plt.show()

