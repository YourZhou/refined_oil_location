import numpy as np
import matplotlib.pyplot as plt


def euclidean_distance(p1, p2):
    return np.sqrt(np.sum((p1 - p2) ** 2))


def time_diff(t1, t2):
    return abs(t1 - t2)


def similarity(point1, point2, eps_space, eps_time):
    spatial_distance = euclidean_distance(point1[:2], point2[:2])
    time_distance = time_diff(point1[2], point2[2])

    return spatial_distance <= eps_space and time_distance <= eps_time


def traj_dbscan(data, eps_space, eps_time, min_pts):
    labels = np.zeros(len(data), dtype=int) - 1
    cluster_id = 0

    for i, point in enumerate(data):
        if labels[i] != -1:
            continue

        neighbors = [j for j, neighbor in enumerate(data) if similarity(point, neighbor, eps_space, eps_time)]

        if len(neighbors) < min_pts:
            labels[i] = 0  # Noise point
        else:
            labels[i] = cluster_id
            for j in neighbors:
                if labels[j] == -1:
                    labels[j] = cluster_id
            cluster_id += 1

    return labels



# 示例数据，每行表示一个轨迹点，前两列是经度和纬度，第三列是时间戳
data = np.array([
    [22.770561, 108.25768, 1645500641000],
    [22.770561, 108.25768, 1645500941000],
    [22.770563, 108.25768, 1645501241000],
    [22.770563, 108.25768, 1645501541000],
    [22.770565, 108.25768, 1645501841000],
    [22.770565, 108.25768, 1645502141000]
])

eps_space = 0.01  # 空间距离阈值
eps_time = 1000 * 60 * 5  # 时间距离阈值，5 分钟
min_pts = 2

labels = traj_dbscan(data, eps_space, eps_time, min_pts)
print(labels)
# 可视化
unique_labels = np.unique(labels)
colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))

for label, color in zip(unique_labels, colors):
    if label == 0:
        # 噪声点用黑色表示
        color = 'k'

    mask = (labels == label)
    plt.scatter(data[mask, 0], data[mask, 1], c=[color], label=label)

plt.xlabel("Latitude")
plt.ylabel("Longitude")

# 设置坐标轴范围
padding = 0.000001
min_latitude, max_latitude = np.min(data[:, 0]) - padding, np.max(data[:, 0]) + padding
min_longitude, max_longitude = np.min(data[:, 1]) - padding, np.max(data[:, 1]) + padding
plt.xlim(min_latitude, max_latitude)
plt.ylim(min_longitude, max_longitude)

plt.legend()
plt.show()
