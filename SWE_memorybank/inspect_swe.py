from pathlib import Path

import h5py
import numpy as np
import matplotlib.pyplot as plt


# ==================== 1. 路径设置 ====================

# 当前 Python 文件所在的文件夹，即项目根目录
PROJECT_DIR = Path(__file__).resolve().parent

DATA_PATH = PROJECT_DIR / "data" / "2D_rdb_NA_NA.h5"
RESULT_DIR = PROJECT_DIR / "results"


# ==================== 2. 读取数据 ====================

def read_first_trajectory():
    """查看 HDF5 结构，只读取第一条轨迹。"""

    if not DATA_PATH.is_file():
        raise FileNotFoundError(
            f"\n没有找到数据文件：\n{DATA_PATH}\n"
            "请等待下载完成，并将文件放入 data 文件夹。"
        )

    with h5py.File(DATA_PATH, "r") as f:
        keys = sorted(f.keys())

        print("文件位置：", DATA_PATH)
        print("顶层对象数量：", len(keys))
        print("前 10 个对象：", keys[:10])

        # 查找内部含有 data 数据集的轨迹组
        trajectory_keys = [
            key for key in keys
            if isinstance(f[key], h5py.Group)
            and "data" in f[key]
            and isinstance(f[key]["data"], h5py.Dataset)
        ]

        if not trajectory_keys:
            raise ValueError(
                "没有找到预期的“轨迹组/data”结构。"
                "请将上面的输出发给我，确认实际文件布局。"
            )

        first_key = trajectory_keys[0]
        group = f[first_key]
        dataset = group["data"]

        print("\n轨迹数量：", len(trajectory_keys))
        print("当前读取的轨迹：", first_key)
        print("轨迹内的对象：", list(group.keys()))
        print("原始数据形状：", dataset.shape)
        print("原始数据类型：", dataset.dtype)

        # 打印第一条轨迹内的数据集信息，不加载其他轨迹
        def show_structure(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"  {name}: shape={obj.shape}, dtype={obj.dtype}")

        print("\n第一条轨迹的内部结构：")
        group.visititems(show_structure)

        # 预期：[时间, 空间x, 空间y, 物理通道]
        if dataset.ndim != 4 or dataset.shape[-1] != 1:
            raise ValueError(
                f"数据形状为 {dataset.shape}，"
                "与预期的 [T, H, W, 1] 不一致，请先确认。"
            )

        # 只读当前轨迹的单通道水深，并转为 float32
        height = np.asarray(dataset[..., 0], dtype=np.float32)

    if not np.isfinite(height).all():
        raise ValueError("这条轨迹中存在 NaN 或 Inf。")

    return height


# ==================== 3. 可视化 ====================

def plot_trajectory(height):
    """画出同一条轨迹在四个时间帧的水深。"""

    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    n_frames = height.shape[0]
    indices = np.linspace(0, n_frames - 1, 4, dtype=int)

    # 各图使用同一色标，便于比较
    vmin = float(height.min())
    vmax = float(height.max())

    fig, axes = plt.subplots(
        1, 4,
        figsize=(14, 3.5),
        constrained_layout=True,
    )

    for ax, index in zip(axes, indices):
        # 原数组按 [x,y] 存储时，转置后让 x 对应横轴
        im = ax.imshow(
            height[index].T,
            origin="lower",
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
        )

        ax.set_title(f"Frame {index}")
        ax.set_xlabel("x grid index")
        ax.set_ylabel("y grid index")

    fig.colorbar(im, ax=axes, label="Water depth")

    save_path = RESULT_DIR / "swe_first_trajectory.png"
    fig.savefig(save_path, dpi=180)

    print("\n图片已保存到：", save_path)

    plt.show()
    plt.close(fig)


# ==================== 4. 构造一个输入输出样例 ====================

def inspect_training_sample(height):
    """演示用连续 4 帧预测下一帧，暂不划分训练集。"""

    history = 4

    # 对每个 2×2 小块取平均，将 128×128 降为 64×64
    # 时间帧数保持不变
    t, nx, ny = height.shape

    if nx % 2 != 0 or ny % 2 != 0:
        raise ValueError("空间网格不是偶数，无法按 2×2 分块取平均。")

    if t <= history:
        raise ValueError("时间帧数不足以构造样例。")

    height_small = height.reshape(
        t, nx // 2, 2, ny // 2, 2
    ).mean(axis=(2, 4))

    x = height_small[:history]
    y = height_small[history:history + 1]

    print("\n空间下采样后 [T,H,W]：", height_small.shape)
    print("输入 x [历史帧数,H,W]：", x.shape)
    print("目标 y [1,H,W]：", y.shape)

    # 增加 batch 维度，演示模型实际接收的形状
    x_batch = x[None, ...]
    y_batch = y[None, ...]

    print("添加 batch 维度后：")
    print("x_batch：", x_batch.shape)
    print("y_batch：", y_batch.shape)


# ==================== 5. 主程序 ====================

def main():
    height = read_first_trajectory()

    print("\n水深数组 [T,H,W]：", height.shape)
    print(f"水深最小值：{height.min():.6f}")
    print(f"水深最大值：{height.max():.6f}")

    inspect_training_sample(height)
    plot_trajectory(height)
·

if __name__ == "__main__":
    main()