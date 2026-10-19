import sys
print("当前解释器：", sys.executable)
import torch

from ffno_core.grid_2d import FNOFactorized2DBlock


def main():
    torch.manual_seed(0)

    # 先用 CPU 确认模型前向和反向传播都正常
    device = torch.device("cpu")

    model = FNOFactorized2DBlock(
        modes=16,
        width=32,
        input_dim=3,       # 当前水深 h + 坐标 x、y
        n_layers=4,
        factor=4,
        n_ff_layers=2,
        share_weight=False,
        share_fork=False,
        ff_weight_norm=False,
        layer_norm=False,
        use_fork=False,
        dropout=0.0,
        in_dropout=0.0,
        mode="full",
    ).to(device)

    # 模拟一个 batch：2 张 64×64 的单通道水深图
    # 官方模型采用 channels-last 布局：[B,H,W,C]
    h = torch.randn(2, 64, 64, 1, device=device)

    coords = torch.linspace(0, 1, 64, device=device)
    xx, yy = torch.meshgrid(coords, coords, indexing="ij")

    grid = torch.stack([xx, yy], dim=-1)
    grid = grid.unsqueeze(0).expand(h.shape[0], -1, -1, -1)

    inputs = torch.cat([h, grid], dim=-1)

    # 官方模型返回字典，从 forecast 中获取预测
    outputs = model(inputs)
    prediction = outputs["forecast"]

    assert prediction.shape == h.shape
    assert torch.isfinite(prediction).all()

    # 人工目标仅用于检查梯度，不代表真实浅水实验
    target = torch.randn_like(prediction)
    loss = torch.mean((prediction - target) ** 2)
    loss.backward()

    grads = [
        p.grad for p in model.parameters()
        if p.requires_grad and p.grad is not None
    ]
    assert grads, "没有生成梯度。"
    assert all(torch.isfinite(g).all() for g in grads)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(
        p.numel() for p in model.parameters()
        if p.requires_grad
    )

    print("模型：官方 FNOFactorized2DBlock")
    print("设备：", device)
    print("输入形状：", tuple(inputs.shape))
    print("输出形状：", tuple(prediction.shape))
    print("总参数量：", total)
    print("可训练参数量：", trainable)
    print("前向传播、反向传播检查通过")


if __name__ == "__main__":
    main()