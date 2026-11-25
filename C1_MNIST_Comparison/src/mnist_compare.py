# -*- coding: utf-8 -*-
"""
C1_MNIST_Comparison —— MLP 与 CNN 在 MNIST 手写数字识别上的对比

功能:
    1. 使用 torchvision.datasets.MNIST 自动下载 MNIST 到 C1_MNIST_Comparison/data
    2. 构建 MLP(全连接) 与 CNN(卷积) 两个模型
    3. 分别训练 3 个 epoch, 在测试集上评估准确率
    4. 首次前向传播时打印输入及各层输出的张量 shape, 便于撰写文档
    5. 绘制训练 Loss 曲线与测试准确率对比图 -> images/training_curves.png

说明:
    WSL 环境没有图形界面, 因此 matplotlib 使用 Agg 后端,
    只调用 plt.savefig 保存图片, 不调用 plt.show 弹窗。
"""

import random
from pathlib import Path

# 必须在 import pyplot 之前设置后端, 否则 matplotlib 可能尝试连接显示服务
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ---------------------------------------------------------------------------
# 路径与超参数配置
# ---------------------------------------------------------------------------
# __file__ 是当前脚本路径, 两次 parent 得到项目子目录 C1_MNIST_Comparison,
# 这样无论从哪个目录运行脚本, 数据和图片路径都不会出错。
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
IMAGES_DIR = BASE_DIR / "images"
FIGURE_PATH = IMAGES_DIR / "training_curves.png"

BATCH_SIZE = 128
EPOCHS = 3
LEARNING_RATE = 1e-3
SEED = 42

# 图表配色: 通过 dataviz 调色板校验(蓝色/橙色, 色盲友好, 对比度 >= 3:1)
SERIES_COLORS = {"MLP": "#2a78d6", "CNN": "#eb6834"}
SURFACE_COLOR = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID_COLOR = "#e4e3df"


# ---------------------------------------------------------------------------
# 模型定义
# ---------------------------------------------------------------------------
class MLP(nn.Module):
    """多层感知机: 先把 28x28 展平成 784 维, 再经过两个隐藏层输出 10 类。

    全连接层把每个像素当作独立特征, 丢弃了像素之间的空间相邻关系,
    因此结构简单、参数少, 但提取局部纹理(笔画)的能力弱于 CNN。
    """

    def __init__(self) -> None:
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 256)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 128)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        return self.fc3(x)


class CNN(nn.Module):
    """卷积神经网络: 两次 "卷积 + BN + ReLU + 池化" 提取空间特征。

    - Conv2d(3x3, padding=1) 保持特征图尺寸不变, 只改变通道数;
    - MaxPool2d(2) 把宽高各减半, 扩大感受野并降低计算量;
    - BatchNorm 稳定训练、加快收敛;
    - Flatten 后接全连接层完成 10 分类。
    """

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),   # 1x28x28  -> 32x28x28
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),                              # 32x28x28 -> 32x14x14
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 32x14x14 -> 64x14x14
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),                              # 64x14x14 -> 64x7x7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),                                 # 64x7x7   -> 3136
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


# ---------------------------------------------------------------------------
# 张量维度流转打印
# ---------------------------------------------------------------------------
def print_tensor_flow(
    model: nn.Module, sample: torch.Tensor, model_name: str
) -> None:
    """打印一份样本经过模型时, 输入以及每一层输出的张量 shape。

    实现方式: 给每个"叶子模块"(Conv2d / Linear / ReLU / MaxPool2d ...)
    注册一次性 forward hook, 在模块完成计算后读取输出 shape。
    这样打印的是真实前向传播的结果, 而不是手算的尺寸。
    """
    print(f"\n[{model_name}] 张量维度流转 (首次输入)")
    print(f"  输入张量: {tuple(sample.shape)}")
    print(f"  {'层名':<24}{'类型':<18}{'输出 shape'}")
    print("  " + "-" * 58)

    handles = []

    def make_hook(layer_name: str, layer_type: str):
        def hook(module, inputs, output) -> None:
            # 部分模块可能返回 tuple, 这里只关心主输出
            out = output[0] if isinstance(output, tuple) else output
            print(f"  {layer_name:<24}{layer_type:<18}{tuple(out.shape)}")

        return hook

    for name, module in model.named_modules():
        # 跳过模型本身和有子模块的容器, 只给真正执行运算的叶子模块挂钩子
        if module is model or list(module.children()):
            continue
        handles.append(
            module.register_forward_hook(make_hook(name, type(module).__name__))
        )

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            model(sample)
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)


# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------
def build_dataloaders() -> tuple[DataLoader, DataLoader]:
    """下载/加载 MNIST 并返回训练集与测试集的 DataLoader。

    ToTensor 把 PIL 图像 (0~255) 转成 [0,1] 的 float 张量, 形状变为 (1,28,28);
    Normalize 再用 MNIST 的全局均值/标准差做标准化, 使输入分布更接近 0 均值,
    有利于梯度下降稳定收敛。
    """
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    train_dataset = datasets.MNIST(
        root=str(DATA_DIR), train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root=str(DATA_DIR), train=False, download=True, transform=transform
    )

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=pin_memory,
    )
    return train_loader, test_loader


# ---------------------------------------------------------------------------
# 训练与评估
# ---------------------------------------------------------------------------
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """训练一个 epoch, 返回按样本数加权的平均训练 Loss。"""
    model.train()
    total_loss = 0.0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # loss.item() 是当前 batch 的平均 loss, 乘 batch 大小还原成总 loss,
        # 最后除以整个数据集大小, 避免最后一个不满的 batch 影响权重。
        total_loss += loss.item() * images.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """在测试集上评估, 返回 (平均 Loss, 准确率)。"""
    model.eval()
    total_loss = 0.0
    correct = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        total_loss += criterion(outputs, labels).item() * images.size(0)

        # argmax 得到预测类别, 与真实标签逐样本比较后累加正确数
        correct += (outputs.argmax(dim=1) == labels).sum().item()

    num_samples = len(loader.dataset)
    return total_loss / num_samples, correct / num_samples


# ---------------------------------------------------------------------------
# 可视化
# ---------------------------------------------------------------------------
def set_axes_style(ax: plt.Axes) -> None:
    """统一坐标轴样式: 去掉上/右边框, 只保留浅色实线网格。"""
    ax.set_facecolor(SURFACE_COLOR)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID_COLOR)
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8, linestyle="-", zorder=0)
    ax.set_axisbelow(True)


def plot_curves(histories: dict[str, dict[str, list[float]]]) -> None:
    """绘制两个子图: 左为训练 Loss 曲线, 右为测试准确率曲线。

    两张图共用 x 轴(Epoch)但各自独立的 y 轴, 不使用双 y 轴,
    避免两种量纲相互干扰。
    """
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 4.8))
    fig.patch.set_facecolor(SURFACE_COLOR)

    epochs = list(range(1, EPOCHS + 1))

    for name, history in histories.items():
        color = SERIES_COLORS[name]

        # Loss 曲线: 2px 线宽, 8px 圆点, 白色描边防重叠
        ax_loss.plot(
            epochs,
            history["train_loss"],
            color=color,
            linewidth=2,
            marker="o",
            markersize=7,
            markeredgecolor=SURFACE_COLOR,
            markeredgewidth=1.5,
            label=name,
            zorder=3,
        )
        # 直接标注末端数值, 只标关键点, 不在每个点上都写数字
        ax_loss.annotate(
            f"{history['train_loss'][-1]:.3f}",
            xy=(epochs[-1], history["train_loss"][-1]),
            xytext=(8, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
            color=TEXT_SECONDARY,
        )

        acc_percent = [acc * 100 for acc in history["test_acc"]]
        ax_acc.plot(
            epochs,
            acc_percent,
            color=color,
            linewidth=2,
            marker="o",
            markersize=7,
            markeredgecolor=SURFACE_COLOR,
            markeredgewidth=1.5,
            label=name,
            zorder=3,
        )
        ax_acc.annotate(
            f"{acc_percent[-1]:.1f}%",
            xy=(epochs[-1], acc_percent[-1]),
            xytext=(8, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
            color=TEXT_SECONDARY,
        )

    # 目标线: 帮助直观看是否达到题目要求的准确率门槛
    for threshold, label in ((95, "MLP target 95%"), (98, "CNN target 98%")):
        ax_acc.axhline(
            threshold, color="#b8b7b1", linewidth=1, linestyle=(0, (4, 4)), zorder=1
        )
        # 标注放在虚线下方、图内左侧的空白区域, 避免与 y 轴标题和折线端点重叠
        ax_acc.text(
            epochs[0] - 0.25,
            threshold - 0.45,
            label,
            fontsize=8,
            color=TEXT_SECONDARY,
            va="top",
        )

    ax_loss.set_title("Training loss per epoch", color=TEXT_PRIMARY, fontsize=12, pad=10)
    ax_loss.set_xlabel("Epoch", color=TEXT_SECONDARY, fontsize=10)
    ax_loss.set_ylabel("Cross-entropy loss", color=TEXT_SECONDARY, fontsize=10)
    ax_loss.set_xticks(epochs)
    ax_loss.set_xlim(epochs[0] - 0.3, epochs[-1] + 0.9)

    ax_acc.set_title("Test accuracy per epoch", color=TEXT_PRIMARY, fontsize=12, pad=10)
    ax_acc.set_xlabel("Epoch", color=TEXT_SECONDARY, fontsize=10)
    ax_acc.set_ylabel("Accuracy (%)", color=TEXT_SECONDARY, fontsize=10)
    ax_acc.set_xticks(epochs)
    ax_acc.set_xlim(epochs[0] - 0.3, epochs[-1] + 0.9)
    ax_acc.set_ylim(90, 100.5)
    ax_acc.set_yticks(range(90, 101, 2))

    for ax in (ax_loss, ax_acc):
        set_axes_style(ax)
        # 两个系列必须有图例, 身份不能只靠颜色
        ax.legend(frameon=False, fontsize=9, labelcolor=TEXT_PRIMARY)

    fig.suptitle(
        "MNIST: MLP vs CNN over 3 epochs",
        color=TEXT_PRIMARY,
        fontsize=14,
        y=1.02,
    )
    fig.tight_layout()

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PATH, dpi=150, bbox_inches="tight", facecolor=SURFACE_COLOR)
    plt.close(fig)
    print(f"曲线已保存: {FIGURE_PATH}")


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    """固定随机种子, 让两次运行结果尽量可复现。"""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main() -> None:
    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"使用设备: {device}")
    if device.type == "cuda":
        print(f"GPU 型号: {torch.cuda.get_device_name(0)}")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    print("\n准备数据集(首次运行会自动下载)...")
    train_loader, test_loader = build_dataloaders()
    print(f"训练集样本数: {len(train_loader.dataset)}")
    print(f"测试集样本数: {len(test_loader.dataset)}")

    # 取第一个 batch 作为样例, 用于打印张量维度流转
    sample_images, sample_labels = next(iter(train_loader))
    print(f"样例 batch: images shape={tuple(sample_images.shape)}, "
          f"labels shape={tuple(sample_labels.shape)}")

    models = {
        "MLP": MLP().to(device),
        "CNN": CNN().to(device),
    }
    histories: dict[str, dict[str, list[float]]] = {}

    for name, model in models.items():
        print("\n" + "=" * 64)
        print(f"开始训练 {name}")
        print("=" * 64)

        # 在正式训练前, 用同一个样例 batch 打印一次完整的 shape 变化
        print_tensor_flow(model, sample_images.to(device), name)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        history: dict[str, list[float]] = {
            "train_loss": [],
            "test_loss": [],
            "test_acc": [],
        }

        for epoch in range(1, EPOCHS + 1):
            train_loss = train_one_epoch(
                model, train_loader, optimizer, criterion, device
            )
            test_loss, test_acc = evaluate(
                model, test_loader, criterion, device
            )

            history["train_loss"].append(train_loss)
            history["test_loss"].append(test_loss)
            history["test_acc"].append(test_acc)

            print(
                f"Epoch {epoch}/{EPOCHS} | "
                f"train_loss={train_loss:.4f} | "
                f"test_loss={test_loss:.4f} | "
                f"test_acc={test_acc:.2%}"
            )

        histories[name] = history

    plot_curves(histories)

    # ------------------------------------------------------------------
    # 最终结果汇总
    # ------------------------------------------------------------------
    print("\n" + "=" * 64)
    print("最终测试集准确率")
    print("=" * 64)
    targets = {"MLP": 95.0, "CNN": 98.0}
    results = {}
    for name, history in histories.items():
        final_acc = history["test_acc"][-1] * 100
        results[name] = final_acc
        status = "达标" if final_acc > targets[name] else "未达标"
        print(
            f"  {name}: {final_acc:.2f}%  "
            f"(目标 > {targets[name]:.0f}%, {status})"
        )

    if results.get("MLP", 0) > 95.0 and results.get("CNN", 0) > 98.0:
        print("\n全部准确率目标已达成。")
    else:
        print("\n[警告] 有模型未达到准确率目标, 可尝试增加 EPOCHS。")


if __name__ == "__main__":
    main()
