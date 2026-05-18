# AION

脑启发主动推理无人机实验原型。当前仓库把 Liquid State Machine、超维表示、Modern Hopfield Network、Global Workspace、内部驱力和 PyBullet 三维环境串成一个可运行的智能体，用来探索“有身体、有记忆、有目标”的主动感知与控制。

## 当前状态

这是一个研究原型，不是完整产品。仓库里已经包含主循环、世界模型预训练脚本、目标导向训练脚本和若干验证脚本，并提交了一个 `pretrained_world_model.pt` 预训练权重。`scripts/run_agent.py` 会优先加载该权重；如果文件不存在，则以未预训练状态启动。

可视化仪表盘相关代码仍在 `src/dashboard.py` 中，但主运行脚本里为了避免无显示环境卡住，默认没有启用 Dashboard。PyBullet GUI 需要本地桌面环境；服务器或 CI 环境建议改为 headless 方式调试。

## 主要模块

- `src/environment_pybullet.py`：PyBullet 三维无人机环境。
- `src/lsm.py`：Liquid State Machine 感知编码。
- `src/adapter.py`：随机投影适配器。
- `src/hrr.py`：超维世界模型表示。
- `src/mhn.py`：Modern Hopfield Network 记忆模块。
- `src/gwt.py`：Global Workspace 信息广播。
- `src/drive.py`：内部驱力与模式切换。
- `src/controller.py`：无人机离散动作控制。
- `scripts/run_agent.py`：智能体主入口。
- `scripts/pretrain_world_model.py`：通过 motor babbling 预训练世界模型。
- `scripts/train_goal_directed.py`：目标导向课程训练实验。

## 快速开始

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/run_agent.py
```

如果需要重新生成世界模型权重：

```bash
python scripts/pretrain_world_model.py
```

目标导向训练入口：

```bash
python scripts/train_goal_directed.py
```

## 运行提示

- 依赖包含 `torch`、`pybullet`、`opencv-python`、`visdom`、`nengo`、`scipy` 等。
- PyBullet GUI 对图形环境有要求；远程服务器上运行时可能需要虚拟显示或改为 DIRECT 模式。
- `pretrained_world_model.pt` 是当前仓库里的示例权重，不代表通用任务性能。
- 该项目仍偏探索性质，接口和实验配置可能随研究推进继续变化。

## 许可证

当前仓库未包含独立 `LICENSE` 文件。如需公开复用或分发，请先补充明确的开源许可证。
