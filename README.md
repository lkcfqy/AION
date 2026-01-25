# AION: Active Inference Online Network

AION 是一个基于主动推理（Active Inference）和仿生神经网络（LSM, MHN）的智能体控制系统。本项目模拟了一个四旋翼无人机在 3D 环境中的生存与探索任务。

## 核心特性

- **液态状态机 (LSM)**: 具有稳态可塑性的脉冲神经网络，用于处理视觉输入。
- **现代 Hopfield 网络 (MHN)**: 用于情景记忆和概念存储。
- **超维计算 (HDC)**: 用于认知建模和因果推理。
- **主动推理 (Active Inference)**: 基于自由能最小化的行为决策机制。
- **生存驱动**: 基于“饥饿”和“电量”的生物驱动系统。

## 环境要求

- Python 3.8+
- PyBullet (物理仿真)
- Visdom (实时可视化)

## 安装

1. 克隆代码库：
   ```bash
   git clone <repository_url>
   cd AION
   ```

2. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```

## 运行指南

### 1. 启动可视化服务器
本项目使用 Visdom 进行实时监控。在运行 Agent 之前，请在一个终端窗口中启动 Visdom 服务器：

```bash
python -m visdom.server
```
*访问 http://localhost:8097 查看仪表盘。*

### 2. 运行 Agent
在另一个终端窗口中启动主程序：

```bash
python scripts/run_agent.py
```

## 文件结构

- `src/`: 核心源代码
  - `lsm.py`: 液态状态机实现
  - `mhn.py`: Hopfield 网络记忆系统
  - `environment_pybullet.py`: PyBullet 仿真环境包装器
  - `dashboard.py`: Visdom 可视化控制
- `scripts/`: 运行脚本
  - `run_agent.py`: Agent 主入口程序

## 交互控制
Agent 将自动运行，经历以下阶段：
1. **Goal Imprinting**: 快速定位并学习目标特征。
2. **Motor Babbling**: 随机运动以学习身体图式（动作-结果映射）。
3. **Active Inference Survival**: 正式生存任务，寻找能量源并避免撞击。
