# Project AION: The Singularity Edition (Completed)
**—— 脉冲动力学、现代霍普菲尔德与主动推理的终极融合架构**

**Status: ✅ Mission Accomplished**

### 🧠 核心能力与学习数据 (Core Capabilities)
本项目构建了一个具有生物合理性的智能体，主要学习和处理以下四类核心数据：

1.  **视觉感知 (Visual Perception)**
    *   **数据源**: `MiniGrid` 环境的局部 RGB 图像流。
    *   **处理机制**: 通过脉冲神经网络 (SNN) 和液体状态机 (LSM) 将像素流转化为时空脉冲模式，模拟视网膜到初级视觉皮层的处理。

2.  **因果逻辑 (Causal Dynamics)**
    *   **数据源**: 智能体与环境交互产生的“状态-动作-结果”序列。
    *   **处理机制**: 利用全息缩减表示 (HRR) 学习符号级的因果关系（即 $State \otimes Action \approx Next State$），构建世界模型。

3.  **情景记忆 (Episodic Memory)**
    *   **数据源**: 探索过程中遇到的高价值或高惊奇度事件。
    *   **处理机制**: 使用现代霍普菲尔德网络 (MHN) 存储和检索复杂记忆模式，并支持离线睡眠回放巩固。

4.  **生存策略 (Survival Strategy)**
    *   **数据源**: 内驱力信号（饥饿度、自由能/惊奇度）。
    *   **处理机制**: 基于自由能原理 (FEP)，学习如何通过主动推理 (Active Inference) 最小化长期惊奇度和生理需求。

---

### 🚀 快速开始 (Quick Start)

请按照以下顺序启动系统：

**1. 启动可视化仪表盘 (Visdom)**
这是观察 AION 内部思维的窗口（必须在独立终端先运行）。
```bash
python -m visdom.server
```
*启动后请访问: [http://localhost:8097](http://localhost:8097)*

**2. 启动智能体 (The Agent)**
运行主程序，启动完整的主动推理循环。
```bash
python scripts/run_agent.py
```

**3. 观察现象**
*   **Phase 1 (Wake)**: Agent 在环境中探索，寻找目标。
*   **Dashboard**: 观察 `Survival Curve` (饥饿与自由能) 和 `Grid View` (视觉输入)。
*   **Phase 2 (Sleep)**: 达成目标后，Agent 自动进入睡眠模式，巩固记忆。

---

### 1. 核心设计哲学 (The Philosophy)
本架构构建了一个“活的”智能体，融合 AI 的生存本能与数学严谨性。
*   **动力层 (SNN/LSM)**：处理毫秒级的感官混沌（已实现自适应稳态机制）。
*   **符号层 (HDC/MHN)**：处理认知级的逻辑因果（已实现 HRR 因果推理）。
*   **桥梁 (The Adapter)**：通过**二值化随机投影**实现数学上的“接地”。
*   **驱动力 (FEP + Homeostasis)**：通过最小化**自由能（惊奇度）**与**生理需求**来驱动行为。

---

### ✅ 0. 预备阶段：环境与仪表盘 (The Setup)
*   **物理环境**: `MiniGrid` (动态环境)。
*   **仪表盘**: `Visdom` 实时监控：
    1.  **LSM Raster Plot**：神经元节律。
    2.  **HDC Similarity**：概念相似度。
    3.  **Energy Landscape**：记忆召回能量。
    4.  **Survival Curve**：生存曲线 (Loss + Hunger)。

### ✅ 1. 阶段一：神经基座与自适应接地 (The Grounding)
*   **自适应液体状态机 (Adaptive LSM)**:
    *   实现了**稳态可塑性 (Homeostatic Plasticity)**：神经元自动调整阈值，保持在“混沌边缘”。
*   **适配器 (The Adapter)**:
    *   **Binary HDC**：通过符号函数 $V_{HDC} = \text{Sign}(W_{rand} \cdot r_{LSM})$ 实现高效接地。

### ✅ 2. 阶段二：联想记忆与世界模型 (The Mind)
*   **现代霍普菲尔德网络 (MHN)**:
    *   实现了 $E = -\text{lse}(\beta, X^T \xi)$，具备强大的去噪与联想能力。
*   **HRR 因果推理**:
    *   实现了 $State \otimes Action \rightarrow NextState$ 的全息推演。
    *   **Permutation**: 引入循环移位解决 XOR 对称性问题。

### ✅ 3. 阶段三：生存驱动与主动推理 (The Controller)
*   **全局工作空间 (GWT)**:
    *   建立了感知、预测与目标的统一交换平台。
*   **修正后的自由能 (The Drive)**:
    *   实现了 $F = \text{Surprise} + \text{Hunger} \cdot \text{GoalDelta}$。
    *   Agent 表现出从“探索 (Curiosity)”到“利用 (Hunger)”的动态切换。
*   **主动推理循环**:
    *   Agent 在每一步都能模拟动作后果，并选择最小化自由能的动作。

### ✅ 4. 阶段四：闭环进化 (The Evolution)
*   **三因素学习规则**:
    *   实现了 $\Delta w = \eta \cdot \text{Pre} \cdot \text{Post} \cdot (-F)$。
    *   网络连接根据多巴胺 (自由能变化) 实时重塑。
*   **睡眠巩固**:
    *   实现了 Episodic Replay。Agent 在离线状态下回放高价值记忆，利用突触可塑性巩固经验。

---

### 5. 技术栈清单 (The Stack)
*   **核心语言**: Python 3.9+
*   **环境**: `minigrid`
*   **SNN 框架**: `Nengo`
*   **HDC 计算**: `TorchHD` / `PyTorch`
*   **可视化**: `Visdom`

---
*Project AION by Google DeepMind Team (Simulated)*