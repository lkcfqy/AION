# Project AION 任务清单

## 0. 预备阶段：环境与仪表盘 (The Setup) [第 1-2 周]
- [x] **0.1 锁定物理环境 (The Body)**
    - [x] 选型：`MiniGrid` (方案 A)
    - [x] 输入定义：将环境的 RGB 像素流 ($W \times H \times 3$) 作为 LSM 的输入
- [x] **0.2 可视化仪表盘 (The Dashboard)**
    - [x] 工具：`Visdom` (实时 Web 面板)
    - [x] **LSM Raster Plot**：监控神经元是否癫痫或死寂
    - [x] **HDC Similarity**：监控当前感官向量 vs 记忆向量的余弦相似度
    - [x] **Energy Landscape**：监控 MHN 的能量下降曲线（记忆召回状态）
    - [x] **Survival Curve**：监控自由能 (Loss) + 饥饿值 (Hunger) 实时曲线

## 1. 阶段一：神经基座与自适应接地 (The Grounding) [第 3-6 周]
- [x] **1.1 自适应液体状态机 (Adaptive LSM)**
    - [x] 架构：3D 稀疏连接 SNN 储水池 (LIF 神经元)
    - [x] **实现稳态可塑性 (Homeostatic Plasticity)** (方案 A)
        - [x] 机制：若发放率 > 目标值，自动提高阈值 $V_{th}$
        - [x] 机制：若发放率 < 目标值，自动降低阈值 $V_{th}$
        - [x] 目标：自动演化到“混沌边缘”，无需手动调参
- [x] **1.2 二值化随机投影适配器 (The Adapter)**
    - [x] 技术选型：**Binary HDC** (全链路 +1/-1)
    - [x] 实现公式：$V_{HDC} = \text{Sign}(W_{rand} \cdot r_{LSM})$
    - [x] 优化：使用 XOR 和 Popcount 加速运算
- [x] **阶段一交付物**
    - [x] "Visualized Grounding" 演示：LSM 节律活动 + 稳定的二值向量输出

## 2. 阶段二：联想记忆与世界模型 (The Mind) [第 7-10 周]
- [x] **2.1 现代霍普菲尔德网络 (MHN)**
    - [x] 功能：去噪与模式补全
    - [x] 实现能量函数：$E = -\text{lse}(\beta, X^T \xi)$ (Log-Sum-Exp)
    - [x] 目标：将抖动向量锁定为能量最低的原型概念
- [x] **2.2 HRR 因果推理**
    - [x] 绑定运算：XOR (用于 $State \otimes Action$)
    - [x] 叠加运算：Majority Rule (多数表决)
    - [x] **世界模型学习**
        - [x] 数据收集：随机游走 $(S_t, A_t, S_{t+1})$
        - [x] 映射学习：$M: S_t \otimes A_t \rightarrow S_{t+1}$
        - [x] 验证：输入“当前位置”+“动作”，预测“新位置特征”

## 3. 阶段三：生存驱动与主动推理 (The Controller) [第 11-14 周]
- [x] **3.1 全局工作空间 (GWT)**
    - [x] 建立竞争瓶颈机制
    - [x] 竞争者：$V_{sense}$ (感知)、$V_{pred}$ (预测)、$V_{goal}$ (目标)
    - [x] 策略：仅显著性最高的信号进入决策层
- [x] **3.2 修正后的自由能 (The Drive)**
    - [x] 策略：拒绝“暗室效应” (方案 A)
    - [x] 实现 Loss 公式：$F = \underbrace{\text{Hamming}(V_{sense}, V_{pred})}_{\text{惊奇度}} + \lambda \cdot \underbrace{\text{HungerLevel}}_{\text{生理需求}}$
- [x] **3.3 主动推理循环 (Active Inference Loop)**
    - [x] 模拟：在 HRR 空间并行模拟所有动作 $\{A_{left}, A_{right}, A_{fwd}\}$
    - [x] 评估：计算每个动作预期的 $F$ 值
    - [x] 决策：执行 $F$ 最小的动作
- [x] **阶段三交付物**
    - [x] "Survival Demo"：避开熔岩 (预测误差驱动) + 寻找食物 (饥饿驱动)

## 4. 阶段四：闭环进化 (The Evolution) [第 15-18 周]
- [x] **4.1 三因素学习规则**
    - [x] 闭环机制：使用 $-F$ (负自由能) 作为全局多巴胺信号
    - [x] 实现公式：$\Delta w = \eta \cdot \text{Pre} \cdot \text{Post} \cdot (-F)$
    - [x] 效果：预测失败或挨饿时触发 LTD (抑制)
- [x] **4.2 睡眠巩固**
    - [x] **SWS (慢波)**：利用 MHN 强化高价值记忆
    - [x] **REM (快动眼)**：切断感官，利用世界模型生成虚构数据，离线训练 SNN (方案 A)

---


