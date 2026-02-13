# 🌟 AION: Active Inference Online Network 🚁✨

Welcome to **AION**! This is an intelligent agent control system powered by **Active Inference** and **Brain-inspired Neural Networks**. 🧠✨ This project simulates a quadcopter drone's survival and exploration journey in a 3D physical environment! 🌍

## ✨ Core Features 🛠️

* 🌊 **Liquid State Machine (LSM)**: A spiking neural network with homeostatic plasticity for processing visual input. 👀
* 🧠 **Modern Hopfield Network (MHN)**: Dense associative memory for episodic memory and concept storage. 📚
* 🧮 **Hyperdimensional Computing (HDC)**: Cognitive modeling and causal inference for the World Model. 🔗
* 🎯 **Active Inference**: Behavioral decision-making mechanism based on Free Energy minimization. 📉
* 🔋 **Biological Drive**: A survival-driven system based on physiological needs like "Hunger" and "Battery". 🍔⚡

## 💻 Environment Requirements 📦

Make sure you have the following dependencies installed:
* Python 3.8+ 🐍
* `numpy`
* `torch`
* `pybullet` (for 3D Physics Simulation)
* `opencv-python`
* `visdom` (for Real-time Visualization)
* `nengo`
* `scipy`

## 🚀 Installation Guide 🔧

1. **Clone the repository**:
```bash
git clone <repository_url>
cd AION

```

2. **Install dependencies**:
```bash
pip install -r requirements.txt

```



## 🎮 How to Run 🏃‍♂️

### 1. Start the Visualization Server 📈

We use Visdom for beautiful real-time monitoring! Open a terminal and start the server before running the agent:

```bash
python -m visdom.server

```

*👉 Visit `http://localhost:8097` in your browser to view the magical dashboard!*

### 2. Run the Agent 🤖

In a new terminal window, start the main agent script:

```bash
python scripts/run_agent.py

```

## 📂 File Structure 📁

* 📁 `src/`: Core Source Code 🧩
* 📄 `lsm.py`: Liquid State Machine implementation 🌊
* 📄 `mhn.py`: Hopfield Network memory system 🧠
* 📄 `environment_pybullet.py`: PyBullet simulation wrapper 🌍
* 📄 `dashboard.py`: Visdom visualization controller 📊
* 📄 `adapter.py`: Analog to HDC random projection adapter 🔄


* 📁 `scripts/`: Execution & Training Scripts 🚀
* 📄 `run_agent.py`: Main entry point for the Agent 🚁
* 📄 `pretrain_lsm.py`: LSM visual perception pre-training 🎓
* 📄 `pretrain_world_model.py`: HRR World Model motor babbling pre-training 🌐
* 📄 `train_goal_directed.py`: Curriculum learning for goal-directed behavior 🏆



## 🔄 Agent Lifecycle 🐣 ➡️ 🦅

Once started, the agent will automatically go through these amazing phases:

1. 🎯 **Goal Imprinting**: Quickly locates and learns goal features via visual tracking.
2. 🤸 **Motor Babbling**: Learns its body schema and action-result mappings through random exploratory movements.
3. 🛡️ **Active Inference Survival**: The ultimate survival task! The agent actively hunts for energy sources while avoiding crashes to keep its battery full. ⚡

---

*Happy Exploring with AION! Feel free to contribute and build smarter AI!* 🎉💬
