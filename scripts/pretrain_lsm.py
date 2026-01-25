"""
AION LSM 感知预训练脚本

第一阶段：让LSM学会区分目标可见/不可见
"""

import sys
import os
import numpy as np
import random

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.environment_pybullet import PyBulletEnv
from src.lsm import AION_LSM_Network
from src.adapter import RandomProjectionAdapter


class LSMPretrainer:
    """LSM感知预训练器"""
    
    def __init__(self):
        print("=== AION LSM 感知预训练 ===")
        self.env = PyBulletEnv(headless=True)
        self.lsm = AION_LSM_Network()
        self.adapter = RandomProjectionAdapter()
        
        # 训练数据收集
        self.data = []  # [(obs, label, position)]
        
    def collect_training_data(self, n_samples=500):
        """收集带标签的训练数据"""
        print(f"\n--- 收集训练数据 ({n_samples} 样本) ---")
        
        positions = [
            # 能看到目标的位置 (目标在 3,0,0.5)
            ([2.0, 0.0, 0.5], 0.0, "CENTER"),   # 正对目标
            ([2.0, -0.5, 0.5], 0.0, "RIGHT"),   # 目标在右边
            ([2.0, 0.5, 0.5], 0.0, "LEFT"),     # 目标在左边
            ([1.5, 0.0, 0.5], 0.0, "CENTER"),   # 更近
            ([2.5, 0.0, 0.5], 0.0, "CENTER"),   # 稍远
            # 看不到目标的位置
            ([0.0, 0.0, 0.5], 3.14, "NONE"),    # 背对目标
            ([0.0, 3.0, 0.5], 0.0, "NONE"),     # 侧面远离
            ([-2.0, 0.0, 0.5], 0.0, "NONE"),    # 目标在身后
        ]
        
        for i in range(n_samples):
            self.env.reset()
            
            # 随机选择位置
            pos, yaw, expected_position = random.choice(positions)
            
            # 添加一些噪声
            pos = [p + random.uniform(-0.3, 0.3) for p in pos]
            yaw += random.uniform(-0.3, 0.3)
            
            # 传送到位置并设置朝向
            self.env.teleport_agent(pos)
            self.env.yaw = yaw
            
            # 获取观察
            obs, _, _, _, _ = self.env.step(0)  # Hover to capture
            
            # 确定实际标签（通过绿色像素）
            green_channel = obs[:, :, 1]
            green_ratio = green_channel.sum() / (obs.shape[0] * obs.shape[1] * 255.0)
            
            is_visible = green_ratio > 0.005
            
            # 确定位置标签
            if not is_visible:
                position_label = "NONE"
            else:
                green_mask = green_channel > 50
                if green_mask.sum() > 0:
                    cols = np.arange(obs.shape[1])
                    center_x = (green_mask.sum(axis=0) * cols).sum() / green_mask.sum()
                    width = obs.shape[1]
                    if center_x < width * 0.35:
                        position_label = "LEFT"
                    elif center_x > width * 0.65:
                        position_label = "RIGHT"
                    else:
                        position_label = "CENTER"
                else:
                    position_label = "CENTER"
            
            self.data.append({
                'obs': obs.copy(),
                'visible': is_visible,
                'position': position_label,
                'green_ratio': green_ratio
            })
            
            if (i + 1) % 50 == 0:
                visible_count = sum(1 for d in self.data if d['visible'])
                print(f"  收集 {i+1}/{n_samples} | 可见: {visible_count}, 不可见: {len(self.data) - visible_count}")
        
        # 统计
        visible_count = sum(1 for d in self.data if d['visible'])
        print(f"\n数据统计: 可见={visible_count}, 不可见={len(self.data) - visible_count}")
        
        position_counts = {}
        for d in self.data:
            p = d['position']
            position_counts[p] = position_counts.get(p, 0) + 1
        print(f"位置分布: {position_counts}")
        
    def supervised_train_lsm(self, epochs=10):
        """监督训练：用线性读出层分类LSM激活"""
        print(f"\n--- 监督训练 (带读出层, {epochs} epochs) ---")
        
        if not self.data:
            print("错误：没有训练数据！")
            return
        
        # 初始化读出层权重 (LSM输出 -> 4类)
        n_lsm_neurons = 500  # 假设LSM有500个神经元
        n_classes = 4
        
        # Xavier初始化
        W_readout = np.random.randn(n_lsm_neurons, n_classes) * np.sqrt(2.0 / n_lsm_neurons)
        b_readout = np.zeros(n_classes)
        
        learning_rate = 0.01
        position_to_idx = {"NONE": 0, "LEFT": 1, "CENTER": 2, "RIGHT": 3}
        idx_to_position = {0: "NONE", 1: "LEFT", 2: "CENTER", 3: "RIGHT"}
        
        for epoch in range(epochs):
            total_loss = 0.0
            correct = 0
            
            random.shuffle(self.data)
            
            for sample in self.data:
                obs = sample['obs']
                target_idx = position_to_idx[sample['position']]
                
                # 前向传播：获取LSM激活
                spikes = self.lsm.step(obs, dopamine=0.0)
                
                # 确保spikes长度正确
                if len(spikes) < n_lsm_neurons:
                    spikes = np.pad(spikes, (0, n_lsm_neurons - len(spikes)))
                else:
                    spikes = spikes[:n_lsm_neurons]
                
                # 读出层：线性分类
                logits = np.dot(spikes, W_readout) + b_readout
                
                # Softmax
                exp_logits = np.exp(logits - logits.max())
                probs = exp_logits / exp_logits.sum()
                
                # 预测
                pred_idx = np.argmax(probs)
                if pred_idx == target_idx:
                    correct += 1
                
                # 交叉熵损失
                loss = -np.log(probs[target_idx] + 1e-8)
                total_loss += loss
                
                # 反向传播：更新读出层
                # dL/d_logits = probs - one_hot(target)
                one_hot = np.zeros(n_classes)
                one_hot[target_idx] = 1.0
                d_logits = probs - one_hot
                
                # dL/dW = spikes.T @ d_logits, dL/db = d_logits
                dW = np.outer(spikes, d_logits)
                db = d_logits
                
                W_readout -= learning_rate * dW
                b_readout -= learning_rate * db
                
                # 同时用多巴胺调制LSM (弱信号)
                if pred_idx == target_idx:
                    dopamine = 0.1
                else:
                    dopamine = -0.1
                self.lsm.step(obs, dopamine=dopamine * 0.1)
            
            accuracy = correct / len(self.data)
            avg_loss = total_loss / len(self.data)
            print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2%}")
        
        # 保存读出层权重
        self.W_readout = W_readout
        self.b_readout = b_readout
        
        print("\n训练完成！")
        return accuracy
    
    def save_weights(self, path="pretrained_lsm.npz"):
        """保存训练后的权重"""
        print(f"保存权重到 {path}")
        np.savez(path, 
                 W_readout=self.W_readout, 
                 b_readout=self.b_readout)
        print("权重保存成功")
        

def main():
    trainer = LSMPretrainer()
    
    # 1. 收集数据 (增加到1000样本)
    trainer.collect_training_data(n_samples=1000)
    
    # 2. 训练 (增加到20 epochs)
    final_acc = trainer.supervised_train_lsm(epochs=20)
    
    # 3. 保存
    if final_acc > 0.6:
        trainer.save_weights()
    else:
        print("警告：准确率较低，可能需要更多数据或调参")
    
    print("\n=== 预训练结束 ===")
    

if __name__ == "__main__":
    main()
