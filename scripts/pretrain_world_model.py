"""
AION 世界模型预训练脚本

第二阶段：大量Motor Babbling训练HRR世界模型
"""

import sys
import os
import numpy as np
import random

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.environment_pybullet import PyBulletEnv
from src.lsm import AION_LSM_Network
from src.adapter import RandomProjectionAdapter
from src.hrr import HDCWorldModel
from src.mhn import ModernHopfieldNetwork
from src.config import HDC_DIM


class WorldModelPretrainer:
    """世界模型预训练器"""
    
    def __init__(self):
        print("=== AION 世界模型预训练 ===")
        self.env = PyBulletEnv(headless=True)
        self.lsm = AION_LSM_Network()
        self.adapter = RandomProjectionAdapter()
        self.world_model = HDCWorldModel()
        self.mhn = ModernHopfieldNetwork()
        
        # 动作定义
        self.action_names = {
            0: "Hover", 1: "Forward", 2: "Rotate_Left",
            3: "Rotate_Right", 4: "Up", 5: "Down"
        }
        self.n_actions = 6
        
    def perceive(self, obs):
        """感知：图像 -> HDC向量"""
        spikes = self.lsm.step(obs, dopamine=0.0)
        hdc = self.adapter.forward(spikes)
        return hdc
    
    def run_motor_babbling(self, steps=5000):
        """大量随机动作学习世界模型"""
        print(f"\n--- Motor Babbling ({steps} 步) ---")
        print("学习动作-结果映射...")
        
        self.env.reset()
        obs, _ = self.env.reset()
        
        prev_concept = None
        prev_action_int = None
        
        # 统计每个动作的学习次数
        action_counts = {i: 0 for i in range(self.n_actions)}
        
        for step in range(steps):
            # 轮询采样确保每个动作被均匀学习
            action_int = step % self.n_actions
            action_counts[action_int] += 1
            
            # 感知当前状态
            current_concept = self.perceive(obs)
            
            # 如果有前一步，学习转移
            if prev_concept is not None:
                self.world_model.learn(prev_concept, prev_action_int, current_concept)
            
            # 存储新概念到MHN
            if self.mhn.memory_count < 200:  # 限制记忆数量
                if self.mhn.memory_count == 0:
                    self.mhn.add_memory(current_concept)
                else:
                    closest = self.mhn.retrieve(current_concept)
                    dist = self._hamming_dist(current_concept, closest)
                    if dist > 0.1:  # 足够新颖
                        self.mhn.add_memory(current_concept)
            
            # 执行动作
            obs, reward, done, truncated, info = self.env.step(action_int)
            
            # 如果碰撞或完成，重置
            if done or info.get('crash', False):
                self.env.reset()
                prev_concept = None
                prev_action_int = None
                continue
            
            prev_concept = current_concept
            prev_action_int = action_int
            
            # 进度报告
            if (step + 1) % 500 == 0:
                print(f"  步数 {step+1}/{steps} | MHN记忆: {self.mhn.memory_count}")
        
        print(f"\n动作分布: {action_counts}")
        print(f"MHN最终记忆数: {self.mhn.memory_count}")
        
    def _hamming_dist(self, a, b):
        """计算Hamming距离"""
        return (a != b).float().mean().item()
    
    def verify_world_model(self):
        """验证世界模型预测能力"""
        print("\n--- 验证世界模型 ---")
        
        self.env.reset()
        obs, _ = self.env.reset()
        
        # 收集一些测试样本
        test_results = {a: [] for a in range(self.n_actions)}
        
        for _ in range(100):
            # 随机位置
            pos = [random.uniform(-2, 2), random.uniform(-2, 2), 0.5]
            self.env.teleport_agent(pos)
            self.env.yaw = random.uniform(0, 6.28)
            
            obs, _, _, _, _ = self.env.step(0)  # Hover获取当前观察
            current = self.perceive(obs)
            
            for action in range(self.n_actions):
                # 预测
                predicted = self.world_model.predict(current, action)
                
                # 实际执行
                obs_after, _, _, _, _ = self.env.step(action)
                actual = self.perceive(obs_after)
                
                # 比较
                pred_clean = self.mhn.retrieve(predicted) if self.mhn.memory_count > 0 else predicted
                dist = self._hamming_dist(pred_clean, actual)
                test_results[action].append(dist)
                
                # 重置到原位置
                self.env.teleport_agent(pos)
                self.env.yaw = random.uniform(0, 6.28)
                obs, _, _, _, _ = self.env.step(0)
        
        # 报告
        print("\n预测误差 (Hamming距离, 越低越好)：")
        for action, dists in test_results.items():
            avg_dist = np.mean(dists)
            print(f"  {self.action_names[action]}: {avg_dist:.4f}")
        
        overall = np.mean([np.mean(d) for d in test_results.values()])
        print(f"\n整体平均误差: {overall:.4f}")
        
        if overall < 0.4:
            print("✅ 世界模型预测能力良好")
        else:
            print("⚠️ 世界模型预测能力需要改进")
        
        return overall


    def save_weights(self, path="pretrained_world_model.pt"):
        """保存世界模型和MHN"""
        import torch
        print(f"\n保存模型权重到 {path}")
        
        state_dict = {
            "world_model": self.world_model.M_per_action,
            "mhn_memory": self.mhn.memory_matrix
        }
        
        torch.save(state_dict, path)
        print("模型保存成功")
def main():
    trainer = WorldModelPretrainer()
    
    # 1. 大量Motor Babbling
    trainer.run_motor_babbling(steps=5000)
    
    # 2. 验证
    error = trainer.verify_world_model()

    # 3. 保存
    trainer.save_weights()
    
    print("\n=== 世界模型预训练结束 ===")
    

if __name__ == "__main__":
    main()
