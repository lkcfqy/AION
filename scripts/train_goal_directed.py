"""
AION 目标导向学习训练脚本 (Phase 3) - 优化版

功能：
1. 加载预训练的LSM和世界模型
2. 实现密集奖励机制 (Dense Rewards - 改进版)
3. 实现课程学习 (Curriculum Learning - 循环重试)
4. 训练策略/微调模型

优化点：
- 降低LSM神经元数量 (1000 -> 400) 提升CPU速度
- 引入视觉对齐奖励 (Visual Centering Reward)
- 改进 Level 2 搜索逻辑 (Systematic Scan)
"""

import sys
import os
import numpy as np
import torch
import random
import time

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.environment_pybullet import PyBulletEnv
from src.lsm import AION_LSM_Network
from src.adapter import RandomProjectionAdapter
from src.hrr import HDCWorldModel
from src.mhn import ModernHopfieldNetwork
from src.config import HDC_DIM

class GoalDirectedTrainer:
    def __init__(self, load_pretrained=True):
        print("=== AION 目标导向学习 (Phase 3) - 优化版 ===")
        
        # 1. 初始化环境和模型
        self.env = PyBulletEnv(headless=True)
        self.lsm = AION_LSM_Network()
        self.adapter = RandomProjectionAdapter()
        self.world_model = HDCWorldModel()
        self.mhn = ModernHopfieldNetwork()
        
        self.action_names = {
            0: "Hover", 1: "Forward", 2: "Rotate_Left",
            3: "Rotate_Right", 4: "Up", 5: "Down"
        }
        self.n_actions = 6
        
        # 2. 加载预训练权重
        if load_pretrained:
            self.load_pretrained_models()
            
    def load_pretrained_models(self):
        """加载 Phase 1 & 2 的预训练权重"""
        print("\n正在加载预训练模型...")
        
        # 加载 LSM 读出层 (可选)
        lsm_path = "pretrained_lsm.npz"
        if os.path.exists(lsm_path):
            data = np.load(lsm_path)
            self.W_readout = data['W_readout']
            self.b_readout = data['b_readout']
            print(f"✅ 加载 LSM 权重: {lsm_path}")
        else:
            print(f"⚠️ 未找到 LSM 权重: {lsm_path} (将使用随机初始化)")
            
        # 加载 World Model & MHN
        wm_path = "pretrained_world_model.pt"
        if os.path.exists(wm_path):
            state_dict = torch.load(wm_path)
            if 'world_model' in state_dict:
                self.world_model.load_state_dict(state_dict['world_model'])
            if 'mhn_memory' in state_dict:
                self.mhn.load_memory(state_dict['mhn_memory'])
            print(f"✅ 加载 World Model & MHN: {wm_path}")
        else:
            print(f"⚠️ 未找到 World Model 权重: {wm_path}")
            
    def compute_reward(self, dist, prev_dist, goal_visible, center_x_normalized=0.5):
        """
        密集奖励函数 (改进版)
        """
        reward = 0.0
        
        # 1. 进度奖励 (Progress)
        delta_dist = prev_dist - dist
        reward += delta_dist * 20.0  
            
        # 2. 视觉引导奖励 (Visual Centering)
        if goal_visible:
            # 越靠近中心，得分越高 (中心分布 0.5)
            centering = 0.5 - abs(center_x_normalized - 0.5)
            reward += centering * 2.0
            
        # 3. 到达奖励 (Arrival)
        if dist < 0.5:
            reward += 100.0
            
        # 4. 生存惩罚
        reward -= 0.02
        
        return reward

    def perceive(self, obs):
        """感知：图像 -> HDC向量"""
        spikes = self.lsm.step(obs, dopamine=0.0)
        hdc = self.adapter.forward(spikes)
        return hdc, spikes
    
    def detect_goal_simple(self, obs):
        """简单颜色检测"""
        green_channel = obs[:, :, 1]
        total = obs.shape[0] * obs.shape[1] * 255.0
        ratio = green_channel.sum() / total
        visible = ratio > 0.005
        return visible, ratio

    def train_curriculum(self):
        """课程学习主循环 (带重试逻辑)"""
        levels = [
            {"name": "Level 1: 直线距离 2m", "dist_range": (1.8, 2.2), "obstacle": False},
            {"name": "Level 2: 随机距离 1-4m", "dist_range": (1.0, 4.0), "obstacle": False},
            {"name": "Level 3: 含障碍物", "dist_range": (2.0, 4.0), "obstacle": True},
        ]
        
        level_idx = 0
        while level_idx < len(levels):
            level = levels[level_idx]
            print(f"\n========================================")
            print(f"开始课程 {level['name']}")
            print(f"========================================")
            
            success_count = 0
            total_episodes = 20
            
            for episode in range(total_episodes):
                self.env.reset()
                
                dist = random.uniform(*level['dist_range'])
                angle = random.uniform(0, 2*np.pi)
                goal_pos = np.array([3.0, 0.0, 0.5])
                agent_x = goal_pos[0] - dist * np.cos(angle)
                agent_y = goal_pos[1] - dist * np.sin(angle)
                
                self.env.reset() 
                
                # 初始朝向
                yaw_noise = 0.5 if level_idx == 0 else 3.14
                desired_yaw = np.arctan2(goal_pos[1] - agent_y, goal_pos[0] - agent_x)
                start_yaw = desired_yaw + random.uniform(-yaw_noise, yaw_noise)
                
                self.env.teleport_agent([agent_x, agent_y, 0.5], yaw=start_yaw)
                obs = self.env._get_observation()
                
                prev_dist = dist
                total_reward = 0
                steps = 0
                max_steps = 2000
                done = False
                prev_concept = None
                prev_action = None
                
                print(f"Episode {episode+1}/{total_episodes} | Dist: {dist:.2f}m")
                
                while not done and steps < max_steps:
                    steps += 1
                    current_concept, spikes = self.perceive(obs)
                    goal_visible, _ = self.detect_goal_simple(obs)
                    
                    action = 0
                    green_channel = obs[:, :, 1]
                    green_mask = green_channel > 50
                    center_x_norm = 0.5
                    
                    if green_mask.sum() > 0:
                        cols = np.arange(obs.shape[1])
                        center_x = (green_mask.sum(axis=0) * cols).sum() / green_mask.sum()
                        center_x_norm = center_x / obs.shape[1]
                        
                        if center_x_norm < 0.4:
                            action = 2 # Left
                        elif center_x_norm > 0.6:
                            action = 3 # Right
                        else:
                            action = 1 # Forward
                    else:
                        # 没看到目标，系统化旋转搜索
                        cycle = steps % 40
                        if cycle < 30:
                            action = 2 
                        else:
                            action = 1
                            
                    next_obs, env_reward, terminated, truncated, info = self.env.step(action)
                    curr_pos = np.array(self.env.get_pos())
                    curr_dist = np.linalg.norm(curr_pos - goal_pos)
                    
                    dense_reward = self.compute_reward(curr_dist, prev_dist, goal_visible, center_x_norm)
                    total_reward += dense_reward
                    
                    if prev_concept is not None:
                        self.world_model.learn(prev_concept, prev_action, current_concept)
                    
                    prev_dist = curr_dist
                    prev_concept = current_concept
                    prev_action = action
                    obs = next_obs
                    
                    if curr_dist < 0.5:
                        print(f"  ✅ 成功到达目标! Reward: {total_reward:.2f}")
                        success_count += 1
                        done = True
                        
                if not done:
                    print(f"  ❌ 超时/失败. Reward: {total_reward:.2f}")
                    
            success_rate = success_count / total_episodes
            print(f"课程 {level['name']} 完成率: {success_rate:.2%}")
            
            if success_rate < 0.7:
                print("⚠️ 表现不佳，重试当前难度...")
                continue 
            else:
                print("🎉 晋级下一难度！")
                level_idx += 1
                
            # Save models after each level
            self.save_models()
            
    def save_models(self):
        """保存模型权重"""
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        print(f"正在保存模型权重 ({timestamp})...")
        
        # 1. Save World Model & MHN
        torch.save({
            'world_model': self.world_model.M_per_action, # Save the list of tensors directly? Or state dict
            # HDCWorldModel structure changed slightly in our fix. 
            # We should probably add a proper state_dict method to HDCWorldModel if not exists, 
            # or just save the list.
            # Let's save the list for simplicity as we implemented load_state_dict to take list.
            'mhn_memory': self.mhn.memory_matrix
        }, "pretrained_world_model.pt")
        
        # 2. Save LSM Readout (If we had one training)
        # In this script we don't train readout explicitly (LSM is fixed reservoir).
        # But if we did, we'd save it.
        print("✅ 模型已保存到 pretrained_world_model.pt")

def main():
    trainer = GoalDirectedTrainer(load_pretrained=True)
    trainer.train_curriculum()

if __name__ == "__main__":
    main()
