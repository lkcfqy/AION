import sys
import os
import time
import torch
import numpy as np
import random

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.environment import AIONEnvironment
from src.lsm import AION_LSM_Network
from src.adapter import RandomProjectionAdapter
from src.mhn import ModernHopfieldNetwork
from src.hrr import HDCWorldModel
from src.gwt import GlobalWorkspace
from src.drive import BiologicalDrive
from src.controller import DroneController
from src.dashboard import AIONDashboard
from src.config import HDC_DIM, OBS_SHAPE

def generate_random_vector():
    vec = torch.randn(HDC_DIM)
    vec = torch.sign(vec)
    vec[vec == 0] = 1.0
    return vec

def hamming_dist(v1, v2):
    if v1 is None or v2 is None: return 1.0
    return (1.0 - torch.mean(v1 * v2).item()) / 2.0

class AIONAgent:
    def __init__(self):
        print("Initializing AION Agent Components...")
        
        # 1. Body & Senses
        self.env = AIONEnvironment()
        self.lsm = AION_LSM_Network()
        self.adapter = RandomProjectionAdapter()
        
        # 2. Mind
        self.mhn = ModernHopfieldNetwork() # Concept memory / Cleanup
        self.world_model = HDCWorldModel() # Causal inference
        
        # 3. Controller
        self.gwt = GlobalWorkspace()
        self.drive = BiologicalDrive()
        self.controller = DroneController()
        
        # 4. Actions (Motor Cortex)
        # Map 0, 1, 2 to HDC vectors
        self.action_vectors = {
            0: generate_random_vector(), # Hover
            1: generate_random_vector(), # Forward
            2: generate_random_vector(), # Rot_Left (Yaw)
            3: generate_random_vector(), # Rot_Right (Yaw)
            4: generate_random_vector(), # Up
            5: generate_random_vector()  # Down
        }
        from src.config import ACTION_NAMES
        self.action_names = ACTION_NAMES
        
        # Dashboard
        # DISABLED FOR HEADLESS DEBUGGING TO AVOID CONNECTION HANGS
        self.dashboard = None
        # try:
        #     self.dashboard = AIONDashboard()
        # except:
        #     self.dashboard = None
        #     print("Warning: Dashboard not available.")
            
        # Episodic Memory (Hippocampus)
        self.episodic_buffer = []
        
        # 搜索策略状态 (改进：交替旋转)
        self.search_direction = 2  # 2=RotL, 3=RotR
        self.search_steps = 0
        self.search_timeout = 20  # 每20步切换方向
        
        # 危险记忆 (避障)
        self.danger_memory = []
        
        # 加载预训练权重
        self.load_pretrained_models()

    def load_pretrained_models(self):
        """加载经过训练的世界模型和记忆"""
        print("尝试加载预训练模型...")
        if os.path.exists("pretrained_world_model.pt"):
            try:
                checkpoint = torch.load("pretrained_world_model.pt")
                if 'world_model' in checkpoint:
                     self.world_model.load_state_dict(checkpoint['world_model'])
                if 'mhn_memory' in checkpoint:
                     self.mhn.load_memory(checkpoint['mhn_memory'])
                print("✅ 成功加载预训练模型 (World Model & MHN)")
            except Exception as e:
                print(f"⚠️ 加载模型失败: {e}")
        else:
             print("⚠️ 未找到 pretrained_world_model.pt，智能体将作为'新生儿'运行。")

    def sleep(self, n_replay=50):
        """
        Sleep Consolidation Phase.
        Replay episodic memories to reinforce LSM weights.
        """
        print("\n--- PHASE: Sleep & Consolidation (Dreaming) ---")
        if not self.episodic_buffer:
            print("No memories to replay.")
            return
            
        # Shuffle memories to break temporal correlation (SWS style)
        memories = list(self.episodic_buffer)
        random.shuffle(memories)
        
        # Prioritized Replay: Sort by abs(dopamine) descending?
        # memories.sort(key=lambda x: abs(x[1]), reverse=True)
        
        count = 0
        for obs, dopamine in memories:
            if count >= n_replay: break
            
            # Only replay significant events?
            # If dopamine is near 0, maybe skip to save time?
            if abs(dopamine) < 0.01: continue
            
            # Activate LSM with Plasticity
            # We ignore the output hd_vector here, we just want delta_W
            self.lsm.step(obs, dopamine=dopamine)
            count += 1
            
        print(f"Consolidated {count} significant memories.")
        self.episodic_buffer.clear() # Clear Day's buffer


    def perceive(self, obs, dopamine=0.0):
        """Process observation to HDC Vector."""
        # 1. LSM Spike Response with Plasticity
        spikes = self.lsm.step(obs, dopamine=dopamine)
        
        # 2. Adapter Projection (Analog -> HDC)
        hdc_raw = self.adapter.forward(spikes)
        
        return hdc_raw
    
    def get_survival_mode(self):
        """
        根据电量决定行为模式 (生存驱动核心)
        
        Returns:
            str: "CRITICAL" | "HUNGRY" | "EXPLORE"
        """
        if self.drive.battery < 0.1:
            return "CRITICAL"  # 紧急求生 - 电量<10%
        elif self.drive.battery < 0.3:
            return "HUNGRY"    # 饥饿状态 - 电量<30%
        else:
            return "EXPLORE"   # 探索状态 - 电量>=30%
    
    def detect_goal_visible(self, obs):
        """
        通过绿色通道检测目标是否可见，并返回目标位置
        
        Args:
            obs: (56, 56, 3) numpy array, G通道是绿色目标掩码
        Returns:
            bool: 目标是否在视野中
            float: 绿色像素占比
            str: 目标位置 ("LEFT" / "CENTER" / "RIGHT" / "NONE")
        """
        green_channel = obs[:, :, 1]  # G通道是绿色目标掩码
        total_pixels = obs.shape[0] * obs.shape[1] * 255.0
        green_ratio = green_channel.sum() / total_pixels
        
        # 阈值: 0.5% 的像素是绿色即认为目标可见
        is_visible = green_ratio > 0.005
        
        if not is_visible:
            return False, green_ratio, "NONE"
        
        # 计算目标在图像中的X位置 (用于视觉伺服)
        # 找到绿色像素的质心X坐标
        green_mask = green_channel > 50  # 二值化
        if green_mask.sum() > 0:
            # 计算绿色区域的X质心
            cols = np.arange(obs.shape[1])
            weighted_sum = (green_mask.sum(axis=0) * cols).sum()
            total_green = green_mask.sum()
            center_x = weighted_sum / total_green
            
            # 图像宽度的1/3为左区，2/3为右区
            width = obs.shape[1]
            if center_x < width * 0.35:
                position = "LEFT"
            elif center_x > width * 0.65:
                position = "RIGHT"
            else:
                position = "CENTER"
        else:
            position = "CENTER"
            
        return is_visible, green_ratio, position

    def goal_imprinting(self):
        """Hack: Place agent at goal to learn Goal Vector."""
        print("--- PHASE: Goal Imprinting ---")
        print("Teleporting to Goal for fast learning...")
        # Goal is at (3, 0, 0.5) in PyBulletEnv. 
        # Teleport close to it: (2, 0, 0.5) to ensure we can reach it quickly
        
        self.env.reset()
        if hasattr(self.env, 'teleport_agent'):
             self.env.teleport_agent([2.0, 0.0, 0.5])
        
        # obs, info = self.env.reset(seed=None) # WRONG: This reverts teleport!
        
        # Manual capture via step(0)
        obs, _, _, _, _ = self.env.step(0) # Hover to get view
        
        while True:
            # Just hover/rotate to see it
            action = 2 # Rotate Left to scan if needed?
            # Or just forward if we are close.
            # Imprinting logic handles 'reward > 0'.
            # If we are at [4,5], goal at [5,5].
            # We step forward (1).
            action = 1 
            obs, reward, done, truncated, info = self.env.step(action)
            
            # Visualize
            if self.dashboard: self.dashboard.update_env_view(obs)
            
            # Process Senses
            concept = self.perceive(obs)
            
            if done and reward > 0:
                print("Goal Found! Imprinting...")
                self.gwt.set_goal(concept)
                
                # Add Goal to MHN Memory
                self.mhn.add_memory(concept)
                
                print("Goal Imprinted.")
                break
                
            if done or truncated:
                self.env.reset()
                self.lsm.reset() # Reset neural state too

    def motor_babbling(self, steps=200):
        """
        Phase 1.5: Motor Babbling (Body Schema Learning).
        Agent tries actions to learn the World Model: T(s, a) -> s'
        改进：增加采样量(200步)，使用轮询确保每个动作均匀采样。
        """
        print(f"--- PHASE: Motor Babbling ({steps} steps) ---")
        print("Learning Action-Consequence mappings...")
        
        # Teleport away from goal to open space for babbling
        if hasattr(self.env, 'teleport_agent'):
             self.env.teleport_agent([2.0, 2.0, 1.0])
             
        obs, _ = self.env.reset(seed=None)
        prev_concept = self.perceive(obs)
        
        for t in range(steps):
            # 改进：轮询而非随机，确保每个动作均匀学习
            action_int = t % 6
            action_hdc = self.action_vectors[action_int]
            
            # Execute
            # obs, reward, _, _, _ = self.env.step(action_int)
            smooth_vel = self.controller.step(action_int)
            obs, reward, _, _, _ = self.env.step(smooth_vel)
            if self.dashboard: self.dashboard.update_env_view(obs)
            
            # Learn
            current_concept = self.perceive(obs)
            
            # WM Update: s + a -> s' (使用action_int直接传递)
            self.world_model.learn(prev_concept, action_int, current_concept)
            
            prev_concept = current_concept
            
            if t % 20 == 0:
                print(f"Babbling Step {t}/{steps}: Action={self.action_names.get(action_int, str(action_int))}")
                
        print(f"Body Schema Learned. Transition counts: {self.world_model.counts}")
        
    def reset_for_mission(self):
        """Reset everything for the real run."""
        print("Resetting for Active Inference Survival Run...")
        self.env.reset()
        self.lsm.reset()
        self.controller.reset()
        # Start with Low Battery to trigger Homing Behavior immediately (Demo Mode)
        self.drive.battery = 0.3

    def run_active_inference(self, max_steps=1000):
        print("--- PHASE: Active Inference Loop ---")
        obs, info = self.env.reset()
        if self.dashboard: self.dashboard.update_env_view(obs)
        
        # Short memory of previous state for learning
        prev_concept = None
        prev_action_hdc = None
        prev_action_int = None
        
        # Running average of Free Energy for baseline
        avg_free_energy = 0.5 # Initial guess
        alpha_f = 0.05
        
        for t in range(max_steps):
            # 1. Perception
            # We need to calculate Dopamine signal BEFORE perception? 
            # Or based on previous step's result?
            # Dopamine reinforces the PREVIOUS action that led to CURRENT state.
            # But here we are reinforcing the LSM weights that process CURRENT input.
            # Actually, standard 3-factor rule reinforces the weights that were active during the behavior.
            # Here we apply it to sensory processing weights.
            # If F is high, we want to change how we see the world?
            # Or maybe just randomness.
            # Let's execute perception first.
            
            # 1. Calculate Dopamine based on previous step's change in Free Energy?
            # Or just instantaneous value vs baseline.
            # D = -(F_prev - avg_F) ? 
            # Let's say we use instantaneous F of *previous* step vs baseline.
            
            # Problem: We haven't computed current F yet.
            # So we pass dopamine=0 for first step, and delayed dopamine later.
            
            # Let's assume prediction error from LAST step determines dopamine NOW.
            # We need to track F from last step.
            
            if t == 0:
                dopamine = 0.0
                last_f = 0.5
            else:
                # Calculate Dopamine
                # If F is lower than baseline, good.
                # D = (avg_F - last_f) * scale
                dopamine = (avg_free_energy - last_f) * 5.0 # Scale up for learning rate
                # Update baseline
                avg_free_energy += alpha_f * (last_f - avg_free_energy)
                
            # Perception with Plasticity
            current_concept = self.perceive(obs, dopamine=dopamine)
            self.gwt.update_sense(current_concept)
            
            # 1.5 Online Concept Learning (Store novel states to MHN)
            # Retrieve closest memory
            if self.mhn.memory_count > 0:
                closest_mem = self.mhn.retrieve(current_concept)
                dist = hamming_dist(current_concept, closest_mem)
                # If novel enough, add to memory
                if dist > 0.1: 
                    # print(f"New Concept Discovered! (Dist: {dist:.3f})")
                    self.mhn.add_memory(current_concept)
            else:
                self.mhn.add_memory(current_concept)
            
            # 2. Learning (World Model)
            # T(S_prev, A_prev) -> S_curr
            if prev_concept is not None:
                self.world_model.learn(prev_concept, prev_action_int, current_concept)
                
            # 3. 生存驱动决策系统
            mode = self.get_survival_mode()
            goal_visible, green_ratio, goal_position = self.detect_goal_visible(obs)
            goal_delta = self.gwt.compute_goal_delta()
            
            # 4. Planning / Action Selection (基于生存模式)
            best_action_int = 0  # 默认: Hover
            pred_concept = current_concept
            
            if mode == "CRITICAL":
                # ═══════════════════════════════════════════════
                # 紧急求生模式 (电量<10%)
                # 策略: 视觉伺服 - 对准目标再冲刺
                # ═══════════════════════════════════════════════
                if goal_visible:
                    # 目标可见 → 视觉伺服导航
                    if goal_position == "LEFT":
                        best_action_int = 2  # RotL - 向左转对准目标
                        print(f"🔴 CRITICAL | Goal on LEFT (green={green_ratio:.4f}) → RotL to center")
                    elif goal_position == "RIGHT":
                        best_action_int = 3  # RotR - 向右转对准目标
                        print(f"🔴 CRITICAL | Goal on RIGHT (green={green_ratio:.4f}) → RotR to center")
                    else:  # CENTER
                        best_action_int = 1  # Forward - 目标居中，全速前进
                        print(f"🔴 CRITICAL | Goal CENTERED (green={green_ratio:.4f}) → SPRINT!")
                else:
                    # 搜索策略: 旋转5步 → 前进3步
                    self.search_steps += 1
                    cycle_pos = self.search_steps % 8
                    
                    if cycle_pos < 5:
                        best_action_int = self.search_direction
                        print(f"🔴 CRITICAL | SCANNING (green={green_ratio:.4f}) → {'RotL' if best_action_int == 2 else 'RotR'}")
                    else:
                        best_action_int = 1
                        print(f"🔴 CRITICAL | EXPLORING (green={green_ratio:.4f}) → Forward")
                    
                    if self.search_steps % 16 == 0:
                        self.search_direction = 3 if self.search_direction == 2 else 2
                    
            elif mode == "HUNGRY":
                # ═══════════════════════════════════════════════
                # 饥饿模式 (电量10%-30%)
                # 策略: 视觉伺服导航 (简化版Active Inference)
                # ═══════════════════════════════════════════════
                if goal_visible:
                    # 目标可见 → 视觉伺服导航
                    if goal_position == "LEFT":
                        best_action_int = 2  # RotL
                        print(f"🟡 HUNGRY | Goal on LEFT (green={green_ratio:.4f}) → RotL")
                    elif goal_position == "RIGHT":
                        best_action_int = 3  # RotR
                        print(f"🟡 HUNGRY | Goal on RIGHT (green={green_ratio:.4f}) → RotR")
                    else:  # CENTER
                        best_action_int = 1  # Forward
                        print(f"🟡 HUNGRY | Goal CENTERED (green={green_ratio:.4f}) → Forward")
                else:
                    # 目标不可见 → 旋转+前进交替搜索
                    self.search_steps += 1
                    cycle_pos = self.search_steps % 10
                    
                    if cycle_pos < 6:
                        best_action_int = self.search_direction
                        print(f"🟡 HUNGRY | SCANNING (green={green_ratio:.4f}) → {'RotL' if best_action_int == 2 else 'RotR'}")
                    else:
                        best_action_int = 1
                        print(f"🟡 HUNGRY | EXPLORING (green={green_ratio:.4f}) → Forward")
                    
                    if self.search_steps % 20 == 0:
                        self.search_direction = 3 if self.search_direction == 2 else 2
                    
            else:  # EXPLORE
                # ═══════════════════════════════════════════════
                # 探索模式 (电量>=30%)
                # 策略: 主动推理 (Active Inference)
                # 1. 想象所有可能的动作结果 (World Model)
                # 2. 计算预期自由能 G (Expected Free Energy)
                # 3. 选择 G 最小的动作 (最大化效用/最小化目标距离)
                # ═══════════════════════════════════════════════
                
                # print(f"🟢 EXPLORE | Active Inference Planning...")
                best_G = float('inf')
                best_action_int = 0 # Default Hover
                
                # 简单的 1-step Lookahead
                for a in range(6): # Try all 6 actions
                    # 1. 想象: s_next = T(s_curr, a)
                    imagined_concept = self.world_model.predict(current_concept, a)
                    
                    # 2. 评估: G ≈ -Utility (Goal Delta) - InfoGain + Cost
                    # 简化: G = DistanceToGoal + ActionCost
                    # 我们希望 Distance 最小
                    
                    dist_to_goal = self.gwt._hamming_dist(imagined_concept, self.gwt.current_goal)
                    
                    # Action Cost (Energy efficiency)
                    # Hover(0) is cheapest, Up(4) is expensive?
                    cost = 0.0
                    if a == 0: cost = 0.0
                    elif a == 4: cost = 0.05
                    else: cost = 0.01
                    
                    G = dist_to_goal + cost
                    
                    if G < best_G:
                        best_G = G
                        best_action_int = a
                
                # 添加一点随机性以防死循环 (Softmax 策略会更好，这里用 Epsilon-Greedy)
                if random.random() < 0.1:
                    best_action_int = random.randint(0, 5)
                    # print("  (Random Exploration)")
                
                # print(f"🟢 EXPLORE | Best Action: {self.action_names.get(best_action_int)} (G={best_G:.4f})")
            
            # 5. Execute
            # obs, reward, done, truncated, info = self.env.step(best_action_int)
            smooth_vel = self.controller.step(best_action_int)
            obs, reward, done, truncated, info = self.env.step(smooth_vel)
            
            # CRASH REFLEX LOGIC
            is_crash = info.get('crash', False)
            if is_crash:
                # PAIN SIGNAL + 存储危险记忆
                dopamine = -1.0 
                self.danger_memory.append(current_concept.clone())
                print(f"CRASH DETECTED! Danger memory size: {len(self.danger_memory)}")
            
            # 6. Update Drive & GWT
            # Station Keeping / Charging logic: if reward > 0 (at charger), battery fills
            got_food = (reward > 0)
            self.drive.step(got_food=got_food)
            self.gwt.update_pred(pred_concept)
            
            # At end of loop, calculate F for next step's dopamine
            surprise = self.gwt.compute_surprise()
            status = self.gwt.get_status()
            current_f = self.drive.compute_free_energy(surprise, goal_delta)
            
            if self.dashboard:
                self.dashboard.update_env_view(obs)
                # Show Battery instead of just Hunger
                self.dashboard.update_survival(current_f, self.drive.hunger) 
                
                # Update LSM Raster
                # Get spikes from the LAST step of LSM
                spikes = self.lsm.last_spikes
                active_indices = np.where(spikes > 0)[0]
                self.dashboard.update_lsm_raster(active_indices)
                
                # Update Energy Landscape
                energy = self.mhn.compute_energy(current_concept)
                self.dashboard.update_energy(energy)
                
                # Update HDC Similarity (Current vs Recalled)
                # How stable/familiar is the current concept?
                recalled = self.mhn.retrieve(current_concept)
                dist = hamming_dist(current_concept, recalled)
                sim = 1.0 - (2.0 * dist)
                self.dashboard.update_hdc_similarity(sim)
            
            # Log every step for debugging
            if True:
                print(f"Step {t} | Act: {self.action_names.get(best_action_int, str(best_action_int))} | "
                      f"Hun: {self.drive.hunger:.2f} | Bat: {self.drive.battery:.2f} | "
                      f"F: {current_f:.2f} | AvgF: {avg_free_energy:.2f} | Dop: {dopamine:.3f}")

            # Store history
            prev_concept = current_concept
            prev_action_hdc = self.action_vectors[best_action_int]
            prev_action_int = best_action_int
            last_f = current_f
            
            # Store Experience in Episodic Buffer
            # We store (Observation, Dopamine) tuple
            # Note: We store the OBS that CAUSED the dopamine (or concurrent with it).
            self.episodic_buffer.append((obs, dopamine))
            
            if done:
                print("🌟 Goal Reached! (Charging Station Docked)")
                print(f"   Battery recharged: {self.drive.battery:.2f} -> 1.00")
                
                # Trigger Sleep Consolidation
                self.sleep(n_replay=200)
                
                print("Resetting Env for next Day...")
                self.env.reset()
                self.lsm.reset()
                prev_concept = None 
                if truncated: break
            
            # 💀 电量耗尽 = 死亡
            if self.drive.battery <= 0:
                print("💀 DEATH: Battery depleted! Agent crashed.")
                print(f"   Survived {t} steps.")
                break

def main():
    print("DEBUG: Script Started", flush=True)
    agent = AIONAgent()
    
    # 1. Imprint Goal
    agent.goal_imprinting()
    
    # 2. Learn Body Schema (Warmup) - 增加采样量
    agent.motor_babbling(steps=200)
    
    # 3. Setup Mission
    agent.reset_for_mission()
    
    # 4. Run Survival
    agent.run_active_inference()

if __name__ == "__main__":
    main()
