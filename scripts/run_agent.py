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
        
        # 4. Actions (Motor Cortex)
        # Map 0, 1, 2 to HDC vectors
        self.action_vectors = {
            0: generate_random_vector(), # Left
            1: generate_random_vector(), # Right
            2: generate_random_vector()  # Forward
        }
        self.action_names = {0: "Left", 1: "Right", 2: "Fwd"}
        
        # Dashboard
        try:
            self.dashboard = AIONDashboard()
        except:
            self.dashboard = None
            print("Warning: Dashboard not available.")
            
        # Episodic Memory (Hippocampus)
        self.episodic_buffer = []

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

    def goal_imprinting(self):
        """Hack: Place agent at goal to learn Goal Vector."""
        print("--- PHASE: Goal Imprinting ---")
        print("Exploring to find goal...")
        obs, info = self.env.reset()
        
        while True:
            # Random walk (Navigation only)
            action = random.choice([0, 1, 2])
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

        # Reset for actual run
        print("Resetting for Active Inference Run...")
        self.env.reset()
        self.lsm.reset()
        self.drive.step(got_food=True) # Start full

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
                self.world_model.learn(prev_concept, prev_action_hdc, current_concept)
                
            # 3. Drive Check
            # Check if we accidentally accomplished goal
            goal_delta = self.gwt.compute_goal_delta()
            
            # 4. Planning / Action Selection
            best_action_int = None
            min_expected_energy = float('inf')
            
            # Exploration factor (Temperature) based on hunger?
            epsilon = 0.3 # Fixed for now
            
            if np.random.rand() < epsilon:
                 best_action_int = np.random.choice([0, 1, 2])
                 # Get pred just for GWT update
                 pred_concept = self.world_model.predict(current_concept, self.action_vectors[best_action_int])
            else:
                # Rational Choice
                best_action_int = 0 # Default
                
                for a_int in [0, 1, 2]: # Left, Right, Forward
                    a_hdc = self.action_vectors[a_int]
                    
                    # Predict Outcome
                    # Pred = WM(Sense, Action)
                    pred_raw = self.world_model.predict(current_concept, a_hdc)
                    
                    # Cleanup Prediction using MHN (Recall known concepts)
                    if self.mhn.memory_count > 0:
                        pred_clean = self.mhn.retrieve(pred_raw)
                    else:
                        pred_clean = pred_raw
                        
                    # Calculate Expected Cost (Distance to Goal)
                    # We utilize the GWT's helper, but need to pass vectors manually
                    dist = hamming_dist(pred_clean, self.gwt.current_goal)
                    
                    # Cost = GoalDist (Greedy)
                    if dist < min_expected_energy:
                        min_expected_energy = dist
                        best_action_int = a_int
                        pred_concept = pred_clean # Store for visualization
            
            # 5. Execute
            obs, reward, done, truncated, info = self.env.step(best_action_int)
            
            # 6. Update Drive & GWT
            self.drive.step(got_food=(reward > 0))
            self.gwt.update_pred(pred_concept)
            
            # At end of loop, calculate F for next step's dopamine
            surprise = self.gwt.compute_surprise()
            status = self.gwt.get_status()
            current_f = self.drive.compute_free_energy(surprise, goal_delta)
            
            if self.dashboard:
                self.dashboard.update_env_view(obs)
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
            
            # Log
            if t % 10 == 0:
                print(f"Step {t} | Act: {self.action_names[best_action_int]} | "
                      f"Hun: {self.drive.hunger:.2f} | F: {current_f:.2f} | AvgF: {avg_free_energy:.2f} | Dop: {dopamine:.3f}")

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
                print("Goal Reached! (Food Eaten).")
                
                # Trigger Sleep Consolidation
                self.sleep(n_replay=200)
                
                print("Resetting Env for next Day...")
                self.env.reset()
                self.lsm.reset()
                prev_concept = None 
                # Reset F baseline?
                # avg_free_energy = 0.5
                if truncated: break

def main():
    agent = AIONAgent()
    
    # 1. Imprint Goal
    agent.goal_imprinting()
    
    # 2. Run
    agent.run_active_inference()

if __name__ == "__main__":
    main()
