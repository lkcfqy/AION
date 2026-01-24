import sys
import os
import numpy as np
import cv2
import math
import random

# Add project root
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.config import OBS_SHAPE
from scripts.run_agent import AIONAgent

class MockIsaacEnv:
    """
    Mock Environment for AION.
    Simulates a 10x10m room with a Green Charging Pad at (8, 8).
    Replaces the heavy Nvidia Isaac Sim for logic verification.
    """
    def __init__(self):
        self.room_size = 10.0 # meters
        self.agent_pos = np.array([5.0, 5.0]) # Start center
        self.agent_dir = 0.0 # Radians (0 = East/Right)
        
        # Goal: Green Charging Pad
        self.goal_pos = np.array([8.0, 8.0])
        self.goal_radius = 1.0
        
        # Mental Visualizer Canvas
        self.canvas_size = 500
        
    def reset(self):
        self.agent_pos = np.array([1.0, 1.0]) # Start corner
        self.agent_dir = np.pi / 4 # Facing diagonal
        return self._render_view(), {}

    def step(self, action_id):
        """
        Mock Physics Engine.
        0: Hover, 1: Fwd, 2: RotL, 3: RotR
        """
        dt = 0.1
        speed = 1.0
        rot_speed = 0.5
        
        # Kinematics
        if action_id == 1: # Forward
            dx = math.cos(self.agent_dir) * speed * dt
            dy = math.sin(self.agent_dir) * speed * dt
            self.agent_pos += np.array([dx, dy])
        elif action_id == 2: # RotL
            self.agent_dir += rot_speed * dt
        elif action_id == 3: # RotR
            self.agent_dir -= rot_speed * dt
            
        # Bound to room
        self.agent_pos = np.clip(self.agent_pos, 0, self.room_size)
        
        # Check Goal (Collision)
        dist = np.linalg.norm(self.agent_pos - self.goal_pos)
        reward = 0.0
        done = False
        
        if dist < 0.5:
            reward = 1.0
            done = True
            
        return self._render_view(), reward, done, False, {}

    def _render_view(self):
        """
        Generate Synthetic RGB Image based on relative position.
        """
        # Create a blank image (Retina)
        img = np.zeros((OBS_SHAPE[0], OBS_SHAPE[1], 3), dtype=np.uint8)
        
        # Calculate Goal position in Agent's local frame
        # Transform global goal to local
        dx = self.goal_pos[0] - self.agent_pos[0]
        dy = self.goal_pos[1] - self.agent_pos[1]
        
        # Rotate by -agent_dir
        local_x = dx * math.cos(-self.agent_dir) - dy * math.sin(-self.agent_dir)
        local_y = dx * math.sin(-self.agent_dir) + dy * math.cos(-self.agent_dir)
        
        # If goal is in front (local_x > 0)
        if local_x > 0:
            # Perspective project (simple)
            # angle = atan2(y, x)
            angle = math.atan2(local_y, local_x)
            
            # FOV +/- 45 degrees
            if abs(angle) < np.pi / 4:
                # Project to screen x
                # screen_x from 0 to W
                # -pi/4 -> 0, +pi/4 -> W
                fov = np.pi / 2
                u = 0.5 - (angle / fov) # 0.5 is center
                px = int(u * OBS_SHAPE[1])
                
                # Size inversely prop to distance
                dist = max(math.sqrt(local_x**2 + local_y**2), 0.1)
                size = int(20.0 / dist)
                
                # Draw Green Square (Charging Pad)
                cv2.rectangle(img, (px-size, 20), (px+size, 40), (0, 255, 0), -1)
                
        # Add Noise
        noise = np.random.randint(0, 30, img.shape, dtype=np.uint8)
        img = cv2.add(img, noise)
        
        return img

def run_mock_mission():
    print("--- STARTING MOCK MISSION ---")
    print("Scenario: Low Battery Survival Run")
    
    # 1. Setup
    # Hijack the Agent's methods to use Mock Env
    agent = AIONAgent()
    agent.env = MockIsaacEnv() # Inject Mock Env
    agent.dashboard = None # Disable dashboard for headless test
    
    # 2. Imprint phase (Quickly)
    print("\nPhase 1: Goal Imprinting (Looking at Charging Pad)...")
    # Force agent to look at goal
    agent.env.agent_pos = np.array([7.0, 8.0])
    agent.env.agent_dir = 0.0 # Looking at (8,8)
    
    obs, _ = agent.env.reset() # This resets pos, let's force set again
    agent.env.agent_pos = np.array([7.0, 7.0])
    agent.env.goal_pos = np.array([8.0, 7.0]) # Goal is right in front
    agent.env.agent_dir = 0.0
    
    obs = agent.env._render_view()
    goal_concept = agent.perceive(obs)
    agent.gwt.set_goal(goal_concept)
    print("Goal Imprinted: 'Green Shape'")
    
    # 3. Mission Start
    print("\nPhase 2: Survival Run")
    agent.env.reset() # Random corner
    agent.drive.battery = 0.25 # CRITICAL BATTERY START!
    
    print(f"Start Status: Pos={agent.env.agent_pos}, Bat={agent.drive.battery}")
    
    for t in range(50):
        # reuse agent loop logic manually or call modified run?
        # Let's call a simplified loop step
        
        # 1. Perceive
        obs = agent.env._render_view()
        concept = agent.perceive(obs)
        agent.gwt.update_sense(concept)
        
        # 2. Drive Check
        agent.drive.step(got_food=False)
        goal_delta = agent.gwt.compute_goal_delta()
        surprise = agent.gwt.compute_surprise()
        f = agent.drive.compute_free_energy(surprise, goal_delta)
        
        # 3. Decision (Greedy for Goal if Battery Low)
        # Scan actions
        best_act = 0
        min_f = float('inf')
        
        for a in [1, 2, 3]: # Fwd, Left, Right
            # Predict
            pred = agent.world_model.predict(concept, agent.action_vectors[a])
            d = agent.gwt._hamming_dist(pred, agent.gwt.current_goal)
            
            # If Low Battery, Cost = GoalDist
            cost = d
            if cost < min_f:
                min_f = cost
                best_act = a
        
        # 4. Act
        _, reward, done, _, _ = agent.env.step(best_act)
        
        act_name = {1:"Fwd", 2:"Left", 3:"Right"}.get(best_act, "Hover")
        print(f"Step {t} | Act: {act_name} | Bat: {agent.drive.battery:.3f} | GoalDist: {goal_delta:.3f}")
        
        if done:
            print("SUCCESS: Docked at Charging Pad!")
            break
            
        if agent.drive.battery <= 0.0:
            print("FAILURE: Battery Depleted. Crashed.")
            break
            
if __name__ == "__main__":
    run_mock_mission()
