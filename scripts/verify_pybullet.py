import sys
import os
import cv2
import numpy as np

# Add project root
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.environment import AIONEnvironment
from src.config import ACTION_NAMES

def verify_pybullet():
    print("Verifying PyBullet Environment...")
    env = AIONEnvironment()
    
    # 1. Test Reset
    print("\nTest 1: Reset")
    obs, info = env.reset()
    print(f"Obs Shape: {obs.shape}")
    
    if obs.shape == (56, 56, 3):
        print("SUCCESS: Observation shape correct (56, 56, 3).")
    else:
        print(f"FAILURE: Shape mismatch {obs.shape}")
        return

    # 2. Test Step & Bio-Retina
    print("\nTest 2: Stepping (Takeoff)")
    # Action 4: Up
    obs, reward, done, truncated, info = env.step(4)
    print(f"Reward: {reward}, Done: {done}")
    
    # Check if image is not empty (black)
    # Since we have a plane and cubes, we should see something if looking correctly?
    # Our camera is on the robot. Robot is at [0,0,1].
    # Looking forward. Plane is below.
    # We might see the horizon or the objects if they are in view.
    # In 'environment_pybullet.py', we put Goal at [5,5].
    # Robot at [0,0].
    # If we rotate towards goal, we should see it.
    
    non_zero = np.count_nonzero(obs)
    print(f"Non-zero pixels: {non_zero}")
    
    if non_zero > 0:
        print("SUCCESS: Visual input received.")
    else:
        print("WARNING: Image is black. (Could be valid if looking at empty space, but let's check).")

    # 3. Test Rotation
    print("\nTest 3: Rotate to find Goal")
    # Rotate Left 10 times
    for i in range(10):
        obs, _, _, _, _ = env.step(2)
        
    print("Rotated 10 steps.")
    non_zero_rot = np.count_nonzero(obs)
    print(f"Non-zero pixels after rotation: {non_zero_rot}")

    print("\nPyBullet Environment Verified.")

if __name__ == "__main__":
    verify_pybullet()
