import sys
import os
import numpy as np

# Add project root
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.controller import DroneController

def verify_controller():
    print("Verifying Cerebellum (DroneController)...")
    ctrl = DroneController()
    
    # Test Case 1: Sudden Acceleration (Step Response)
    print("\nTest 1: Step Response (Hover -> Forward)")
    print(f"Initial Vel: {ctrl.current_vel}")
    
    # Target: Forward [0.5, 0, 0, 0]
    # We step for 10 frames
    for i in range(10):
        vel = ctrl.step(1) # Action 1 = Forward
        print(f"Step {i}: Vx={vel[0]:.3f} (Target=0.5)")
        
    # Check if we approached 0.5 smoothly
    if vel[0] > 0.4 and vel[0] < 0.5:
        print("SUCCESS: Velocity smoothed towards target.")
    else:
        print("FAILURE: Velocity response anomalous.")

    # Test Case 2: Sudden Stop
    print("\nTest 2: Braking (Forward -> Hover)")
    for i in range(10):
        vel = ctrl.step(0) # Action 0 = Hover
        print(f"Step {i}: Vx={vel[0]:.3f} (Target=0.0)")
        
    if vel[0] < 0.1:
        print("SUCCESS: Velocity decayed smoothly.")
    else:
        print("FAILURE: Braking too slow.")

if __name__ == "__main__":
    verify_controller()
