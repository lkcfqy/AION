import sys
import os
import torch
import numpy as np

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.gwt import GlobalWorkspace
from src.config import HDC_DIM

def generate_random_vector():
    vec = torch.randn(HDC_DIM)
    vec = torch.sign(vec)
    vec[vec == 0] = 1.0
    return vec

def main():
    print("Initializing Global Workspace...")
    gwt = GlobalWorkspace()
    
    # 1. Generate Dummy Vectors
    print("Generating Sense, Pred, Goal vectors...")
    vec_sense = generate_random_vector()
    vec_goal = generate_random_vector()
    
    # CASE A: Perfect Prediction (Low Surprise)
    # Pred = Sense
    vec_pred_perfect = vec_sense.clone()
    
    # CASE B: Random Prediction (High Surprise ~0.5)
    vec_pred_random = generate_random_vector()
    
    # CASE C: Inverse Prediction (Max Surprise 1.0)
    vec_pred_inverse = -1.0 * vec_sense
    
    # 2. Test Case A
    print("\nTest Case A: Perfect Prediction")
    gwt.update_sense(vec_sense)
    gwt.update_pred(vec_pred_perfect)
    gwt.set_goal(vec_goal)
    
    surprise = gwt.compute_surprise()
    print(f"Surprise: {surprise:.4f} (Expected: 0.0)")
    if surprise < 0.01:
        print("PASS")
    else:
        print("FAIL")
        
    # 3. Test Case B
    print("\nTest Case B: Random Prediction")
    gwt.update_pred(vec_pred_random)
    surprise = gwt.compute_surprise()
    print(f"Surprise: {surprise:.4f} (Expected: ~0.5)")
    if 0.45 < surprise < 0.55:
        print("PASS")
    else:
        print("FAIL")
        
    # 4. Test Case C
    print("\nTest Case C: Inverse Prediction")
    gwt.update_pred(vec_pred_inverse)
    surprise = gwt.compute_surprise()
    print(f"Surprise: {surprise:.4f} (Expected: 1.0)")
    if surprise > 0.99:
        print("PASS")
    else:
        print("FAIL")
        
    # 5. Goal Delta
    print("\nTest Goal Delta")
    goal_delta = gwt.compute_goal_delta()
    print(f"Goal Delta: {goal_delta:.4f} (Expected: ~0.5 for random goal)")
    
    # Test goal achievement
    gwt.update_sense(vec_goal)
    goal_delta_achieved = gwt.compute_goal_delta()
    print(f"Goal Delta (Achieved): {goal_delta_achieved:.4f} (Expected: 0.0)")
    
    if goal_delta_achieved < 0.01:
        print("PASS")
    else:
        print("FAIL")

if __name__ == "__main__":
    main()
