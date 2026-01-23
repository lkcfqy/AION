import sys
import os
import torch
import numpy as np

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.hrr import HDCWorldModel, bind
from src.config import HDC_DIM

def generate_random_vector():
    vec = torch.randn(HDC_DIM)
    vec = torch.sign(vec)
    vec[vec == 0] = 1.0
    return vec

def hamming_similarity(v1, v2):
    return torch.mean(v1 * v2).item()

from src.mhn import ModernHopfieldNetwork

def main():
    print("Initializing HRR World Model...")
    wm = HDCWorldModel()
    
    # Initialize MHN for Cleanup
    print("Initializing MHN (The Cleaner)...")
    mhn = ModernHopfieldNetwork()
    
    # 1. Define World States and Actions
    print("Defining 3 Locations (A, B, C) and 2 Actions (Left, Right)...")
    loc_A = generate_random_vector()
    loc_B = generate_random_vector()
    loc_C = generate_random_vector()
    
    # Store concepts in MHN
    mhn.add_memory(loc_A)
    mhn.add_memory(loc_B)
    mhn.add_memory(loc_C)
    
    act_Right = generate_random_vector()
    act_Left = generate_random_vector()
    
    # 2. Define Transitions
    transitions = [
        (loc_A, act_Right, loc_B, "A -> R -> B"),
        (loc_B, act_Right, loc_C, "B -> R -> C"),
        (loc_C, act_Left,  loc_B, "C -> L -> B"),
        (loc_B, act_Left,  loc_A, "B -> L -> A")
    ]
    
    # 3. Learn
    print("\nLearning Transitions...")
    for s, a, ns, name in transitions:
        wm.learn(s, a, ns)
        print(f"Learned: {name}")
        
    # 4. Predict and Verify
    print("\nTesting Predictions (with MHN Cleanup)...")
    
    pass_count = 0
    total_count = len(transitions)
    
    for s, a, expected_ns, name in transitions:
        # Raw Prediction (Noisy)
        pred_noisy = wm.predict(s, a)
        sim_noisy = hamming_similarity(pred_noisy, expected_ns)
        
        # Cleanup with MHN
        pred_clean = mhn.retrieve(pred_noisy)
        sim_clean = hamming_similarity(pred_clean, expected_ns)
        
        status = "PASS" if sim_clean > 0.99 else "FAIL"
        if status == "PASS": pass_count += 1
        
        print(f"Query: {name.split('->')[0]} + {name.split('->')[1]} -> ?")
        print(f"  Raw Sim:   {sim_noisy:.4f}")
        print(f"  Clean Sim: {sim_clean:.4f}")
        print(f"  Status:    {status}")
        print("-" * 30)

    # 5. Compositional Test (Multi-step)
    print("\nMulti-step Prediction Test (A + R + R -> ?)")
    # Step 1
    step1_noisy = wm.predict(loc_A, act_Right)
    step1_clean = mhn.retrieve(step1_noisy) # B
    
    # Step 2
    step2_noisy = wm.predict(step1_clean, act_Right)
    step2_clean = mhn.retrieve(step2_noisy) # C
    
    sim_final = hamming_similarity(step2_clean, loc_C)
    print(f"Final Sim with C: {sim_final:.4f}")
    if sim_final > 0.99: 
        print("SUCCESS: Multi-step prediction worked!")
    else:
        print("WARNING: Signal degraded too much.")
        
    if pass_count == total_count:
        print("\nOVERALL SUCCESS: World Model learned structure.")
    else:
        print("\nOVERALL FAIL: Some predictions failed.")


if __name__ == "__main__":
    main()
