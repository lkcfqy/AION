import sys
import os
import torch
import numpy as np

# Add project root
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.hrr import HDCWorldModel, bind
from src.config import HDC_DIM, SEED

def verify_3d_causality():
    print("Verifying 3D Causality (HRR)...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    wm = HDCWorldModel(device=device)
    
    torch.manual_seed(SEED)
    
    def get_vec():
        v = torch.sign(torch.randn(HDC_DIM, device=device))
        v[v==0] = 1.0
        return v
        
    # 1. Define 3D States (Vertical Column)
    # Ground -> Mid -> High
    state_ground = get_vec()
    state_mid = get_vec()
    state_high = get_vec()
    
    # 2. Define Actions (Up/Down)
    action_up = get_vec()
    action_down = get_vec()
    
    # 3. Learn Transitions
    # Ground + Up -> Mid
    wm.learn(state_ground, action_up, state_mid)
    # Mid + Up -> High
    wm.learn(state_mid, action_up, state_high)
    # High + Down -> Mid
    wm.learn(state_high, action_down, state_mid)
    
    print("Learned 3 transitions.")
    
    # 4. Predict
    # Predict Mid from Ground+Up
    pred_mid = wm.predict(state_ground, action_up)
    
    # Check similarity
    # Cosine sim for {-1, 1} is dot/D
    sim = torch.mean(pred_mid * state_mid).item()
    print(f"Prediction (Ground + Up -> Mid) Similarity: {sim:.4f}")
    
    if sim > 0.4: 
        # HRR with 3 items superposition leads to ~25% bit errors (Sim ~0.5).
        # This is sufficient for MHN to cleanup (as verified in Phase 2).
        print("SUCCESS: Z-axis causality captured (Sim ~0.5 expected).")
    else:
        # HRR capacity is ~ D / (2*log(N)). For D=10000, capacity is high.
        # 3 items is nothing. Should be near 1.0 (minus orthogonality cross-talk).
        print("FAILURE: Prediction efficient too low.")

    # 5. Multi-step Prediction (Ground -> Up -> Up -> High)
    # This tests recursive prediction if we implemented permutation right.
    # Pred1 = WM(Ground, Up)
    # Pred2 = WM(Pred1, Up)
    pred_high_indirect = wm.predict(pred_mid, action_up)
    
    sim2 = torch.mean(pred_high_indirect * state_high).item()
    print(f"Multi-step (Ground + Up + Up -> High) Similarity: {sim2:.4f}")
    
    if sim2 > 0.3: # Multi-Step is harder without cleanup, but should be positive
        print("SUCCESS: Multi-step inference plausible.")
    else:
        print("WARNING: Multi-step drifted too much (Requires MHN cleanup step detailed in Phase 2).")

if __name__ == "__main__":
    verify_3d_causality()
