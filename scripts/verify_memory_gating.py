import sys
import os
import torch
import numpy as np

# Add project root
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.mhn import ModernHopfieldNetwork
from src.config import MEMORY_THRESHOLD, HDC_DIM

def verify_memory_gating():
    print("Verifying Dynamic Memory Gating...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mhn = ModernHopfieldNetwork(device=device)
    
    # 1. Create Base Pattern
    torch.manual_seed(42)
    base_pattern = torch.sign(torch.randn(HDC_DIM))
    base_pattern[base_pattern == 0] = 1.0
    
    # 2. Add Base Pattern (Should Succeed)
    added = mhn.add_memory(base_pattern)
    print(f"Adding Base Pattern: {'Success' if added else 'Failed'} (Expected: Success)")
    if not added: return False
    
    # 3. Try adding Same Pattern (Should Fail)
    added = mhn.add_memory(base_pattern)
    print(f"Adding Duplicate: {'Success' if added else 'Skipped'} (Expected: Skipped)")
    if added: 
        print("FAILURE: Duplicate should be skipped.")
        return False
        
    # 4. Try adding Highly Similar Pattern (Should Fail)
    # create noise by flipping 2% of bits (Sim = 0.96 > 0.9)
    noise_idx = torch.randperm(HDC_DIM)[:int(HDC_DIM * 0.02)]
    sim_pattern = base_pattern.clone()
    sim_pattern[noise_idx] *= -1.0 # Flip
    
    added = mhn.add_memory(sim_pattern)
    print(f"Adding Similar (98%): {'Success' if added else 'Skipped'} (Expected: Skipped)")
    if added:
        print("FAILURE: Similar pattern should be skipped.")
        return False
        
    # 5. Try adding Distinct Pattern (Should Succeed)
    # Flip 50% bits -> Orthogonal
    ortho_pattern = torch.sign(torch.randn(HDC_DIM))
    added = mhn.add_memory(ortho_pattern)
    print(f"Adding Novel Pattern: {'Success' if added else 'Failed'} (Expected: Success)")
    
    if added:
        print("\nDYNAMIC GATING VERIFIED.")
        return True
    else:
        print("FAILURE: Novel pattern was skipped.")
        return False

if __name__ == "__main__":
    verify_memory_gating()
