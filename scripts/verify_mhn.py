import sys
import os
import torch
import numpy as np

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.mhn import ModernHopfieldNetwork
from src.config import HDC_DIM

def generate_random_pattern(dim):
    p = torch.randn(dim)
    p = torch.sign(p)
    p[p==0] = 1.0
    return p

def add_noise(pattern, noise_ratio):
    """Flip noise_ratio fraction of bits"""
    p = pattern.clone()
    dim = p.shape[0]
    n_flip = int(dim * noise_ratio)
    
    # Randomly select indices to flip
    indices = torch.randperm(dim)[:n_flip]
    p[indices] *= -1.0
    return p

def hamming_similarity(v1, v2):
    return torch.mean(v1 * v2).item()

def main():
    print("Initializing Modern Hopfield Network...")
    mhn = ModernHopfieldNetwork()
    
    # 1. Store Memories
    print("Storing 5 random concept patterns...")
    patterns = []
    for i in range(5):
        p = generate_random_pattern(HDC_DIM)
        mhn.add_memory(p)
        patterns.append(p)
        
    print(f"Memory Matrix Shape: {mhn.memory_matrix.shape}")
    
    # 2. Test Retrieval with Noise
    target_idx = 0
    target_pattern = patterns[target_idx]
    
    noise_levels = [0.1, 0.2, 0.3, 0.4, 0.45, 0.5]
    
    print(f"\nTesting Retrieval on Pattern #{target_idx}")
    print(f"{'Noise':<10} | {'Input Sim':<10} | {'Output Sim':<10} | {'Status'}")
    print("-" * 50)
    
    for noise in noise_levels:
        # Create noisy query
        query = add_noise(target_pattern, noise)
        
        # Calculate input similarity (Ideal: 1 - 2*noise)
        input_sim = hamming_similarity(query, target_pattern)
        
        # Retrieve
        recalled = mhn.retrieve(query)
        
        # Calculate output similarity
        output_sim = hamming_similarity(recalled, target_pattern)
        
        status = "PERFECT" if output_sim == 1.0 else ("GOOD" if output_sim > 0.9 else "FAIL")
        
        print(f"{noise:<10.2f} | {input_sim:<10.4f} | {output_sim:<10.4f} | {status}")

    print("\n------------------------------------------------")
    # Verify non-interference
    # Query with Pattern 1, should not recall Pattern 0
    print("Interference Check: Query Pattern #1")
    recalled_1 = mhn.retrieve(patterns[1])
    sim_0 = hamming_similarity(recalled_1, patterns[0])
    sim_1 = hamming_similarity(recalled_1, patterns[1])
    print(f"Sim with Pattern #0: {sim_0:.4f} (Should be low)")
    print(f"Sim with Pattern #1: {sim_1:.4f} (Should be 1.0)")
    
    if sim_1 == 1.0 and abs(sim_0) < 0.1:
        print("SUCCESS: MHN is working correctly.")
    else:
        print("WARNING: Interference detected.")

if __name__ == "__main__":
    main()
