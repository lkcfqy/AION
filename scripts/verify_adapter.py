import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.adapter import RandomProjectionAdapter
from src.config import LSM_N_NEURONS

def cosine_similarity(v1, v2):
    return torch.dot(v1, v2) / (torch.norm(v1) * torch.norm(v2))

def hamming_similarity(v1, v2):
    # v1, v2 are {-1, 1}
    # Match (+1, +1) or (-1, -1) -> product is 1
    # Mismatch -> product is -1
    # HamSim = mean(product) -> 1 is identical, -1 is opposite, 0 is orthogonal
    return torch.mean(v1 * v2)

def main():
    print("Initializing Adapter...")
    adapter = RandomProjectionAdapter()
    
    print("\nTest 1: Similarity Preservation (LSH Property)")
    
    # 1. Base Vector A
    vec_a = torch.randn(LSM_N_NEURONS)
    
    # 2. Similar Vector B (A + noise)
    # Mix A with noise. 
    noise_level = 0.5
    vec_b = vec_a + noise_level * torch.randn(LSM_N_NEURONS)
    
    # 3. Orthogonal/Random Vector C
    vec_c = torch.randn(LSM_N_NEURONS)
    
    # Calculate Input Similarities
    sim_ab_in = cosine_similarity(vec_a, vec_b)
    sim_ac_in = cosine_similarity(vec_a, vec_c)
    
    print(f"Input Sim(A, B): {sim_ab_in:.4f} (Should be high)")
    print(f"Input Sim(A, C): {sim_ac_in:.4f} (Should be ~0)")
    
    # Project to HDC
    hdc_a = adapter.forward(vec_a)
    hdc_b = adapter.forward(vec_b)
    hdc_c = adapter.forward(vec_c)
    
    # Calculate Output Similarities
    sim_ab_out = hamming_similarity(hdc_a, hdc_b)
    sim_ac_out = hamming_similarity(hdc_a, hdc_c)
    
    print(f"Output HDC Sim(A, B): {sim_ab_out:.4f}")
    print(f"Output HDC Sim(A, C): {sim_ac_out:.4f}")
    
    print("\n------------------------------------------------")
    if sim_ab_out > 0.5 and abs(sim_ac_out) < 0.1:
        print("SUCCESS: Adapter preserves similarity structure.")
    else:
        print("WARNING: Adapter behavior unexpected.")

    # Optional: Plot curve
    # Vary similarity of input, measure similarity of output
    print("\nTest 2: Correlation Curve...")
    correlations = []
    mix_ratios = np.linspace(0, 1, 11) # 0 = Identical, 1 = Orthogonal (replace with noise)
    
    # Ideally: Input Cosine = X. Output Hamming = 1 - (angle/pi)?
    # SimHash theory: Output Hamming Sim = 1 - 2 * angle / pi? No.
    # Pr[h(x)=h(y)] = 1 - theta/pi.
    # Hamming Sim (normalized -1 to 1) = (Matches - Mismatches) / N
    # = (1 - theta/pi) - (theta/pi) = 1 - 2*theta/pi.
    # Cosine = cos(theta). -> theta = acos(Cosine).
    # Expected Sim = 1 - 2 * acos(Cosine) / pi.
    
    print(f"{'Input Cos':<10} | {'Pred HDC':<10} | {'Actual HDC':<10}")
    
    for r in mix_ratios:
        # Interpolate between A and Random C
        # vec = (1-r)*A + r*C
        # normalize
        vec_mix = (1-r) * vec_a + r * vec_c
        
        sim_in = cosine_similarity(vec_a, vec_mix)
        hdc_mix = adapter.forward(vec_mix)
        sim_out = hamming_similarity(hdc_a, hdc_mix)
        
        # Theoretical prediction
        theta = torch.acos(sim_in)
        predicted_sim_out = 1.0 - 2.0 * theta / np.pi
        
        print(f"{sim_in.item():<10.4f} | {predicted_sim_out.item():<10.4f} | {sim_out.item():<10.4f}")

if __name__ == "__main__":
    main()
