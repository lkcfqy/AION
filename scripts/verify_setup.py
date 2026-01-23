import sys
import os
import time
import numpy as np

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.environment import AIONEnvironment
from src.dashboard import AIONDashboard

def main():
    print("Initializing AION Environment...")
    env = AIONEnvironment()
    
    print("Initializing AION Dashboard...")
    try:
        dashboard = AIONDashboard()
    except Exception as e:
        print(f"\n[ERROR] {e}")
        print("Please ensure Visdom server is running: 'python -m visdom.server'")
        return

    print("\nStarting verification loop (Random Walk)...")
    obs, info = env.reset()
    dashboard.update_env_view(obs)

    # Simulation loop
    for i in range(50):
        # 1. Random Action
        action = env.action_space.sample()
        
        # 2. Step Environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        # 3. Update Dashboard
        dashboard.update_env_view(obs)
        
        # 4. Generate Dummy Data for other panels (Verification only)
        # Random spikes: 10 neurons firing randomly out of 100
        dummy_spikes = np.random.choice(100, 5, replace=False)
        dashboard.update_lsm_raster(dummy_spikes)
        
        # Random similarity: oscillation
        dummy_sim = 0.5 + 0.4 * np.sin(i * 0.2)
        dashboard.update_hdc_similarity(dummy_sim)
        
        # Random Energy: decaying
        dummy_energy = np.exp(-i * 0.05)
        dashboard.update_energy(dummy_energy)
        
        # Random Survival: Hunger increasing, Free Energy fluctuating
        dummy_hunger = i * 0.02
        dummy_fe = np.abs(np.random.randn())
        dashboard.update_survival(dummy_fe, dummy_hunger)

        if terminated or truncated:
            obs, info = env.reset()

        print(f"Step {i+1}/50 completed.", end='\r')
        time.sleep(0.1) # Slow down to make it visible
    
    print("\n\nVerification Complete!")
    print("Please check the Visdom dashboard at http://localhost:8097")
    env.close()

if __name__ == "__main__":
    main()
