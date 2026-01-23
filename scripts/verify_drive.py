import sys
import os
import time
import numpy as np

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.drive import BiologicalDrive
from src.dashboard import AIONDashboard

def main():
    print("1. Initializing Dashboard...")
    try:
        dashboard = AIONDashboard()
    except:
        print("Visdom not running? Please run 'python -m visdom.server'")
        return

    print("2. Initializing Biological Drive...")
    drive = BiologicalDrive()
    
    # Simulation Parameters
    n_steps = 500
    
    # Fake inputs
    # Let's simulate a scenario:
    # Agent wanders around (Surprise fluctuates low)
    # But it doesn't find food. Hunger rises.
    # Suddenly at step 300, it finds food.
    
    print("Starting Simulation...")
    print(f"Goal: Observe Hunger rising and then resetting.")
    
    for t in range(n_steps):
        # 1. Simulate Inputs
        # Surprise: mostly low (predictable), occassional spike
        surprise = 0.1 + np.random.normal(0, 0.02)
        if np.random.rand() < 0.05: surprise += 0.3 # random shock
        surprise = max(0.0, min(1.0, surprise))
        
        # Goal Delta: High (far from food)
        goal_delta = 0.8
        
        # 2. Check Event (Food found at step 300, 450)
        got_food = (t == 300 or t == 450)
        
        if got_food:
            print(f"Step {t}: Food Found! Yum.")
            # If food found, we assume goal_delta momentarily drops to 0 (eating)
            goal_delta = 0.0
            
        # 3. Update Drive
        drive.step(got_food=got_food)
        
        # 4. Compute Loss
        free_energy = drive.compute_free_energy(surprise, goal_delta)
        
        # 5. Visualize
        # Dashboard expects: update_survival(free_energy, hunger)
        dashboard.update_survival(free_energy, drive.hunger)
        
        # Slow down for visual effect
        time.sleep(0.01)
        
        if t % 50 == 0:
            print(f"Step {t} | Hunger: {drive.hunger:.3f} | F: {free_energy:.3f}")

    print("Simulation Complete.")
    print("Please check Visdom 'Survival Curve'.")
    print("You should see:")
    print("1. Orange Line (Hunger) linearly increasing.")
    print("2. Blue Line (Free Energy) increasing along with Hunger.")
    print("3. Drops at step 300 and 450.")

if __name__ == "__main__":
    main()
