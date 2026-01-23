import sys
import os
import time
import numpy as np

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.environment import AIONEnvironment
from src.dashboard import AIONDashboard
from src.lsm import AION_LSM_Network
from src.config import TARGET_FIRING_RATE

def main():
    print("1. Initializing Environment...")
    env = AIONEnvironment()
    
    print("2. Initializing Dashboard...")
    try:
        dashboard = AIONDashboard()
    except:
        print("Visdom not running. Continuing without dashboard.")
        dashboard = None

    print("3. Initializing LSM (This may take a moment to build weights)...")
    lsm = AION_LSM_Network()

    print("\nStarting Homeostasis Verification...")
    print(f"Goal: Observe firing rate convergence to ~{TARGET_FIRING_RATE} Hz")
    
    obs, info = env.reset()
    dashboard.update_env_view(obs) if dashboard else None
    
    # Run for 500 steps
    n_steps = 500
    
    # Track metrics
    history_mean_rates = []
    
    print("\n[Step | Mean Rate (Hz) | Max Rate (Hz) | Active Neurons]")
    
    for i in range(n_steps):
        # 1. Random Action for dynamic visual input
        if i % 10 == 0: # Change action every 10 steps to change view
            action = env.action_space.sample()
            obs, _, _, _, _ = env.step(action)
            dashboard.update_env_view(obs) if dashboard else None
            
        # 2. Step LSM
        spikes = lsm.step(obs)
        
        # 3. Analyze
        active_neurons = np.where(spikes > 0)[0]
        
        # Calculate instantaneous population rate (Hz) for this step
        # Sum of spikes / N_neurons / dt
        # But spikes is vector of 0 or 1/dt? Nengo spikes are amplitude 1/dt.
        # Wait, sim.data[probe] returns amplitudes which are 1/dt if spiking?
        # Nengo default: spikes are 1/dt amplitude. So sum(spikes) is total Hz sum.
        # Mean rate = sum(spikes) / N
        
        # Actually nengo probe raw output: if synapse=None, it returns raw values.
        # For Spikes, it's 1/dt when firing.
        # Let's verify: sum(spikes) / 1000 = mean rate of population?
        # Yes.
        
        mean_rate = np.mean(spikes) # This will be in Hz
        max_rate = np.max(spikes) if len(spikes) > 0 else 0
        
        history_mean_rates.append(mean_rate)
        
        # 4. Dashboard
        if dashboard:
            dashboard.update_lsm_raster(active_neurons)
            # Plot rate curve on Energy plot for now (reusing panel)
            dashboard.vis.line(X=np.array([i]), Y=np.array([mean_rate]), win=dashboard.win_energy, update='append', opts=dict(title="Population Mean Firing Rate (Hz)"))

        print(f"{i:4d} | {mean_rate:6.2f} | {max_rate:6.2f} | {len(active_neurons)}", end='\r')

    print("\n\nVerification Complete.")
    avg_final = np.mean(history_mean_rates[-20:])
    print(f"Final Average Mean Rate (last 20 steps): {avg_final:.2f} Hz")
    print(f"Target: {TARGET_FIRING_RATE} Hz")
    
    if abs(avg_final - TARGET_FIRING_RATE) < 10:
        print("SUCCESS: Firing rate is within reasonable range.")
    else:
        print("WARNING: Firing rate did not converge tightly. (This is expected for short runs or poorly tuned Plasticity Rate).")

    env.close()

if __name__ == "__main__":
    main()
