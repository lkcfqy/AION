"""
AION Project Global Configuration
"""

# Environment Settings
ENV_ID = "MiniGrid-Empty-8x8-v0"
RENDER_MODE = "rgb_array"
OBS_SHAPE = (56, 56, 3)  # MiniGrid default 7x7 view * 8px tile = 56x56

# Visdom Settings
VISDOM_SERVER = "http://localhost"
VISDOM_PORT = 8097
VISDOM_ENV = "AION_Dashboard"

# Simulation Settings
SEED = 42

# LSM (Liquid State Machine) Settings
LSM_N_NEURONS = 1000
LSM_SPARSITY = 0.1     # 10% recurrent connectivity
LSM_IN_SPARSITY = 0.1  # 10% input connectivity (to handle 12k inputs)
TARGET_FIRING_RATE = 20 # Hz (Target rate for homeostasis)
PLASTICITY_RATE = 0.005 # Learning rate (More stable)
RATE_TAU = 0.05        # Rate estimation time constant (50ms)
TAU_RC = 0.02          # Membrane time constant (20ms)
TAU_REF = 0.002        # Refractory period (2ms)
DT = 0.001             # Simulation step (1ms)

# HDC Settings
HDC_DIM = 10000        # Hyperdimensional vector size
MHN_BETA = 20.0        # Modern Hopfield Network inverse temperature

# Drive Settings
LAMBDA_HUNGER = 1.0    # Balance between surprise and hunger
HUNGER_INC = 0.001     # Hunger increment per step
