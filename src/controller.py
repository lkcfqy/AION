import numpy as np

class DroneController:
    """
    The Cerebellum.
    Translates discrete cognitive intents into smoothing continuous velocity commands.
    Prevents oscillation and crashes.
    Phase 2.2
    """
    def __init__(self):
        # 0: Hover, 1: Fwd, 2: RotL, 3: RotR, 4: Up, 5: Down
        self.action_map = {
            0: np.array([0.0, 0.0, 0.0, 0.0]), # vx, vy, vz, w
            1: np.array([0.5, 0.0, 0.0, 0.0]), # Forward (x-axis)
            2: np.array([0.0, 0.0, 0.0, 1.0]), # Rotate Left (Yaw)
            3: np.array([0.0, 0.0, 0.0, -1.0]),# Rotate Right (Yaw)
            4: np.array([0.0, 0.0, 0.5, 0.0]), # Up (z-axis)
            5: np.array([0.0, 0.0, -0.5, 0.0]) # Down (z-axis)
        }
        
        self.current_vel = np.array([0.0, 0.0, 0.0, 0.0]) # Linear + Angular
        self.alpha = 0.2 # Smoothing factor (EMA)
        
    def step(self, action_id):
        """
        Args:
            action_id: int (0-5)
        Returns:
            smooth_vel: np.array (4,) [vx, vy, vz, w]
        """
        target_vel = self.action_map.get(action_id, np.array([0.0, 0.0, 0.0, 0.0]))
        
        # EMA Smoothing
        # v_t = alpha * v_target + (1 - alpha) * v_{t-1}
        self.current_vel = self.alpha * target_vel + (1.0 - self.alpha) * self.current_vel
        
        return self.current_vel
        
    def reset(self):
        self.current_vel = np.zeros(4)
