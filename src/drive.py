import numpy as np
from src.config import LAMBDA_HUNGER, HUNGER_INC

class BiologicalDrive:
    """
    Biological Drive System.
    Manages physiological needs (Hunger) and computes Total Free Energy.
    Task 3.2: The Drive
    F = Surprise + Lambda * Hunger * GoalDelta
    """
    def __init__(self):
        self.hunger = 0.0 # 0.0 to 1.0
        self.lambda_hunger = LAMBDA_HUNGER
        self.inc_rate = HUNGER_INC
        
    def step(self, got_food=False):
        """
        Update physiological state.
        Call this every time step.
        """
        if got_food:
            self.hunger = 0.0
        else:
            self.hunger += self.inc_rate
            
        # Clamp to [0, 1]
        self.hunger = min(max(self.hunger, 0.0), 1.0)
        
    def compute_free_energy(self, surprise, goal_delta):
        """
        Compute the modified Free Energy (Loss Function).
        Args:
            surprise: float [0, 1] (Dist between Sense and Pred)
            goal_delta: float [0, 1] (Dist between Sense and Goal)
        Returns:
            F: float
        """
        # Formula: F = S + (lambda * H * G)
        # Note: If hunger is 0, F = S (Pure Curiosity / Predictability)
        # If hunger is 1, F = S + G (Survival overrides)
        
        # We might want lambda to be stronger to override surprise?
        # But for now, simple addition.
        
        drive_term = self.lambda_hunger * self.hunger * goal_delta
        return surprise + drive_term
        
    def get_status(self):
        return {
            "hunger": self.hunger
        }
