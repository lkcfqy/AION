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
        self.battery = 1.0 # 1.0 (Full) to 0.0 (Empty)
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
            
        # Clamp Hunger
        self.hunger = min(max(self.hunger, 0.0), 1.0)
        
        # Battery Logic
        # Flight consumes battery. 
        # If got_food (charging), battery fills up.
        if got_food:
            self.battery = 1.0
        else:
            self.battery -= 0.0005 # Slow decay
            
        self.battery = max(self.battery, 0.0)

    def compute_free_energy(self, surprise, goal_delta):
        """
        Compute the modified Free Energy (Loss Function).
        F = Surprise + Lambda * MetabolicError * GoalDelta
        MetabolicError = max(Hunger, 1.0 - Battery)
        """
        # Metabolic Error is the maximum of Hunger or Low Battery
        metabolic_error = max(self.hunger, 1.0 - self.battery)
        
        drive_term = self.lambda_hunger * metabolic_error * goal_delta
        return surprise + drive_term
        
    def get_status(self):
        return {
            "hunger": self.hunger,
            "battery": self.battery
        }
