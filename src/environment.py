from src.environment_pybullet import PyBulletEnv

class AIONEnvironment(PyBulletEnv):
    """
    Wrapper for PyBullet Environment to provide standardized inputs.
    Inherits from PyBulletEnv (Phase 1.1).
    """
    def __init__(self, headless=False):
        # Initialize PyBulletEnv
        # headless=True for server deployment, False for local visualization
        super().__init__(headless=headless)
        
    def step(self, action):
        # PyBulletEnv step returns (obs, reward, done, truncated, info)
        # Our agent expects this signature.
        return super().step(action)

    def teleport_agent(self, pos):
        super().teleport_agent(pos)

