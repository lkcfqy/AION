import numpy as np
import cv2
import torch
from src.config import OBS_SHAPE

class IsaacEnv:
    """
    Wrapper for Nvidia Isaac Sim / OmniIsaacGymEnvs.
    Handles the interface between AION agent and the physical simulation.
    """
    def __init__(self, headless=True):
        print("Initializing Isaac Sim Environment...")
        
        # TODO: Import OmniIsaacGymEnvs here
        # self.isaac_env = ...
        
        # Placeholder for camera resolution
        self.camera_res = (1280, 720) 
        
    def reset(self):
        """
        Reset the drone environment.
        Returns:
            obs: (56, 56, 3) processed image
            info: dict
        """
        print("Resetting Isaac Env...")
        # Mock initial observation for now
        raw_img = np.zeros((self.camera_res[1], self.camera_res[0], 3), dtype=np.uint8)
        
        processed_obs = self._process_visuals(raw_img)
        return processed_obs, {}

    def step(self, action_id):
        """
        Execute action.
        Args:
            action_id: int (0-5)
        Returns:
            obs, reward, done, truncated, info
        """
        # TODO: Map action_id to velocity command
        # if action_id == 0: ...
        
        # Mock step
        raw_img = np.zeros((self.camera_res[1], self.camera_res[0], 3), dtype=np.uint8)
        processed_obs = self._process_visuals(raw_img)
        
        reward = 0.0
        done = False
        truncated = False
        info = {}
        
        return processed_obs, reward, done, truncated, info
        
    def _process_visuals(self, raw_img):
        """
        Bio-Retina Pipeline:
        HD RGB -> Grayscale -> Edge Detection -> Resize
        """
        # 1. Grayscale
        gray = cv2.cvtColor(raw_img, cv2.COLOR_RGB2GRAY)
        
        # 2. Edge Detection (Canny)
        # Simulate retina ganglion cells (center-surround)
        edges = cv2.Canny(gray, 100, 200)
        
        # 3. Resize to OBS_SHAPE (56, 56)
        # Note: OBS_SHAPE is (56, 56, 3) in config currently, but edges is 1 channel.
        # We might need to stack it or RGB it to keep LSM compatibility if LSM expects 3 channels.
        # Let's check config. OBS_SHAPE = (56, 56, 3).
        # We can replicate channels or use just 1 if we change config.
        # Plan says "Resize to (56, 56) or (64, 64) to adapt src/config.py".
        # If config expects 3 channels, we should probably return 3 channels for minimal friction,
        # or change config.
        
        resized = cv2.resize(edges, (OBS_SHAPE[0], OBS_SHAPE[1]))
        
        # Conver back to 3 channels for compatibility specific to current LSM input?
        # LSM flattens it anyway.
        # But if we want to visualize it easily as an image in Visdom, 3 channels is nice (even if grayscale).
        
        obs = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
        
        # Normalize to [0, 1] ? 
        # env.py usually returns 0-255 or 0-1? 
        # Minigrid returns 0-255 uint8. 
        # We'll return uint8 here.
        
        return obs
