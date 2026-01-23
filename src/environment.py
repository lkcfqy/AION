import gymnasium as gym
import minigrid
import numpy as np
from gymnasium.core import ObservationWrapper
from minigrid.wrappers import ImgObsWrapper, RGBImgPartialObsWrapper

from src.config import ENV_ID, RENDER_MODE

class AIONEnvironment:
    """
    Wrapper for MiniGrid environment to provide standardized RGB inputs.
    Task 0.1: Body / Grounding
    """
    def __init__(self, render_mode=RENDER_MODE):
        # Create base environment
        self._env = gym.make(ENV_ID, render_mode=render_mode)
        
        # Wrap to get RGB image observations
        # RGBImgPartialObsWrapper provides raw RGB pixels of the agent's field of view
        self._env = RGBImgPartialObsWrapper(self._env) 
        self._env = ImgObsWrapper(self._env) # Removes the 'mission' string, keeping only 'image'

        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space

    def reset(self, seed=None):
        obs, info = self._env.reset(seed=seed)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self._env.step(action)
        return obs, reward, terminated, truncated, info

    def close(self):
        self._env.close()

    @property
    def unwrapped(self):
        return self._env.unwrapped
