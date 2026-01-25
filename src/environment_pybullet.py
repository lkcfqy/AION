import numpy as np
import cv2
import pybullet as p
import pybullet_data
import time
import math
from src.config import OBS_SHAPE, RENDER_MODE, ASSET_PATH
import os

class PyBulletEnv:
    """
    Wrapper for PyBullet Physics Engine.
    Simulates a Quadcopter in a 3D physical world.
    """
    def __init__(self, headless=True):
        print("Initializing PyBullet Environment...")
        self.headless = headless
        
        # 1. Connect to Engine
        if headless:
            # Use EGL for GPU acceleration in headless mode
            import sys
            options = ""
            if sys.platform == 'linux':
                options = "--use_egl"
            self.client = p.connect(p.DIRECT, options=options)
        else:
            # GUI Mode defaults to GPU (OpenGL)
            self.client = p.connect(p.GUI)
            
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        # 2. Camera Settings
        self.width = 640
        self.height = 480
        self.fov = 60
        self.aspect = self.width / self.height
        self.near = 0.1
        self.far = 100
        self.projection_matrix = p.computeProjectionMatrixFOV(self.fov, self.aspect, self.near, self.far)

        # 3. Setup World
        self.reset()
        
    def reset(self, seed=None):
        """
        Reset simulation.
        """
        p.resetSimulation()
        p.setGravity(0, 0, -9.8)
        
        # Load Ground
        self.planeId = p.loadURDF("plane.urdf")
        
        # Load Drone (Using a simple cube to represent drone for now if no drone URDF)
        # Or use R2D2 as placeholder? Let's use a Cube for simplicity and attach a camera.
        startPos = [0, 0, 1]
        startOrientation = p.getQuaternionFromEuler([0, 0, 0])
        # Try to load from ASSET_PATH, check existence first to avoid warnings
        cube_path = os.path.join(ASSET_PATH, "cube.urdf")
        if os.path.exists(cube_path):
             self.robotId = p.loadURDF(cube_path, startPos, startOrientation, globalScaling=0.5)
        else:
             # Fallback to pybullet_data/cube.urdf (default)
             self.robotId = p.loadURDF("cube.urdf", startPos, startOrientation, globalScaling=0.5)
        
        # Load Goal (Green colored object)
        sphere_path = os.path.join(ASSET_PATH, "sphere2.urdf")
        if os.path.exists(sphere_path):
            self.goalId = p.loadURDF(sphere_path, [3, 0, 0.5], p.getQuaternionFromEuler([0,0,0]), globalScaling=0.8)
        else:
            self.goalId = p.loadURDF("sphere2.urdf", [3, 0, 0.5], p.getQuaternionFromEuler([0,0,0]), globalScaling=0.8)

        p.changeVisualShape(self.goalId, -1, rgbaColor=[0, 1, 0, 1]) # Green
        p.changeVisualShape(self.goalId, -1, rgbaColor=[0, 1, 0, 1]) # Green
        
        # Add obstacle (放到侧面，不挡目标)
        self.obsId = p.loadURDF("cube.urdf", [2, 2, 0.5], p.getQuaternionFromEuler([0,0,0]))
        p.changeVisualShape(self.obsId, -1, rgbaColor=[1, 0, 0, 1]) # Red
        
        # Dynamics
        self.velocity = np.zeros(3)
        self.yaw = 0.0
        
        # Initial Observation
        return self._get_observation(), {}

    def step(self, action):
        """
        Apply action and step simulation.
        Args:
            action: int ID (0-5) OR np.array([vx, vy, vz, yaw_rate])
        """
        # 1. Decode Action to Velocity Command
        speed = 1.0
        rot_speed = 3.0 # Increase rotation speed to be visible!
        target_vel = np.array([0.0, 0.0, 0.0])
        yaw_rate = 0.0
        
        if isinstance(action, int) or isinstance(action, np.int64) or isinstance(action, float):
             action_id = int(action)
             if action_id == 1:   # Forward (Relative to Yaw)
                  target_vel[0] = math.cos(self.yaw) * speed
                  target_vel[1] = math.sin(self.yaw) * speed
             elif action_id == 2: # RotL
                  yaw_rate = rot_speed
             elif action_id == 3: # RotR
                  yaw_rate = -rot_speed
             elif action_id == 4: # Up
                  target_vel[2] = 0.5
             elif action_id == 5: # Down
                  target_vel[2] = -0.5
        elif isinstance(action, np.ndarray):
             # Controller Input: direct velocity vector [vx, vy, vz, yaw_rate]
             # Assuming input is in BODY frame (Forward is X), we need to rotate it to WORLD frame if it's not already?
             # Controller output is typically Body Frame for drones.
             # [0.5, 0, 0] -> Forward.
             # We need to rotate linear part by Yaw.
             
             body_vel = action[:3]
             w_z = action[3]
             
             # Rotate body_vel to world_vel by Yaw
             c, s = math.cos(self.yaw), math.sin(self.yaw)
             R_z = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
             world_vel = R_z.dot(body_vel)
             
             target_vel = world_vel
             yaw_rate = w_z
        else:
             print(f"Warning: Unknown action type {type(action)}")
             
        # 2. Apply Velocity (Kinematic Control for stability)
        # Update Yaw
        self.yaw += yaw_rate * 0.1 # dt approx
        
        # Apply Linear Velocity
        # In real physics we apply force, but for stability we invoke resetBaseVelocity
        p.resetBaseVelocity(self.robotId, linearVelocity=target_vel.tolist(), angularVelocity=[0,0,yaw_rate])
        
        # 3. Step Physics
        p.stepSimulation()
        
        # 4. Get Observation
        obs = self._get_observation()
        
        # 5. Check Collision / Reward
        contact_points = p.getContactPoints(self.robotId)
        collision = len(contact_points) > 0
        
        # Check if touched goal (Green Cube)
        # We need to check if contact is with goalId
        reward = 0.0
        done = False
        is_crash = False
        
        for contact in contact_points:
            auth_body_id = contact[2] # bodyB
            if auth_body_id == self.goalId:
                reward = 1.0
                done = True
            elif auth_body_id == self.obsId or auth_body_id == self.planeId:
                 # Ground or Obstacle
                 # If ground check altitude?
                 pos, _ = p.getBasePositionAndOrientation(self.robotId)
                 if pos[2] < 0.1: # Landed/Crashed
                     pass
                 else:
                     is_crash = True
        
        info = {'crash': is_crash}
        
        return obs, reward, done, False, info
        
    def teleport_agent(self, pos, yaw=0.0):
        """
        Teleport agent to specific position (x, y, z) and yaw.
        """
        self.yaw = yaw
        quat = p.getQuaternionFromEuler([0, 0, yaw])
        p.resetBasePositionAndOrientation(self.robotId, pos, quat)

    def get_pos(self):
        """Return agent's current [x, y, z] position."""
        pos, _ = p.getBasePositionAndOrientation(self.robotId)
        return list(pos)
        
    def _get_observation(self):
        """
        Capture Camera Image and process via Bio-Retina.
        """
        # Camera Position: Attached to Robot
        pos, ori = p.getBasePositionAndOrientation(self.robotId)
        rot_mat = p.getMatrixFromQuaternion(ori) # 9 elements
        
        # Camera Offset (Front of robot)
        # Simple approximation: Cam is at pos
        
        # View Matrix
        # Target is in front of robot
        # Forward vector given quaternion?
        # Let's simple calculate target based on Yaw
        # Or use rotation matrix.
        # Rot Matrix: [r11 r12 r13, r21 r22 r23, r31 r32 r33]
        # X-axis is Forward usually? Or Y? In Bullet Cube, local axes align with global when rot=0.
        # Let's look "Forward" (X+)
        
        # Forward vector is 1st column of matrix (if X-forward)
        fwd = [rot_mat[0], rot_mat[3], rot_mat[6]]
        # Up vector is 3rd column (Z-up)
        up = [rot_mat[2], rot_mat[5], rot_mat[8]]
        
        cam_target = [pos[0] + fwd[0], pos[1] + fwd[1], pos[2] + fwd[2]]
        
        view_matrix = p.computeViewMatrix(pos, cam_target, [0, 0, 1])
        
        # Capture
        w, h, rgba, depth, mask = p.getCameraImage(self.width, self.height, view_matrix, self.projection_matrix)
        
        # RGBA -> RGB
        rgb = np.array(rgba, dtype=np.uint8).reshape((self.height, self.width, 4))[:, :, :3]
        
        # Bio-Retina Pipeline
        return self._process_visuals(rgb)

    def _process_visuals(self, raw_img):
        """
        Bio-Retina Pipeline (改进版):
        HD RGB -> 边缘 + 颜色分离 -> 多通道输出
        
        输出:
        - R通道: 边缘检测 (形状感知)
        - G通道: 绿色目标掩码 (目标识别)
        - B通道: 红色障碍掩码 (危险识别)
        """
        # 1. 边缘检测
        gray = cv2.cvtColor(raw_img, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        
        # 2. 颜色分离 (HSV空间)
        hsv = cv2.cvtColor(raw_img, cv2.COLOR_RGB2HSV)
        
        # 绿色掩码 (目标 - Goal)
        lower_green = np.array([35, 50, 50])
        upper_green = np.array([85, 255, 255])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        
        # 红色掩码 (障碍物 - Obstacle)
        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 50, 50])
        upper_red2 = np.array([180, 255, 255])
        red_mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
        
        # 3. Resize to OBS_SHAPE (56, 56)
        resized_edges = cv2.resize(edges, (OBS_SHAPE[0], OBS_SHAPE[1]))
        resized_green = cv2.resize(green_mask, (OBS_SHAPE[0], OBS_SHAPE[1]))
        resized_red = cv2.resize(red_mask, (OBS_SHAPE[0], OBS_SHAPE[1]))
        
        # 4. 合并为3通道: [边缘, 绿色目标, 红色障碍]
        obs = np.stack([resized_edges, resized_green, resized_red], axis=-1)
        
        return obs.astype(np.uint8)

