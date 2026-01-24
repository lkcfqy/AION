import numpy as np
import cv2
import pybullet as p
import pybullet_data
import time
import math
from src.config import OBS_SHAPE, RENDER_MODE

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
        self.robotId = p.loadURDF("cube.urdf", startPos, startOrientation, globalScaling=0.5)
        
        # Load Goal (Green colored object)
        # Use Sphere so Bio-Retina (Edge Detection) can distinguish it from Obstacles (Cubes)
        # "sphere2.urdf" is standard in pybullet_data
        self.goalId = p.loadURDF("sphere2.urdf", [5, 5, 0.5], p.getQuaternionFromEuler([0,0,0]), globalScaling=0.5)
        p.changeVisualShape(self.goalId, -1, rgbaColor=[0, 1, 0, 1]) # Green
        
        # Add obstacle
        # Move it aside so it doesn't block the Goal (5,5) from Start (0,0)
        self.obsId = p.loadURDF("cube.urdf", [3, 0, 0.5], p.getQuaternionFromEuler([0,0,0]))
        p.changeVisualShape(self.obsId, -1, rgbaColor=[1, 0, 0, 1]) # Red
        
        # Dynamics
        self.velocity = np.zeros(3)
        self.yaw = 0.0
        
        # Initial Observation
        return self._get_observation(), {}

    def step(self, action_id):
        """
        Apply action and step simulation.
        Action Map (6-DOF):
        0: Hover, 1: Fwd, 2: RotL, 3: RotR, 4: Up, 5: Down
        """
        # 1. Decode Action to Velocity Command
        speed = 1.0
        rot_speed = 3.0 # Increase rotation speed to be visible!
        target_vel = np.array([0.0, 0.0, 0.0])
        yaw_rate = 0.0
        
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
        
    def teleport_agent(self, pos):
        """
        Teleport agent to specific position (x, y, z).
        Used for Goal Imprinting phase.
        """
        p.resetBasePositionAndOrientation(self.robotId, pos, [0,0,0,1])
        
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
        Bio-Retina Pipeline:
        HD RGB -> Grayscale -> Edge Detection -> Resize
        """
        # 1. Grayscale
        gray = cv2.cvtColor(raw_img, cv2.COLOR_RGB2GRAY)
        
        # 2. Edge Detection (Canny)
        edges = cv2.Canny(gray, 100, 200)
        
        # 3. Resize to OBS_SHAPE (56, 56)
        resized = cv2.resize(edges, (OBS_SHAPE[0], OBS_SHAPE[1]))
        
        # 4. Return as 3-channel for compatibility
        obs = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
        
        return obs
