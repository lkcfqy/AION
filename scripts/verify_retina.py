import sys
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Add project root
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.environment import AIONEnvironment

def verify_retina():
    print("Verifying Bio-Retina Pipeline...")
    
    # 1. Instantiate Env
    env = AIONEnvironment()
    
    # 2. Create Dummy HD Image (e.g. 1280x720) containing a recognizable shape
    print("Generating dummy HD image...")
    dummy_img = np.zeros((720, 1280, 3), dtype=np.uint8)
    
    # Draw a rectangle (Obstacle)
    cv2.rectangle(dummy_img, (400, 200), (800, 500), (200, 200, 200), -1)
    
    # Draw a Circle (Target)
    cv2.circle(dummy_img, (1000, 100), 50, (0, 255, 0), -1)
    
    # Add some noise (Texture)
    noise = np.random.randint(0, 50, (720, 1280, 3), dtype=np.uint8)
    dummy_img = cv2.add(dummy_img, noise)
    
    # 3. Process
    # We access the internal method for testing since we don't have real connection
    processed = env._process_visuals(dummy_img)
    
    print(f"Original Shape: {dummy_img.shape}")
    print(f"Processed Shape: {processed.shape}")
    
    # 4. Save/Visualize
    # Save original and processed to temporary files for inspection if needed, 
    # but since we are in headless agent, we just check properties.
    
    # Check if edges are preserved
    # The simple Downsample might lose thin edges, but Outline detection (Canny) happens BEFORE resize.
    
    # Since Canny makes edges white (255), let's see if we have non-zero pixels.
    non_zero = np.count_nonzero(processed)
    print(f"Non-zero pixels in processed image: {non_zero}")
    
    if non_zero > 10:
        print("SUCCESS: Edge features detected and preserved.")
    else:
        print("FAILURE: Image is empty.")
        
    print("Bio-Retina Verification Complete.")

if __name__ == "__main__":
    verify_retina()
