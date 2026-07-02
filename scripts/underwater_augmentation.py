import cv2
import numpy as np

def apply_underwater_effects(image):
    """
    Applies custom underwater augmentations to a standard image.
    Simulates depth color absorption, turbidity, low-light, and backscatter.
    """
    # 1. Simulate Turbidity & Depth (Absorb Red, Boost Green/Blue)
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    b, g, r = cv2.split(blurred)
    
    r = cv2.multiply(r, 0.5)  # Water absorbs red light quickly
    g = cv2.multiply(g, 1.1)  # Boost green
    b = cv2.multiply(b, 1.2)  # Boost blue
    
    water_tint = cv2.merge([b, g, r])
    water_tint = np.clip(water_tint, 0, 255).astype(np.uint8)

    # 2. Simulate Low-Light (Decrease brightness/contrast)
    low_light = cv2.convertScaleAbs(water_tint, alpha=0.7, beta=-20)

    # 3. Simulate Backscatter (Inject Gaussian noise)
    noise = np.random.normal(0, 15, low_light.shape).astype(np.uint8)
    final_image = cv2.add(low_light, noise)
    
    return final_image

if __name__ == "__main__":
    print("Underwater augmentation module ready to be imported into the YOLO pipeline!")