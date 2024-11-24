import cv2
import numpy as np
from numba import jit

@jit(nopython=True)
def _calculate_local_stats(integral, integral_squared, i, j, window_size, rows, cols):
    y1 = max(0, i - window_size//2)
    y2 = min(rows, i + window_size//2 + 1)
    x1 = max(0, j - window_size//2)
    x2 = min(cols, j + window_size//2 + 1)
    
    count = (y2 - y1) * (x2 - x1)
    
    sum_val = (integral[y2, x2] - integral[y2, x1] - 
                integral[y1, x2] + integral[y1, x1])
    sum_sq = (integral_squared[y2, x2] - integral_squared[y2, x1] -
                integral_squared[y1, x2] + integral_squared[y1, x1])
    
    mean = sum_val / count
    variance = max(0, (sum_sq - (sum_val**2) / count) / count)
    std = np.sqrt(variance)
    
    return mean, std

def sauvola_binarization(image, window_size=15, k=0.2, R=128, padding_mode='reflect'):
    # Input validation
    if window_size % 2 == 0:
        window_size = window_size + 1
    
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Pad the image
    pad_size = window_size // 2
    padded_image = cv2.copyMakeBorder(
        image,
        pad_size, pad_size, pad_size, pad_size,
        cv2.BORDER_REFLECT
    )
    
    # Calculate integral images
    padded_image = padded_image.astype(np.float32)
    integral = cv2.integral(padded_image)
    integral_squared = cv2.integral(padded_image**2)
    
    # Get dimensions
    rows, cols = image.shape
    threshold = np.zeros_like(image, dtype=np.float32)
    
    # Prepare coordinates for vectorized operations
    y_coords, x_coords = np.meshgrid(
        np.arange(rows),
        np.arange(cols),
        indexing='ij'
    )
    
    # Calculate local statistics and threshold
    for i in range(rows):
        for j in range(cols):
            mean, std = _calculate_local_stats(
                integral, integral_squared,
                i + pad_size, j + pad_size,
                window_size, rows + 2*pad_size, cols + 2*pad_size
            )
            threshold[i, j] = mean * (1 + k * ((std / R) - 1))
    
    # Apply threshold and return binary image
    return np.where(image > threshold, 255, 0).astype(np.uint8)

    print('deez nuts')

if __name__ == "__main__":
    image_path = "/home/ansel/Downloads/IMG_20230917_225358.jpg"  # Replace with your image path
    image = cv2.imread(image_path)
    
    # Apply Sauvola binarization
    binary_image = sauvola_binarization(
        image,
        window_size=15,  # Adjust this parameter as needed
        k=0.2,          # Adjust this parameter as needed
        R=128           # Adjust this parameter as needed
    )
    
    # Display results
    cv2.imshow('Original', image)
    cv2.imshow('Binarized', binary_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Optionally save the result
    cv2.imwrite('binarized_output.jpg', binary_image)

    # Check if image is loaded
    if image is None:
        print("Error: Could not load image")
        exit()
