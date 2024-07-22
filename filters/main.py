import cv2
import numpy as np
import matplotlib.pyplot as plt

# Function to apply filters
def apply_filters(image_path):
    # Read the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Apply Median filter
    median_filtered = cv2.medianBlur(image, 5)
    
    # Apply Prewitt filter (approximation using Sobel)
    prewitt_kernel_x = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
    prewitt_kernel_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
    prewitt_filtered_x = cv2.filter2D(image, -1, prewitt_kernel_x)
    prewitt_filtered_y = cv2.filter2D(image, -1, prewitt_kernel_y)
    prewitt_filtered = cv2.addWeighted(prewitt_filtered_x, 0.5, prewitt_filtered_y, 0.5, 0)

    # Apply Sobel filter
    sobel_filtered_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
    sobel_filtered_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
    sobel_filtered = cv2.addWeighted(sobel_filtered_x, 0.5, sobel_filtered_y, 0.5, 0)
    
    # Apply Laplacian filter
    laplacian_filtered = cv2.Laplacian(image, cv2.CV_64F)
    
    # Plot the results
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 3, 1), plt.imshow(image, cmap='gray')
    plt.title('Original Image'), plt.axis('off')
    
    plt.subplot(2, 3, 2), plt.imshow(median_filtered, cmap='gray')
    plt.title('Median Filter'), plt.axis('off')
    
    plt.subplot(2, 3, 3), plt.imshow(prewitt_filtered, cmap='gray')
    plt.title('Prewitt Filter'), plt.axis('off')
    
    plt.subplot(2, 3, 4), plt.imshow(sobel_filtered, cmap='gray')
    plt.title('Sobel Filter'), plt.axis('off')
    
    plt.subplot(2, 3, 5), plt.imshow(laplacian_filtered, cmap='gray')
    plt.title('Laplacian Filter'), plt.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    image_path =r'car.jpg'
    apply_filters(image_path)
