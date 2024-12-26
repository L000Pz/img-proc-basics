import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def gaussian_noise(img, mean=0, sigma=50):
    """
    Applies Gaussian noise to a grayscale image.

    Gaussian noise is random noise with a normal distribution. This function adds 
    Gaussian noise to an input image, allowing the user to control the mean and 
    standard deviation of the noise.

    Parameters:
        img (numpy.ndarray): The input grayscale image as a 2D numpy array.
        mean (float, optional): The mean of the Gaussian distribution. Default is 0.
        sigma (float, optional): The standard deviation (spread) of the Gaussian 
                                 distribution. Higher values result in stronger noise. 
                                 Default is 50.

    Returns:
        None: The function displays the original image alongside the noisy image 
              using Matplotlib but does not return any values. The noisy image can 
              be saved or returned if needed.
    """
    gaussian_noise = np.random.normal(mean, sigma, img.shape)
    applied_noise = img + gaussian_noise
    applied_noise = np.clip(applied_noise, 0, 255)  # Limiting the values between 0 and 255
    
    added_gaus = applied_noise.astype(np.uint8)  # Turning any float values into 8-bit unsigned integers
    
    plt.subplot(1, 2, 1), plt.imshow(img, cmap='gray'), plt.title('Original')
    plt.subplot(1, 2, 2), plt.imshow(added_gaus, cmap='gray'), plt.title('Gaussian Noise')
    plt.show()
    return

def edge_detection(img):
    """
    Performs edge detection on a grayscale image using the Sobel operator.

    This function computes the horizontal and vertical gradients of an image 
    using the Sobel operator and combines them to calculate the gradient 
    magnitude. It visualizes the original image, the Sobel gradients in the X 
    and Y directions, and the gradient magnitude.

    Parameters:
        img (numpy.ndarray): The input grayscale image as a 2D numpy array.

    Returns:
        None: The function displays the original image, Sobel X, Sobel Y, 
              and Sobel magnitude images using Matplotlib but does not return 
              any values.
    """
    sobel_x = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=3)
    sobel_y = cv.Sobel(img, cv.CV_64F, 0, 1, ksize=3)
    
    magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    
    # Normalize to 0 to 255
    magnitude = np.uint8(255 * magnitude / np.max(magnitude))
    
    
    plt.subplot2grid((3, 2), (0, 0), colspan=2), plt.imshow(img, cmap='gray'), plt.title('Original')
    plt.subplot2grid((3, 2), (1, 0)), plt.imshow(sobel_x, cmap='gray'), plt.title('Sobel X')
    plt.subplot2grid((3, 2), (1, 1)), plt.imshow(sobel_y, cmap='gray'), plt.title('Sobel Y')
    plt.subplot2grid((3, 2), (2, 0), colspan=2), plt.imshow(magnitude, cmap='gray'), plt.title('Sobel Magnitude')
    
    plt.show()
    return

def histogram_equalization(img):
    """
    Performs histogram equalization on a grayscale image to improve contrast.

    Histogram equalization enhances the contrast of an image by redistributing 
    the intensity levels of its histogram, making it more uniform. This process 
    is particularly effective for images with poor contrast or limited dynamic 
    range.

    Parameters:
        img (numpy.ndarray): The input grayscale image as a 2D numpy array.

    Returns:
        None: The function displays the original image alongside the histogram 
              equalized image using Matplotlib but does not return any values.
    """
    equ = cv.equalizeHist(img)
    
    plt.subplot(1,2,1), plt.imshow(img, cmap='gray'), plt.title('Original')
    plt.subplot(1,2,2), plt.imshow(equ, cmap='gray'), plt.title('After Histogram Equalization')
    
    plt.show()
    return

def gaussian_blur(img):
    """
    Applies Gaussian blur to a grayscale image using different kernel sizes.

    Gaussian blur smooths an image by convolving it with a Gaussian kernel. 
    This function demonstrates the effect of applying Gaussian blur with 
    varying kernel sizes (3x3, 5x5, and 7x7) to compare the results.

    Parameters:
        img (numpy.ndarray): The input grayscale image as a 2D numpy array.

    Returns:
        None: The function displays the original image alongside the blurred 
              images with different kernel sizes using Matplotlib but does not 
              return any values.
    """
    blur3x3 = cv.GaussianBlur(img, (3,3), 0)
    blur5x5 = cv.GaussianBlur(img, (5,5), 0)
    blur7x7 = cv.GaussianBlur(img, (7,7), 0)
    
    plt.subplot(1,4,1), plt.imshow(img, cmap='gray'), plt.title('Original')
    plt.subplot(1,4,2), plt.imshow(blur3x3, cmap='gray'), plt.title('Gaussian Blur(K=3x3)')
    plt.subplot(1,4,3), plt.imshow(blur5x5, cmap='gray'), plt.title('Gaussian Blur(K=5x5)')
    plt.subplot(1,4,4), plt.imshow(blur7x7, cmap='gray'), plt.title('Gaussian Blur(K=7x7)')
    plt.show()
    
    return

def main():
    # Load images
    img = cv.imread(r"img\Lenna(Bano).png", 0)  # 0 converts image to grayscale
    unEqu = cv.imread(r"img\Unequalized.jpg", 0)

    if img is None or unEqu is None:
        print("Error: Could not load images. Check file paths.")
        return

    # CLI Menu
    while True:
        print("\nCLI Menu:")
        print("1. Add Gaussian Noise")
        print("2. Perform Edge Detection")
        print("3. Perform Histogram Equalization")
        print("4. Apply Gaussian Blur")
        print("5. Exit")

        choice = input("Enter your choice: ")

        if choice == '1':
            gaussian_noise(img)
        elif choice == '2':
            edge_detection(img)
        elif choice == '3':
            histogram_equalization(unEqu)
        elif choice == '4':
            gaussian_blur(img)
        elif choice == '5':
            print("Exiting program.")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__=="__main__":
    main()
