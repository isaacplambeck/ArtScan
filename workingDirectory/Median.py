from PIL import Image
import numpy as np
#import matplotlib.pyplot as plt

def find_median_image(image1, image2, image3):
    # Open the images
    img1 = Image.open(image1)
    img2 = Image.open(image2)
    img3 = Image.open(image3)

    # Convert images to numpy arrays
    array1 = np.array(img1)
    array2 = np.array(img2)
    array3 = np.array(img3)
    # Calculate the median array
    median_array = np.median([array1, array2, array3], axis=0).astype(np.uint8)
    

    # Create a new image from the median array
    median_image = Image.fromarray(median_array)

    return median_image

if __name__ == "__main__":
    # Specify the paths to the images
    image_path1 = "C:/seniorDesign/git/sddec23-18/test/assets/MedianTest1.jpeg"
    image_path2 = "C:/seniorDesign/git/sddec23-18/test/assets/MedianTest2.jpeg"
    image_path3 = "C:/seniorDesign/git/sddec23-18/test/assets/MedianTest2.jpeg"

    # Find the median image
    result_image = find_median_image(image_path1, image_path2, image_path3)

    # Save or display the result image
    result_image.show()
