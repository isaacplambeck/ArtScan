from PIL import Image
import numpy as np
#import matplotlib.pyplot as plt

def find_median_image(setOfCroppedImages):
    # Open the images

    setOfNPImages = []
    for i in range(len(setOfCroppedImages)):
        img = Image.open(setOfCroppedImages[i])
        nparrayImage = np.array(img)
        setOfNPImages.append(nparrayImage)

    median_array = np.median(setOfNPImages, axis=0).astype(np.uint8)

    # Create a new image from the median array
    median_image = Image.fromarray(median_array)

    return median_image

if __name__ == "__main__":
    # Specify the paths to the images
    image_path1 = "C:/seniorDesign/git/sddec23-18/test/assets/out.jpg"
    image_path2 = "C:/seniorDesign/git/sddec23-18/test/assets/outSecond.jpg"
    image_path3 = "C:/seniorDesign/git/sddec23-18/test/assets/outRed.jpg"
    setOfCroppedImages = [image_path1]
    

    # Find the median image
    result_image = find_median_image(setOfCroppedImages)

    # Save or display the result image
    result_image.show()
