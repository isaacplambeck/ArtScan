import cv2
import numpy as np

# Load the image
image = cv2.imread('C:/seniorDesign/git/sddec23-18/test/assets/snorResized.png')

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Threshold the grayscale image to create a mask
_, mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

# Invert the mask to select the background
mask_inv = cv2.bitwise_not(mask)

# Create a white background image
background = np.zeros_like(image)
background[:] = (255, 255, 255)

# Combine the background and foreground using the masks
fg = cv2.bitwise_and(image, image, mask=mask_inv)
bg = cv2.bitwise_and(background, background, mask=mask)
result = cv2.add(fg, bg)

# Display the result
cv2.imshow('Result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
