import cv2
import numpy as np

# Load the images
    #img1 = cv2.imread("C:/seniorDesign/git/sddec23-18/test/assets/appScreenshots/Step1.png")
    #img2 = cv2.imread("C:/seniorDesign/git/sddec23-18/test/assets/appScreenshots/Step2.png")
    #img3 = cv2.imread("C:/seniorDesign/git/sddec23-18/test/assets/appScreenshots/Step3.png")

#starter = cv2.imread('C:/seniorDesign/git/sddec23-18/test/assets/snorResized.png') #For Screenshots
image = cv2.imread('C:/seniorDesign/git/sddec23-18/test/assets/dog.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply thresholding to the image
_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

# Apply morphological operations to fill in any gaps in the edges
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

# Find contours in the image
contours, hierarchy = cv2.findContours(morph, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Find the contour with the largest area (which should be the document)
largest_contour = max(contours, key=cv2.contourArea)

# Get the four corners of the largest contour using approxPolyDP
epsilon = 0.1*cv2.arcLength(largest_contour,True)
corners = cv2.approxPolyDP(largest_contour, epsilon, True)

# Apply perspective transform to the image
pts1 = np.float32(corners)
pts2 = np.float32([[0,0],[0,500],[500,500],[500,0]])
M = cv2.getPerspectiveTransform(pts1,pts2)
transformed = cv2.warpPerspective(image,M,(500,500))

# Display the original image, the thresholded image, and the transformed image
#cv2.imshow('original1', starter) #For Screenshots
cv2.imshow('original', image)
cv2.imshow('threshold', thresh)
cv2.imshow('transformed', transformed)
cv2.waitKey(0)


# Wrap in function (rectangle)
# Set the color myself instead of having it guess (grey level)
# 
