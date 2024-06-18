import cv2
import cvui
import numpy as np

# Define constants
WINDOW_NAME = "OpenCV + cvui"
WINDOW_WIDTH = 1920
WINDOW_HEIGHT = 1080
TAB_COUNT = 3

# Load the images
img1 = cv2.imread("C:/seniorDesign/git/sddec23-18/test/assets/appScreenshots/Step1.png")
img2 = cv2.imread("C:/seniorDesign/git/sddec23-18/test/assets/appScreenshots/Step2.png")
img3 = cv2.imread("C:/seniorDesign/git/sddec23-18/test/assets/appScreenshots/Step3.png")

# Resize the images to match the target area
img1 = cv2.resize(img1, (862, 900))
img2 = cv2.resize(img2, (862, 900))
img3 = cv2.resize(img3, (862, 900))

# Initialize the window and tabs
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cvui.init(WINDOW_NAME)
tab_index = 0

# Loop until the user closes the window
while True:
    # Create a blank frame for the tab content
    tab_frame = np.zeros((WINDOW_HEIGHT - 30, WINDOW_WIDTH, 3), dtype=np.uint8)

    # Handle events and draw the UI
    cvui.context(WINDOW_NAME)
    cvui.beginColumn(tab_frame, 10, 10, -1, -1, 6)
    for i in range(TAB_COUNT):
        if cvui.button("Tab %d" % (i + 1)):
            tab_index = i

    cvui.endColumn()
    cvui.beginColumn(tab_frame, 10, 60, -1, -1, 6)

    # Show the appropriate image for the selected tab
    if tab_index == 0:
        img = img1
    elif tab_index == 1:
        img = img2
    else:
        img = img3

    h, w, _ = img.shape
    x_offset = (WINDOW_WIDTH - w) // 2
    y_offset = (WINDOW_HEIGHT - 90 - h) // 2
    tab_frame[y_offset:y_offset+h, x_offset:x_offset+w] = img

    cvui.endColumn()

    # Display the tab content in the window
    cv2.imshow(WINDOW_NAME, tab_frame)

    # Check for keypresses and exit if necessary
    if cv2.waitKey(20) == 27:
        break

# Clean up
cv2.destroyAllWindows()
