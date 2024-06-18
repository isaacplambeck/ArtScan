import tkinter as tk
from tkinter import *
from PIL import Image, ImageTk #pip install Pillow
from tkinter.filedialog import askopenfilename #file
import cv2 #pip install opencv-python
import numpy as np #pip install numpy
from tkinter import ttk #tab control
import cvui #pip install cvui
import numpy as np
import cv2.aruco as aruco #aruco

def create_window(image, thresh, transformed):
    # Define constants
    WINDOW_NAME = "OpenCV + cvui"
    WINDOW_WIDTH = 1920
    WINDOW_HEIGHT = 1080
    TAB_COUNT = 3

    # Load the images
    #img1 = cv2.imread("C:/seniorDesign/git/sddec23-18/test/assets/appScreenshots/Step1.png")
    #img2 = cv2.imread("C:/seniorDesign/git/sddec23-18/test/assets/appScreenshots/Step2.png")
    #img3 = cv2.imread("C:/seniorDesign/git/sddec23-18/test/assets/appScreenshots/Step3.png")

    # Resize the images to match the target area
    scale_percent1 = 200 # percent of original size
    width1 = int(image.shape[1] * scale_percent1 / 100)
    height1 = int(image.shape[0] * scale_percent1 / 100)
    if height1 > 950 or width1 > 950:
        while True:
            if height1 > 950 or width1 > 950:
                scale_percent1 = scale_percent1 - 5
                width1 = int(image.shape[1] * scale_percent1 / 100)
                height1 = int(image.shape[0] * scale_percent1 / 100)
            else:
                break

    scale_percent2 = 200 # percent of original size
    width2 = int(thresh.shape[1] * scale_percent2 / 100)
    height2 = int(thresh.shape[0] * scale_percent2 / 100)
    if height2 > 950 or width2 > 950:
        while True:
            if height2 > 950 or width2 > 950:
                scale_percent2 -= 5
                width2 = int(thresh.shape[1] * scale_percent2 / 100)
                height2 = int(thresh.shape[0] * scale_percent2 / 100)
            else:
                break

    scale_percent3 = 200 # percent of original size
    width3 = int(transformed.shape[1] * scale_percent3 / 100)
    height3 = int(transformed.shape[0] * scale_percent3 / 100)
    if height3 > 950 or width3 > 950:
        while True:
            if height3 > 950 or width3 > 950:
                scale_percent3 -= 5
                width3 = int(transformed.shape[1] * scale_percent3 / 100)
                height3 = int(transformed.shape[0] * scale_percent3 / 100)
            else:
                break

    img1 = cv2.resize(image, (width1, height1))
    img2 = cv2.resize(thresh, (width2, height2))
    img3 = cv2.resize(transformed, (width3, height3))

    #If it's grayscale
    if len(img1.shape) == 2:
        img1 = np.expand_dims(img1, axis=2)
    if len(img2.shape) == 2:
        img2 = np.expand_dims(img2, axis=2)
    if len(img3.shape) == 2:
        img3 = np.expand_dims(img3, axis=2)

    # Initialize the window and tabs
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cvui.init(WINDOW_NAME)
    tab_index = 0

    # Loop until the user closes the window
    while True:
    # Create a blank frame for the tab content
        tab_frame = np.zeros((WINDOW_HEIGHT - 30, WINDOW_WIDTH, 3), dtype=np.uint8) #black background
        #tab_frame = np.full((WINDOW_HEIGHT - 30, WINDOW_WIDTH, 3), (255, 255, 255), dtype=np.uint8) #white background
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

def upload_image():
    #Get file
    filename = askopenfilename()
    print(f"Selected file: {filename}")
    image = cv2.imread(filename)
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
    #Call window with tabs function
    create_window(image, thresh, transformed)
    
window = Tk()
window.geometry("1280x720")
#window.attributes('-fullscreen',True)
window.title("ArtScan")

appLogoUpdated = 'C:/seniorDesign/git/sddec23-18/workingDirectory/assets/ArtscanLogoResize.png'

logo = Image.open(appLogoUpdated)
logo_tk = ImageTk.PhotoImage(logo)

label = tk.Label(window, image=logo_tk)
label.pack()

#Button(window, text = "Press", command = create_window, height=20,width=40).pack()

uploadBtn = Button(window, text = "Upload Image", command = upload_image, height=10,width=35,padx=10, pady=10).pack()


cv2.destroyAllWindows()
window.mainloop()