import tkinter as tk
from tkinter import *
from PIL import Image #pip install Pillow
from tkinter.filedialog import askopenfilename #file
import cv2 #pip install opencv-python
import numpy as np #pip install numpy
from tkinter import ttk #tab control
import cvui #pip install cvui
import numpy as np
import cv2.aruco as aruco #aruco
from tkinter import filedialog
import sys
import os.path as pathfile
from imutils.perspective import four_point_transform
from skimage import exposure
import argparse
import imutils
import math
from os.path import exists
from io import StringIO
from threading import Thread

def pantoneTest(image, percentError, scalePercent):
    #pantone = cv2.imread("C:/seniorDesign/git/sddec23-18/test/assets/doomPantone.jpg")
    setOfImages = []
    ranHeightWidth = 0
    for i in range(len(image)):
        pantone = cv2.imread(image[i])
        #cv2.imshow(image[i])
        # scale_percent1 = 200  # percent of the original size
        # width1 = int(pantone.shape[1] * scale_percent1 / 100)
        # height1 = int(pantone.shape[0] * scale_percent1 / 100)
        
        # if height1 > 950 or width1 > 950:
        #     while True:
        #         if height1 > 950 or width1 > 950:
        #             scale_percent1 -= 5
        #             width1 = int(pantone.shape[1] * scale_percent1 / 100)
        #             height1 = int(pantone.shape[0] * scale_percent1 / 100)
        #         else:
        #             break
        
        # pantoneResized = cv2.resize(pantone, (width1, height1))
        #aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)

        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
        parameters =  cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(dictionary, parameters)

        # Detect ArUco markers
        #corners, ids, _ = aruco.detectMarkers(pantoneResized, aruco_dict)
        markerCorners, markerIds, rejectedCandidates = detector.detectMarkers(pantone)
        
        line_color = (0, 255, 0)
        
        for i in range(len(markerIds)):
            marker_id = markerIds[i][0]  # Get the ID of the current marker
            
            # Check if you want to change the corner for a specific marker ID
            if marker_id == 1:
                # Swap the top-left and top-right corners
                markerCorners[i][0][0], markerCorners[i][0][2] = markerCorners[i][0][3], markerCorners[i][0][0]
                markerId1BeforeNP = markerCorners[i][0]

        for i in range(len(markerIds)):
            marker_id = markerIds[i][0]  # Get the ID of the current marker
            
            # Check if you want to change the corner for a specific marker ID
            if marker_id == 2:
                # Swap the top-left and top-right corners
                markerCorners[i][0][0], markerCorners[i][0][2] = markerCorners[i][0][1], markerCorners[i][0][0]
                markerId2BeforeNP = markerCorners[i][0]

        for i in range(len(markerIds)):
            marker_id = markerIds[i][0]  # Get the ID of the current marker
            
            # Check if you want to change the corner for a specific marker ID
            if marker_id == 0:
                # Swap the top-left and top-right corners
                markerCorners[i][0][0], markerCorners[i][0][2] = markerCorners[i][0][2], markerCorners[i][0][0]
                markerId0BeforeNP = markerCorners[i][0]

        for i in range(len(markerIds)):
            marker_id = markerIds[i][0]  # Get the ID of the current marker
            
            # Check if you want to change the corner for a specific marker ID
            if marker_id == 3:
                # Swap the top-left and top-right corners
                #bottom_right_cornerBeforeNP = markerCorners[i][0]
                markerId3BeforeNP = markerCorners[i][0]

        pantoneImageWithLines = pantone.copy()

        aruco.drawDetectedMarkers(pantoneImageWithLines, markerCorners, markerIds) #shows red markers on corners touching image

        print("Marker Corners: ", markerCorners[0])
        print("Marker IDs: ", markerIds)
        #arrMarker = markerIds
        marker0 = np.where(markerIds == 0)
        print("Marker 0 (top left): ", marker0)

        # top_left_cornerBeforeNP = markerCorners[0][0]
        # top_right_cornerBeforeNP = markerCorners[1][0]
        # bottom_right_cornerBeforeNP = markerCorners[2][0]
        # bottom_left_cornerBeforeNP = markerCorners[3][0]

        markerId0 = np.int32(markerId0BeforeNP)[0] #top LEFT
        markerId1 = np.int32(markerId1BeforeNP)[0] #top RIGHT
        markerId2 = np.int32(markerId2BeforeNP)[0] #bottom LEFT
        markerId3 = np.int32(markerId3BeforeNP)[0] #bottom RIGHT

        
        # cv2.line(pantoneImageWithLines, tuple(markerId3), tuple(markerId3), line_color, 200)
        # #cv2.line(pantoneImageWithLines, tuple(bottom_right_corner[0]), tuple(bottom_right_corner[0]), line_color, 200)
        # cv2.imshow("markers", cv2.resize(pantoneImageWithLines, (1920, 1080)))
        

        if len(markerCorners) > 0:
            marker_index = 0  # Change this to the index of the marker you're interested in
            corners = markerCorners[marker_index][0]

            # Extract the coordinates of the individual corners
            # top_left_corner = corners[0]
            # top_right_corner = corners[1]
            # bottom_right_corner = corners[2]
            # bottom_left_corner = corners[3]

            # # Print or use the coordinates as needed
            # print("Top Left Corner:", np.int32(top_left_corner))
            # print("Top Right Corner:", top_right_corner)
            # print("Bottom Right Corner:", bottom_right_corner)
            # print("Bottom Left Corner:", bottom_left_corner)

            corners = np.int32(corners)
            line_color = (0, 255, 0)

            #0-1
            #0-2
            #1-3
            #2-3
            ran = bool(False)
            for i in range(len(markerCorners)):
                for j in range(i + 1, len(markerCorners)):
                    # Calculate the centers of each marker
                    if i == 0 and j == 1 or i == 0 and j == 2 or i == 1 and j == 3 or i == 2 and j == 3:
                        
                        if(ran):
                            center1 = markerCorners[i][0]
                            center2 = markerCorners[j][0]
                            center1 = np.int32(center1)
                            center2 = np.int32(center2)
                            #cv2.line(pantoneImageWithLines, tuple(center1[0]), tuple(center2[0]), line_color, 2)
                            
                        else:
                            center3 = markerCorners[i][0]
                            center4 = markerCorners[j][0]
                            center3 = np.int32(center3)
                            center4 = np.int32(center4)
                            #cv2.line(pantoneImageWithLines, tuple(center3[0]), tuple(center4[0]), line_color, 2)
                            ran = bool(True)


            # print("center1[0]", center1[0])
            # print("center2[0]", center2[0])
            # print("center3[0]", center3[0])
            # print("center4[0]", center4[0])

            # cv2.circle(pantoneResized, center1[0], 5, [255,0,0], thickness=3, lineType=8, shift=0)
            # cv2.circle(pantoneResized, center2[0], 5, [255,0,0], thickness=3, lineType=8, shift=0)
            # cv2.circle(pantoneResized, center3[0], 5, [255,0,0], thickness=3, lineType=8, shift=0)
            # cv2.circle(pantoneResized, center4[0], 5, [255,0,0], thickness=3, lineType=8, shift=0)

            # cv2.imshow("markers", cv2.resize(pantoneImageWithLines, (1920, 1080)))
            # top_left = center1[0]
            # top_right = center2[0]
            # bottom_left = center3[0]
            # bottom_right = center4[0]

            print("NP: Top Left Corner:", markerId0)
            print("NP: Top Right Corner:", markerId1)
            print("NP: Bottom Right Corner:", markerId3)
            print("NP: Bottom Left Corner:", markerId2)

            if(ranHeightWidth == 0):
                heightCorners = markerId3[1] - markerId1[1]
                widthCorner = markerId1[0] - markerId0[0]
                ranHeightWidth = 1
            # heightCorners = 2000
            # widthCorner = 1500


            # bottom_left_corner = np.int32(top_left_cornerBeforeNP) #bottom LEFT
            # bottom_right_corner = np.int32(top_right_cornerBeforeNP) #bottom RIGHT
            # top_left_corner = np.int32(bottom_right_cornerBeforeNP) #top LEFT
            # top_right_corner = np.int32(bottom_left_cornerBeforeNP) #top RIGHT

            # Define the destination points for perspective transform (a perfect rectangle)
            #width = 500  # Adjust the width to your desired output
            #height = 500  # Adjust the height to your desired output
            height1, width1, channels = pantone.shape
            print("height1", height1)
            print("and width1", width1)
            #dst_points = np.float32([[0, 0], [width1, 0], [width1, height1], [0, height1]])
            dst_points = np.float32([[0, 0], [widthCorner, 0], [widthCorner, heightCorners], [0, heightCorners]])

            # Get the corner points as source points
            #src_points = np.float32([top_left, top_right, bottom_right, bottom_left])
            num = int(percentError)
            markerId0[0] += num
            markerId0[1] += num
            markerId1[0] -= num
            markerId1[1] += num
            markerId3[0] -= num
            markerId3[1] -= num
            markerId2[0] += num
            markerId2[1] -= num
            src_points = np.float32([markerId0, markerId1, markerId3, markerId2])

            # Perform perspective transform to get a bird's eye view
            perspective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
            #birdseye_view = cv2.warpPerspective(pantone, perspective_matrix, (width1, height1))
            birdseye_view = cv2.warpPerspective(pantone, perspective_matrix, (widthCorner, heightCorners))

            # Crop the image based on the transformed perspective
            cropped_image = birdseye_view

            #final_image = cv2.resize(cropped_image, (width1, height1))

            #arr = [set of images to return]
            #return arr
            cv2.imwrite("PantoneCropResizeOutput.jpg", cropped_image) #The Scaling Issue is here
            setOfImages.append(cropped_image)
    return setOfImages    
    #return cropped_image, pantoneImageWithLines

def find_median_image(setOfCroppedImages):
    # Open the images

    setOfNPImages = []
    for i in range(len(setOfCroppedImages)):
        img = cv2.cvtColor(setOfCroppedImages[i], cv2.COLOR_BGR2RGB)
        setOfNPImages.append(img)

    median_array = np.median(setOfNPImages, axis=0).astype(np.uint8)

    # Create a new image from the median array
    median_image = Image.fromarray(median_array)

    return median_image

def create_window(image, percentError, scalePercent):
    # Define constants

    pantoneResult = pantoneTest(image, percentError, scalePercent)
    medianOutput = find_median_image(pantoneResult)
    medianCv2Img = cv2.cvtColor(np.array(medianOutput), cv2.COLOR_RGB2BGR)
    pantoneResult.append(medianCv2Img)
    #medianOutput.show()
    heightPan, widthPan = pantoneResult[0].shape[:2]
    WINDOW_NAME = "OpenCV + cvui"
    #WINDOW_WIDTH = widthPan
    #WINDOW_HEIGHT = heightPan
    WINDOW_WIDTH = 1920
    WINDOW_HEIGHT = 1300
    TAB_COUNT = len(pantoneResult)

    img = pantoneTest(image, percentError, scalePercent)
    


    # Initialize the window and tabs
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cvui.init(WINDOW_NAME)
    tab_index = 0

    # Loop until the user closes the window
    while True:
    # Create a blank frame for the tab content
        #tab_frame = np.zeros((WINDOW_HEIGHT - 30, WINDOW_WIDTH, 3), dtype=np.uint8) #black background
        tab_frame = np.zeros((WINDOW_HEIGHT - 30, WINDOW_WIDTH, 3), dtype=np.uint8) #black background
        #tab_frame = np.full((WINDOW_HEIGHT - 30, WINDOW_WIDTH, 3), (255, 255, 255), dtype=np.uint8) #white background
        # Handle events and draw the UI
        cvui.context(WINDOW_NAME)
        cvui.beginColumn(tab_frame, 10, 10, -1, -1, 6)
        for i in range(TAB_COUNT):
            if cvui.button("Tab %d" % (i + 1)):
                tab_index = i
            #if i == (len(pantoneResult) - 1):
                #cvui.button("Median Image")
                #tab_index = (len(pantoneResult) - 1)
            
        #cvui.button("Median Image")

        cvui.endColumn()
        cvui.beginColumn(tab_frame, 10, 60, -1, -1, 6)

        #pantoneResult = pantoneTest()

        #for i in range(pantoneResult):
        if tab_index == 0:
            img = pantoneResult[0]
            #WINDOW_WIDTH = 1920
            #WINDOW_HEIGHT = 1080
        elif tab_index == TAB_COUNT - 1:
            img = pantoneResult[TAB_COUNT - 1]
        else:
            img = pantoneResult[tab_index]
            #WINDOW_WIDTH = 1920
            #WINDOW_HEIGHT = 1300
            #tab_frame = np.zeros((WINDOW_HEIGHT - 30, WINDOW_WIDTH, 3), dtype=np.uint8) #black background
       # elif tab_index == 2:
            #img = img3
        #elif tab_index == 3:
            
        #else:
            

        # Calculate the scaling factors for width and height
        scale_w = WINDOW_WIDTH / float(img.shape[1])
        scale_h = (WINDOW_HEIGHT - 90) / float(img.shape[0])

        # Choose the smaller scaling factor to maintain the aspect ratio
        scale_factor = min(scale_w, scale_h)

        # Scale the image using the calculated factor
        scaled_img = cv2.resize(img, None, fx=scale_factor, fy=scale_factor)

        # Calculate offsets for the scaled image
        h, w, _ = scaled_img.shape
        x_offset = (WINDOW_WIDTH - w) // 2
        y_offset = (WINDOW_HEIGHT - 90 - h) // 2

        # Place the scaled image onto the frame
        tab_frame[y_offset:y_offset+h, x_offset:x_offset+w] = scaled_img
        #cvui.text(tab_frame, 90, 50, "Hello world")
        # count = [0]
        # cvui.counter(tab_frame, 90, 50, count)
        # print(count)
        # status = cvui.iarea(x_offset, y_offset, w, h)

        # if status == cvui.CLICK:
        #     print("Mouse pointer is at (%d,%d)", cvui.mouse().x, cvui.mouse().y)

        cvui.endColumn()
        # Display the tab content in the window
        cv2.imshow(WINDOW_NAME, tab_frame)

        # Check for keypresses and exit if necessary
        if cv2.waitKey(20) == 27:
            break

def open_new_window():

    #root.withdraw()
    #root.quit()
    new_window = tk.Toplevel(root)
    new_window.title("Console Output")
    text_area = tk.Text(new_window, wrap=tk.WORD)
    text_area.pack(expand=True, fill=tk.BOTH)
    message = "test"
    text_area.insert(tk.END, message)
    
    
    # sys.stdout = StdoutRedirector(text_area)
   

    # root = tk.Tk()
    # root.title("Console Output for Color Correction")
    # text_area = tk.Text(root, wrap=tk.WORD)
    # text_area.pack(expand=True, fill=tk.BOTH)
    # sys.stdout = StdoutRedirector(text_area)

    #new_window.mainloop()
    #startColorCorrection(image)

def close_window():
    root.withdraw()
    #root.quit()
   # open_new_window()
    #root.after(100, root.withdraw())  # Hide the root window
    #root.after(100, open_new_window)
    #root.quit()  # Quit the mainloop
    
    #root.destroy()  # Close the root window
    #root.after(500, open_new_window)  # Open a new window after a small delay
    #open_new_window()

def start_program(image):
    t = Thread(target=startColorCorrection(image), daemon=True)
    t.start()

def start_main(image):
    t = Thread(target=startColorCorrection, args=(image,), daemon=True)
    t.start()

def startColorCorrection(image):
    # construct the argument parser and parse the arguments
    # ap = argparse.ArgumentParser()
    # ap.add_argument("-r", "--reference", required=True,
    #                 help="path to the input reference image")
    # ap.add_argument("-v", "--view", required=False, default=False, action='store_true',
    #                 help="Image Preview?")
    # ap.add_argument("-o", "--output", required=False, default=False,
    #                 help="Image Output Path")
    # ap.add_argument("-i", "--input", required=True,
    #                 help="path to the input image to apply color correction to")
    # args = vars(ap.parse_args())

    # load the reference image and input images from disk
    # root.destroy()
    # newStdout = tk.Tk()
    # newStdout.title("Console Output")
    # text_area = tk.Text(newStdout, wrap=tk.WORD)
    # text_area.pack(expand=True, fill=tk.BOTH)
    # newStdout.mainloop

    #close_window()
    #open_new_window()
    
    stringio = StringIO()
    previous_stdout = sys.stdout
    sys.stdout = stringio

    
    #sys.stdout = StdoutRedirector(text_area)


    StdOutColor.insert(tk.END, "[INFO] loading images...")
    print("[INFO] loading images...")

    #sys.stdout = previous_stdout
    myString = stringio.getvalue()  
    StdOutColor.insert(tk.END, myString)
    StdOutColor.see("end")

    referenceImg = "C:/seniorDesign/git/sddec23-18/Color Correction/PantoneRef2.jpg"
    # raw = cv2.imread(args["reference"])
    # img1 = cv2.imread(args["input"])
    file_exists = pathfile.isfile(referenceImg)
    print(file_exists)

    if not file_exists:
        print('[WARNING] Referenz File not exisits '+str(referenceImg))
        #sys.stdout = previous_stdout
        myString = stringio.getvalue()  
        StdOutColor.insert(tk.END, myString)
        StdOutColor.see("end")
        #StdOutColor.insert(tk.END, '[WARNING] Referenz File not exisits '+str(referenceImg))
        sys.exit()


    raw = cv2.imread(referenceImg)
    img1 = cv2.imread(image[0])

    (h, w) = img1.shape[:2]

    # compute the center coordinate of the image
    (cX, cY) = (w // 2, h // 2)

    # crop the image into four parts which will be labelled as
    # top left, top right, bottom left, and bottom right.
    img1TopLeft = img1[0:cY, 0:cX]
    img1TopRight = img1[0:cY, cX:w]
    img1BottomLeft = img1[cY:h, 0:cX]
    img1BottomRight = img1[cY:h, cX:w]

    img1TopLeft_L = cv2.resize(img1TopLeft, (3*cX, 3*cY))
    img1TopRight_L = cv2.resize(img1TopRight, (3*cX, 3*cY))
    img1BottomRight_L = cv2.resize(img1BottomRight, (3*cX, 3*cY))
    img1BottomLeft_L = cv2.resize(img1BottomLeft, (3*cX, 3*cY))


    img1_L = img1TopLeft_L
    #img1 = img1TopLeft

    img2_L = img1TopRight_L
    #img2 = img1TopRight

    img3_L = img1BottomRight_L
    #img3 = img1BottomRight

    img4_L = img1BottomLeft_L
    #img4 = img1BottomLeft
    # resize the reference and input images

    #raw = imutils.resize(raw, width=301)
    #img1 = imutils.resize(img1, width=301)
    # raw = imutils.resize(raw, width=600)
    # img1 = imutils.resize(img1, width=600)
    # display the reference and input images to our screen
    if viewBool.get() == 1:
        cv2.imshow("Reference", raw)
        resizeImg1 = cv2.resize(img1, (1300, 1300))
        cv2.imshow("Input", resizeImg1)
        cv2.waitKey(5000)


    # find the coordinates of the aruco markers in image
    print("[INFO] finding aruco markers")
    #StdOutColor.insert(tk.END, "[INFO] finding aruco markers")
    #sys.stdout = previous_stdout
    myString = stringio.getvalue()  
    StdOutColor.insert(tk.END, myString)
    StdOutColor.see("end")
    arucoCoords = find_aruco_markers(img1)
    if arucoCoords is not None:
        #StdOutColor.insert(tk.END, "Found Aruco Marker Coordinates ", int(arucoCoords[0][0]))
        #sys.stdout = previous_stdout
        print("Found Aruco Marker Coordinates ", int(arucoCoords[0][0]))
        myString = stringio.getvalue()  
        StdOutColor.insert(tk.END, myString)
        StdOutColor.see("end")

    imgTopLeft = img1[(int(arucoCoords[0][1]) - 150):(int(arucoCoords[0][1]) + 350), (int(arucoCoords[0][0]) + 100):(int(arucoCoords[0][0]) + 800)]
    (h, w) = imgTopLeft.shape[:2]
    imgTopLeft = cv2.resize(imgTopLeft, (int(3*w), int(3*h)))

    imgTopRight = img1[(int(arucoCoords[1][1]) - 150):(int(arucoCoords[1][1]) + 350), (int(arucoCoords[1][0]) - 800):(int(arucoCoords[1][0]))]
    (h, w) = imgTopRight.shape[:2]
    imgTopRight = cv2.resize(imgTopRight, ((2*w), (2*h)))

    imgBottomRight = img1[(int(arucoCoords[2][1]) - 300):(int(arucoCoords[2][1]) + 150), (int(arucoCoords[2][0]) - 800):(int(arucoCoords[2][0]))]
    (h, w) = imgBottomRight.shape[:2]
    imgBottomRight = cv2.resize(imgBottomRight, (int(3*w), int(3*h)))

    imgBottomLeft = img1[(int(arucoCoords[3][1]) - 300):(int(arucoCoords[3][1]) + 150), (int(arucoCoords[3][0]) + 100):(int(arucoCoords[3][0]) + 800)]
    (h, w) = imgBottomLeft.shape[:2]
    imgBottomLeft = cv2.resize(imgBottomLeft, (int(3*w), int(3*h)))





    # find the color matching card in each image
    print("[INFO] finding color matching cards...")
    myString = stringio.getvalue()  
    StdOutColor.insert(tk.END, myString)
    StdOutColor.see("end")
    print("Attempting to find card in Reference")
    myString = stringio.getvalue()  
    StdOutColor.insert(tk.END, myString)
    StdOutColor.see("end")
    rawCard = find_color_card(raw)
    if rawCard is not None:
        print("Found Reference.")
        myString = stringio.getvalue()  
        StdOutColor.insert(tk.END, myString)
        StdOutColor.see("end")

    print("Attempting to find card in Image1")
    myString = stringio.getvalue()  
    StdOutColor.insert(tk.END, myString)
    StdOutColor.see("end")
    imageCard1 = find_color_card(imgTopLeft)
    if imageCard1 is not None:
        print("Found Image1")
        myString = stringio.getvalue()  
        StdOutColor.insert(tk.END, myString)
        StdOutColor.see("end")

    print("Attempting to find card in Image2")
    myString = stringio.getvalue()  
    StdOutColor.insert(tk.END, myString)
    StdOutColor.see("end")
    imageCard2 = find_color_card(imgTopRight)
    if imageCard2 is not None:
        print("Found Image2")
        myString = stringio.getvalue()  
        StdOutColor.insert(tk.END, myString)
        StdOutColor.see("end")

    print("Attempting to find card in Image3")
    myString = stringio.getvalue()  
    StdOutColor.insert(tk.END, myString)
    StdOutColor.see("end")
    imageCard3 = find_color_card(imgBottomRight)
    if imageCard3 is not None:
        print("Found Image3")
        myString = stringio.getvalue()  
        StdOutColor.insert(tk.END, myString)
        StdOutColor.see("end")

    print("Attempting to find card in Image4")
    myString = stringio.getvalue()  
    StdOutColor.insert(tk.END, myString)
    StdOutColor.see("end")
    StdOutColor.see("end")
    imageCard4 = find_color_card(imgBottomLeft)
    if imageCard4 is not None:
        print("Found Image4")
        myString = stringio.getvalue()  
        StdOutColor.insert(tk.END, myString)
        StdOutColor.see("end")


    # if the color matching card is not found in either the reference
    # image or the input image, gracefully exit
    if rawCard is None or imageCard1 is None or imageCard2 is None or imageCard3 is None or imageCard4 is None:
        print("[INFO] could not find color matching card in all images")
        myString = stringio.getvalue()  
        StdOutColor.insert(tk.END, myString)
        StdOutColor.see("end")
        sys.exit(0)

    # show the color matching card in the reference image and input image,
    # respectively
    if viewBool.get() == 1:
        cv2.imshow("Reference Color Card", rawCard)
        cv2.waitKey(2000)
        cv2.imshow("Input Color Card 1", imageCard1)
        cv2.waitKey(2000)
        cv2.imshow("Input Color Card 2", imageCard2)
        cv2.waitKey(2000)
        cv2.imshow("Input Color Card 3", imageCard3)
        cv2.waitKey(2000)
        cv2.imshow("Input Color Card 4", imageCard4)
        cv2.waitKey(2000)

    # apply histogram matching from the color matching card in the
    # reference image to the color matching card in the input image
    print("[INFO] matching images...")
    myString = stringio.getvalue()  
    StdOutColor.insert(tk.END, myString)
    StdOutColor.see("end")

    print("[INFO] This may take awhile...")
    myString = stringio.getvalue()  
    StdOutColor.insert(tk.END, myString)
    StdOutColor.see("end")

    # imageCard2 = exposure.match_histograms(img1, ref,
    # inputCard = exposure.match_histograms(inputCard, referenceCard, multichannel=True)
    result2 = match_histograms_mod(imageCard1, imageCard2, imageCard3, imageCard4, rawCard, img1, arucoCoords)
    
    # show our input color matching card after histogram matching
    #cv2.imshow("Input Color Card After Matching", result2)
    #cv2.waitKey(2000)


    if viewBool.get() == 1:
        result2Resize = cv2.resize(result2, (1300, 1300))
        cv2.imshow("result2", result2Resize)
        cv2.waitKey(5000)

    outputFile = input("Enter the output filename: ")
    myString = stringio.getvalue()  
    StdOutColor.insert(tk.END, myString)
    StdOutColor.see("end")

    if outputFile:
        file_ok = exists(outputFile.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')))

        if file_ok:
            cv2.imwrite(outputFile, result2)
            print("[SUCCESSUL] Your Image was written to: "+outputFile+"")
            myString = stringio.getvalue()  
            StdOutColor.insert(tk.END, myString)
            StdOutColor.see("end")
        else:
            print("[WARNING] Sorry, But this is no valid Image Name "+outputFile+"\nPlease Change Parameter!")
            myString = stringio.getvalue()  
            StdOutColor.insert(tk.END, myString)
            StdOutColor.see("end")

    if viewBool.get() == 1:
        cv2.waitKey(0)

    if not viewBool.get() == 1:
        if not outputFile:
            print('[EMPTY] You Need at least one Paramter "--view" or "--output".')
            myString = stringio.getvalue()  
            StdOutColor.insert(tk.END, myString)
            StdOutColor.see("end")

def upload_imageCrop():
    #Get file
    # filename = askopenfilename()
    filename = filedialog.askopenfilenames()
    print(f"Selected file: {filename}")
    #image = cv2.imread(filename)
    image = filename


    #popupwin()
    create_window(image, 0, 0)

def upload_imageCropEnterCropNumber():
    #Get file
    # filename = askopenfilename()
    filename = filedialog.askopenfilenames()
    print(f"Selected file: {filename}")
    #image = cv2.imread(filename)
    image = filename


    popupwin(image)
    #create_window(image)

def close_win(top, entry, image, entrySizeX, entrySizeY, entryScalePercent):
   percentError = entry.get()
   outputX = entrySizeX.get()
   outputY = entrySizeY.get()
   scalePercent = entryScalePercent.get()

   print(entry.get())
   top.destroy()
   create_window(image, percentError, scalePercent) #Pass outputX and outputY too later
   
def insert_val(e):
   e.insert(0, "Hello World!")

def popupwin(image):
   #Create a Toplevel window
    top= Toplevel(root)
    top.geometry("850x350")

    #Create an Entry Widget in the Toplevel window
    
    label= Label(top, text="Enter Extra Crop Size Pixel Amount (Default 0)", font= ('Helvetica 15 bold'))
    label.pack(pady=20)
    entry= Entry(top, width= 25)
    entry.pack()
    label= Label(top, text="Enter Output Image Size (X then Y, Default is input Image Size)", font= ('Helvetica 15 bold'))
    label.pack(pady=20)
    entrySizeX= Entry(top, width= 25)
    entrySizeX.pack()
    entrySizeY= Entry(top, width= 25, text="Y")
    entrySizeY.pack()
    label= Label(top, text="Enter Scale Image Percent (Default 0)", font= ('Helvetica 15 bold'))
    label.pack(pady=20)
    entryScalePercent= Entry(top, width= 25)
    entryScalePercent.pack()
    button= Button(top, text="Ok", command=lambda:close_win(top, entry, image, entrySizeX, entrySizeY, entryScalePercent))
    button.pack(pady=5, side= TOP)
    

   #Create a Button to print something in the Entry widget
   #Button(top,text= "Insert", command= lambda:insert_val(entry)).pack(pady= 5,side=TOP)
   #Create a Button Widget in the Toplevel Window
   #button= Button(top, text="Ok", command=lambda:close_win(top))
   #button.pack(pady=5, side= TOP)

def upload_imageColor():
    #Get file
    # filename = askopenfilename()
    filename = filedialog.askopenfilenames()
    print(f"Selected file: {filename}")
    #image = cv2.imread(filename)
    image = filename
    
    #startColorCorrection(image)
    start_main(image)
    #start_program(image)
    #close_window()
    #open_new_window(image)
    #redirect_output(image)

def start_appCrop():
    # Place your app startup logic here
    # For demonstration purposes, a simple print statement is used
    print("Art Scanning App is starting...")
    print(percentErrorBool)
    if percentErrorBool.get() == 1:
        upload_imageCropEnterCropNumber()
    else:
        upload_imageCrop()

def start_appColor():
    #StdOutWindow.deiconify()
    StdOutColor.pack(expand=True, fill=tk.BOTH)
    # Place your app startup logic here
    # For demonstration purposes, a simple print statement is used
    print("Art Scanning App is starting...")
    upload_imageColor()

# def redirect_output(image):

#     text_area = tk.Text(root, wrap=tk.WORD)
#     text_area.pack(expand=True, fill=tk.BOTH)

#     def update_text_widget():
#         message = sys.stdout.getvalue()  # Get the stdout content
#         text_area.insert(tk.END, message)  # Update text widget with stdout content
#         text_area.see(tk.END)  # Scroll to the end to show latest messages
#         text_area.after(100, update_text_widget)  # Schedule the next update

#     #sys.stdout = sys.StringIO()  # Redirect stdout to a StringIO object
#     update_text_widget()  # Start updating the text widget periodically
#     startColorCorrection(image)


def find_aruco_markers(image):
    # load the ArUCo dictionary, grab the ArUCo parameters, and
    # detect the markers in the input image
    #arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_250)
    arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
    arucoParams = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(arucoDict, arucoParams)

    (corners, ids, rejected) = detector.detectMarkers(image)
    
    # try to extract the coordinates of the color correction card
    try:
        # otherwise, we've found the four ArUco markers, so we can
        # continue by flattening the ArUco IDs list
        ids = ids.flatten()

        # extract the top-left marker
        i = np.squeeze(np.where(ids == 0))
        topLeft = np.squeeze(corners[i])[0]
        print("ArUco topLeft = ",topLeft)

        # extract the top-right marker
        i = np.squeeze(np.where(ids == 1))
        topRight = np.squeeze(corners[i])[1]
        print("AruCo topRight = ",topRight)

        # extract the bottom-right marker
        i = np.squeeze(np.where(ids == 3))
        bottomRight = np.squeeze(corners[i])[2]
        print("ArUco bottomRight = ",bottomRight)

        # extract the bottom-left marker
        i = np.squeeze(np.where(ids == 2))
        bottomLeft = np.squeeze(corners[i])[3]
        print("ArUco bottomLeft = ",bottomLeft)

        print("ArUco Coordinates: topLeft is",topLeft,", topRight is",topRight,", bottomRight is",bottomRight,", bottomLeft is",bottomLeft)
    # we could not find ArUco coordinates, so gracefully return
    except:
        return None
    
    # build our list of reference points and apply a perspective
    # transform to obtain a top-down, bird’s-eye view of the color
    # matching card
    arucoCoords = np.array([topLeft, topRight,
                           bottomRight, bottomLeft])
    
    # return the coordinates of the ArUco markers
    return arucoCoords


def find_color_card(image):
    # load the ArUCo dictionary, grab the ArUCo parameters, and
    # detect the markers in the input image
    #arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_ARUCO_ORIGINAL)
    #arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_ARUCO_ORIGINAL)
    arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
    arucoParams = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(arucoDict, arucoParams)

    

    # Edit Parameters for aruco detection
    arucoParams.adaptiveThreshConstant = 6
    arucoParams.adaptiveThreshWinSizeMax = 70
    #arucoParams.adaptiveThreshWinSizeMin = 2
    arucoParams.adaptiveThreshWinSizeStep = 6
    # arucoParams.aprilTagCriticalRad = 0
    # #arucoParams.aprilTagMaxNmaxima = 15
    # #arucoParams.aprilTagMinClusterPixels = 3
    # #arucoParams.aprilTagMinWhiteBlackDiff = 3
    # arucoParams.cornerRefinementMinAccuracy = 0.05
    # arucoParams.cornerRefinementWinSize = 7
    # #arucoParams.errorCorrectionRate = 0.4
    # arucoParams.maxMarkerPerimeterRate = 10
    # arucoParams.minMarkerPerimeterRate = 0.01
    # arucoParams.minOtsuStdDev = 3
    # arucoParams.minDistanceToBorder = 1
    # #arucoParams.useAruco3Detection = False
    # arucoParams.minMarkerDistanceRate = 0.01
    # #arucoParams.perspectiveRemoveIgnoredMarginPerCell = 1
    # #arucoParams.perspectiveRemovePixelPerCell = 7
    # arucoParams.polygonalApproxAccuracyRate = .02
    



    (corners, ids, rejected) = detector.detectMarkers(image)

    # try to extract the coordinates of the color correction card
    try:
        # otherwise, we've found the four ArUco markers, so we can
        # continue by flattening the ArUco IDs list
        ids = ids.flatten()

        # extract the top-left marker
        i = np.squeeze(np.where(ids == 923))
        topLeft = np.squeeze(corners[i])[0]
        print("topLeft = ",topLeft)

        # extract the top-right marker
        i = np.squeeze(np.where(ids == 1001))
        topRight = np.squeeze(corners[i])[1]
        print("topRight = ",topRight)

        # extract the bottom-right marker
        i = np.squeeze(np.where(ids == 241))
        bottomRight = np.squeeze(corners[i])[2]
        print("bottomRight = ",bottomRight)

        # extract the bottom-left marker
        i = np.squeeze(np.where(ids == 1007))
        bottomLeft = np.squeeze(corners[i])[3]
        print("bottomLeft = ",bottomLeft)
        
        print("Coordinates: topLeft is",topLeft,", topRight is",topRight,", bottomRight is",bottomRight,", bottomLeft is",bottomLeft)
    # we could not find color correction card, so gracefully return
    except:
        return None

    # build our list of reference points and apply a perspective
    # transform to obtain a top-down, bird’s-eye view of the color
    # matching card
    cardCoords = np.array([topLeft, topRight,
                           bottomRight, bottomLeft])
    
    card = four_point_transform(image, cardCoords)
    # return the color matching card to the calling function
    return card


def _match_cumulative_cdf_mod(source, template, full):
    """
    Return modified full image array so that the cumulative density function of
    source array matches the cumulative density function of the template.
    """
    src_values, src_unique_indices, src_counts = np.unique(source.ravel(),
                                                           return_inverse=True,
                                                           return_counts=True)
    tmpl_values, tmpl_counts = np.unique(template.ravel(), return_counts=True)

    # calculate normalized quantiles for each array
    src_quantiles = np.cumsum(src_counts) / source.size
    tmpl_quantiles = np.cumsum(tmpl_counts) / template.size

    interp_a_values = np.interp(src_quantiles, tmpl_quantiles, tmpl_values)

    # Here we compute values which the channel RGB value of full image will be modified to.
    interpb = []
    for i in range(0, 256):
        interpb.append(-1)

    # first compute which values in src image transform to and mark those values.

    for i in range(0, len(interp_a_values)):
        frm = src_values[i]
        to = interp_a_values[i]
        interpb[frm] = to

    # some of the pixel values might not be there in interp_a_values, interpolate those values using their
    # previous and next neighbours
    prev_value = -1
    prev_index = -1
    for i in range(0, 256):
        if interpb[i] == -1:
            next_index = -1
            next_value = -1
            for j in range(i + 1, 256):
                if interpb[j] >= 0:
                    next_value = interpb[j]
                    next_index = j
            if prev_index < 0:
                interpb[i] = (i + 1) * next_value / (next_index + 1)
            elif next_index < 0:
                interpb[i] = prev_value + ((255 - prev_value) * (i - prev_index) / (255 - prev_index))
            else:
                interpb[i] = prev_value + (i - prev_index) * (next_value - prev_value) / (next_index - prev_index)
        else:
            prev_value = interpb[i]
            prev_index = i

    # finally transform pixel values in full image using interpb interpolation values.
    wid = full.shape[1]
    hei = full.shape[0]
    ret2 = np.zeros((hei, wid))
    for i in range(0, hei):
        for j in range(0, wid):
            ret2[i][j] = interpb[full[i][j]]
    return ret2

def getDistance(corner, point):
    d = math.dist(corner, point)
    return d

def match_histograms_mod(inputCard1, inputCard2, inputCard3, inputCard4, referenceCard, fullImage, arucoCoords):
    """
        Return modified full image, by using histogram equalizatin on input and
         reference cards and applying that transformation on fullImage.
    """
    if inputCard1.ndim != referenceCard.ndim:
        raise ValueError('Image and reference must have the same number '
                         'of channels.')
    if inputCard2.ndim != referenceCard.ndim:
        raise ValueError('Image and reference must have the same number '
                         'of channels.') 
    if inputCard3.ndim != referenceCard.ndim:
        raise ValueError('Image and reference must have the same number '
                         'of channels.') 
    if inputCard4.ndim != referenceCard.ndim:
        raise ValueError('Image and reference must have the same number '
                         'of channels.')

 
    matched1 = np.empty(fullImage.shape, dtype=fullImage.dtype)
    matched2 = np.empty(fullImage.shape, dtype=fullImage.dtype)
    matched3 = np.empty(fullImage.shape, dtype=fullImage.dtype)
    matched4 = np.empty(fullImage.shape, dtype=fullImage.dtype)
    weightedMatched = fullImage


    for channel in range(inputCard1.shape[-1]):
        matched_channel1 = _match_cumulative_cdf_mod(inputCard1[..., channel], referenceCard[..., channel], fullImage[..., channel])
        matched1[..., channel] = matched_channel1
    
    for channel in range(inputCard2.shape[-1]):
        matched_channel2 = _match_cumulative_cdf_mod(inputCard2[..., channel], referenceCard[..., channel], fullImage[..., channel])
        matched2[..., channel] = matched_channel2

    for channel in range(inputCard3.shape[-1]):
        matched_channel3 = _match_cumulative_cdf_mod(inputCard3[..., channel], referenceCard[..., channel], fullImage[..., channel])
        matched3[..., channel] = matched_channel3

    for channel in range(inputCard4.shape[-1]):
        matched_channel4 = _match_cumulative_cdf_mod(inputCard4[..., channel], referenceCard[..., channel], fullImage[..., channel])
        matched4[..., channel] = matched_channel4

    corner1 = arucoCoords[0]
    corner2 = arucoCoords[1]
    corner3 = arucoCoords[2]
    corner4 = arucoCoords[3]

    x1 = min(int(corner1[0]), int(corner4[0]))
    x2 = max(int(corner2[0]), int(corner3[0]))
    y1 = min(int(corner1[1]), int(corner2[1]))
    y2 = max(int(corner3[1]), int(corner4[1]))

    corner1 = (x1, y1)
    corner2 = (x2, y1)
    corner3 = (x2, y2)
    corner4 = (x1, y2)


    for x in range(x1+2, x2-2):
        for y in range(y1+2, y2-2):
            d1 = getDistance(corner1, (x,y))
            d2 = getDistance(corner2, (x,y))
            d3 = getDistance(corner3, (x,y))
            d4 = getDistance(corner4, (x,y))

            w1 = (1/d1) / ((1/d1) + (1/d2) + (1/d3) + (1/d4))
            w2 = (1/d2) / ((1/d1) + (1/d2) + (1/d3) + (1/d4))
            w3 = (1/d3) / ((1/d1) + (1/d2) + (1/d3) + (1/d4))
            w4 = (1/d4) / ((1/d1) + (1/d2) + (1/d3) + (1/d4))

            r1,g1,b1 = matched1[y, x]
            r2,g2,b2 = matched2[y, x]
            r3,g3,b3 = matched3[y, x]
            r4,g4,b4 = matched4[y, x]

            r = int((w1*r1) + (w2*r2) + (w3*r3) + (w4*r4))
            g = int((w1*g1) + (w2*g2) + (w3*g3) + (w4*g4))
            b = int((w1*b1) + (w2*b2) + (w3*b3) + (w4*b4))
            rgb = (r,g,b)

            weightedMatched[y, x] = rgb



    return weightedMatched

def create_startup_window():
    global root
    root = tk.Tk()
    # global StdOutWindow
    # StdOutWindow = tk.Tk()
    # StdOutWindow.withdraw()
    root.title("Art Scanning App")
    root.geometry("1280x720")

    # Create a frame for the startup window
    frame = tk.Frame(root, bg="#f0f0f0", padx=20, pady=40)
    frame.pack(expand=True)


    # Add an image (replace 'path_to_your_image.png' with the actual path)
    image = tk.PhotoImage(file='C:/seniorDesign/git/sddec23-18/workingDirectory/assets/ArtscanLogoResize.png')  # Change this to your image path
    image_label = tk.Label(frame, image=image)
    image_label.pack()

    # Add a title label
    title_label = tk.Label(frame, text="Welcome to ArtScan: A Super-High Resolution Art Scanning Application", font=("Arial", 18), pady=10)
    title_label.pack()


    # Add a start button
    start_button = tk.Button(frame, text="Crop and Perspective Change", command=start_appCrop, font=("Arial", 14), padx=20, pady=8, bg="#4CAF50", fg="white")
    start_button.pack(pady=20)

    # Add a checkbox
    global percentErrorBool
    percentErrorBool = tk.IntVar()
    checkboxPercent = tk.Checkbutton(frame, text="Enter Pixel Crop and Image Output Size", variable=percentErrorBool, font=("Arial", 14), onvalue=1, offvalue=0)
    checkboxPercent.pack()

    start_button = tk.Button(frame, text="Color Correct", command=start_appColor, font=("Arial", 14), padx=20, pady=8, bg="#4CAF50", fg="white")
    start_button.pack(pady=20)

    # Add a checkbox
    global viewBool
    viewBool = tk.IntVar()
    checkbox = tk.Checkbutton(frame, text="Color Correct View", variable=viewBool, font=("Arial", 14), onvalue=1, offvalue=0)
    checkbox.pack()

    global StdOutColor
    StdOutColor = tk.Text(frame, wrap=tk.WORD)
    StdOutColor.pack_forget()
    # StdOutColor.pack(expand=True, fill=tk.BOTH)

    root.mainloop()

# Call the function to create the startup window
create_startup_window()
