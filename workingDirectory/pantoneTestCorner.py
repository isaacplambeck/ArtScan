import cv2
import cv2.aruco as aruco

def pantoneTest():
    pantone = cv2.imread("C:/seniorDesign/git/sddec23-18/test/assets/doomPantone.jpg")
    scale_percent1 = 200  # percent of the original size
    width1 = int(pantone.shape[1] * scale_percent1 / 100)
    height1 = int(pantone.shape[0] * scale_percent1 / 100)
    
    if height1 > 950 or width1 > 950:
        while True:
            if height1 > 950 or width1 > 950:
                scale_percent1 -= 5
                width1 = int(pantone.shape[1] * scale_percent1 / 100)
                height1 = int(pantone.shape[0] * scale_percent1 / 100)
            else:
                break

    pantoneResized = cv2.resize(pantone, (width1, height1))
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)

    # Detect ArUco markers
    corners, ids, _ = aruco.ArucoDetector.detectMarkers(pantoneResized, aruco_dict)

    if ids is not None:
        aruco.drawDetectedMarkers(pantoneResized, corners, ids)

    return pantoneResized

# Call the function
resulting_image = pantoneTest()

# Display the resulting image
cv2.imshow("Pantone Test", resulting_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
