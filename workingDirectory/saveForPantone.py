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
    #aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
    parameters =  cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)

    # Detect ArUco markers
    #corners, ids, _ = aruco.detectMarkers(pantoneResized, aruco_dict)
    markerCorners, markerIds, rejectedCandidates = detector.detectMarkers(pantoneResized)
    
    
    for i in range(len(markerIds)):
        marker_id = markerIds[i][0]  # Get the ID of the current marker
        
        # Check if you want to change the corner for a specific marker ID
        if marker_id == 1:
            # Swap the top-left and top-right corners
            markerCorners[i][0][0], markerCorners[i][0][2] = markerCorners[i][0][3], markerCorners[i][0][0]

    for i in range(len(markerIds)):
        marker_id = markerIds[i][0]  # Get the ID of the current marker
        
        # Check if you want to change the corner for a specific marker ID
        if marker_id == 2:
            # Swap the top-left and top-right corners
            markerCorners[i][0][0], markerCorners[i][0][2] = markerCorners[i][0][1], markerCorners[i][0][0]

    for i in range(len(markerIds)):
        marker_id = markerIds[i][0]  # Get the ID of the current marker
        
        # Check if you want to change the corner for a specific marker ID
        if marker_id == 0:
            # Swap the top-left and top-right corners
            markerCorners[i][0][0], markerCorners[i][0][2] = markerCorners[i][0][2], markerCorners[i][0][0]

    aruco.drawDetectedMarkers(pantoneResized, markerCorners, markerIds)


    if len(markerCorners) > 0:
        marker_index = 0  # Change this to the index of the marker you're interested in
        corners = markerCorners[marker_index][0]

        # Extract the coordinates of the individual corners
        top_left_corner = corners[0]
        top_right_corner = corners[1]
        bottom_right_corner = corners[2]
        bottom_left_corner = corners[3]

        # Print or use the coordinates as needed
        #print("Top Left Corner:", top_left_corner)
        #print("Top Right Corner:", top_right_corner)
        #print("Bottom Right Corner:", bottom_right_corner)
        #print("Bottom Left Corner:", bottom_left_corner)

        corners = np.int32(corners)
        line_color = (0, 255, 0)
      
        
        for i in range(len(markerCorners)):
            for j in range(i + 1, len(markerCorners)):
                # Calculate the centers of each marker
                if i == 0 and j == 1 or i == 0 and j == 2 or i == 1 and j == 3 or i == 2 and j == 3:
                    center1 = markerCorners[i][0]
                    center2 = markerCorners[j][0]
                    center1 = np.int32(center1)
                    center2 = np.int32(center2)
                
                # Draw a line between the centers of the markers
                    cv2.line(pantoneResized, tuple(center1[0]), tuple(center2[0]), line_color, 2)


            # Calculate angle and length of the line
            angle = np.arctan2(center2[1] - center1[1], center2[0] - center1[0])
            line_length = np.sqrt((center2[1] - center1[1]) ** 2 + (center2[0] - center1[0]) ** 2)

            # Create a rotation matrix to align the line horizontally
            rotation_matrix = cv2.getRotationMatrix2D(center=center1, angle=np.degrees(-angle), scale=1)

            # Rotate the image
            rotated_image = cv2.warpAffine(pantoneResized, rotation_matrix, (pantoneResized.shape[1], pantoneResized.shape[0]))

            # Crop the image based on the rotated line's length and width
            cropped_image = rotated_image[int(center1[1] - line_length / 2):int(center1[1] + line_length / 2),
                                          int(center1[0] - line_length / 2):int(center1[0] + line_length / 2)]

        #Top Left Corner: [ 92. 729.]
        #Top Right Corner: [ 92. 729.]
        #Bottom Right Corner: [ 92. 729.]
        #Bottom Left Corner: [ 39. 785.]


    return cropped_image