from imutils.perspective import four_point_transform
from skimage import exposure
import numpy as np
import argparse
import imutils
import cv2
import sys
import math
from os.path import exists
import os.path as pathfile
from PIL import Image


# python Test_Pantone2.py --reference ref.jpg --input input.jpg --output out.jpg





def find_color_card(image):
    # load the ArUCo dictionary, grab the ArUCo parameters, and
    # detect the markers in the input image
    arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_ARUCO_ORIGINAL)
    arucoParams = cv2.aruco.DetectorParameters_create()

    # Edit Parameters for aruco detection
    arucoParams.adaptiveThreshConstant = 6
    arucoParams.adaptiveThreshWinSizeMax = 50
    #arucoParams.adaptiveThreshWinSizeMin = 2
    arucoParams.adaptiveThreshWinSizeStep = 6
    arucoParams.aprilTagCriticalRad = 0
    #arucoParams.aprilTagMaxNmaxima = 15
    #arucoParams.aprilTagMinClusterPixels = 3
    #arucoParams.aprilTagMinWhiteBlackDiff = 3
    arucoParams.cornerRefinementMinAccuracy = 0.05
    arucoParams.cornerRefinementWinSize = 7
    #arucoParams.errorCorrectionRate = 0.4
    arucoParams.maxMarkerPerimeterRate = 10
    arucoParams.minMarkerPerimeterRate = 0.01
    arucoParams.minOtsuStdDev = 3
    arucoParams.minDistanceToBorder = 1
    #arucoParams.useAruco3Detection = False
    arucoParams.minMarkerDistanceRate = 0.01
    #arucoParams.perspectiveRemoveIgnoredMarginPerCell = 1
    #arucoParams.perspectiveRemovePixelPerCell = 7
    arucoParams.polygonalApproxAccuracyRate = .02
    



    (corners, ids, rejected) = cv2.aruco.detectMarkers(image,
                                                       arucoDict, parameters=arucoParams)

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

def match_histograms_mod(inputCard1, inputCard2, inputCard3, inputCard4, referenceCard, fullImage):
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

    (h, w) = fullImage.shape[:2]
    corner1 = (0,0)
    corner2 = (w, 0)
    corner3 = (w, h)
    corner4 = (0, h)

    for x in range(1, h-1):
        for y in range(1, w-1):
            d1 = getDistance(corner1, (x,y))
            d2 = getDistance(corner2, (x,y))
            d3 = getDistance(corner3, (x,y))
            d4 = getDistance(corner4, (x,y))

            w1 = (1/d1) / ((1/d1) + (1/d2) + (1/d3) + (1/d4))
            w2 = (1/d2) / ((1/d1) + (1/d2) + (1/d3) + (1/d4))
            w3 = (1/d3) / ((1/d1) + (1/d2) + (1/d3) + (1/d4))
            w4 = (1/d4) / ((1/d1) + (1/d2) + (1/d3) + (1/d4))

            r1,g1,b1 = matched1[x, y]
            r2,g2,b2 = matched2[x, y]
            r3,g3,b3 = matched3[x, y]
            r4,g4,b4 = matched4[x, y]

            r = int((w1*r1) + (w2*r2) + (w3*r3) + (w4*r4))
            g = int((w1*g1) + (w2*g2) + (w3*g3) + (w4*g4))
            b = int((w1*b1) + (w2*b2) + (w3*b3) + (w4*b4))
            rgb = (r,g,b)

            weightedMatched[x, y] = rgb



    return weightedMatched


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-r", "--reference", required=True,
                help="path to the input reference image")
ap.add_argument("-v", "--view", required=False, default=False, action='store_true',
                help="Image Preview?")
ap.add_argument("-o", "--output", required=False, default=False,
                help="Image Output Path")
ap.add_argument("-i", "--input", required=True,
                help="path to the input image to apply color correction to")
args = vars(ap.parse_args())

# load the reference image and input images from disk
print("[INFO] loading images...")
# raw = cv2.imread(args["reference"])
# img1 = cv2.imread(args["input"])
file_exists = pathfile.isfile(args["reference"])
print(file_exists)

if not file_exists:
    print('[WARNING] Referenz File not exisits '+str(args["reference"]))
    sys.exit()


raw = cv2.imread(args["reference"])
img1 = cv2.imread(args["input"])

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
if args['view']:
    cv2.imshow("Reference", raw)
    cv2.imshow("Input", img1)

# find the color matching card in each image
print("[INFO] finding color matching cards...")
print("Attempting to find card in Reference")
rawCard = find_color_card(raw)
if rawCard is not None:
    print("Found Reference.")

print("Attempting to find card in Image1")
imageCard1 = find_color_card(img1_L)
if imageCard1 is not None:
    print("Found Image1")

print("Attempting to find card in Image2")
imageCard2 = find_color_card(img2_L)
if imageCard2 is not None:
    print("Found Image2")

print("Attempting to find card in Image3")
imageCard3 = find_color_card(img3_L)
if imageCard3 is not None:
    print("Found Image3")

print("Attempting to find card in Image4")
imageCard4 = find_color_card(img4_L)
if imageCard4 is not None:
    print("Found Image4")


# if the color matching card is not found in either the reference
# image or the input image, gracefully exit
if rawCard is None or imageCard1 is None or imageCard2 is None or imageCard3 is None or imageCard4 is None:
    print("[INFO] could not find color matching card in all images")
    sys.exit(0)

# show the color matching card in the reference image and input image,
# respectively
if args['view']:
    cv2.imshow("Reference Color Card", rawCard)
    cv2.imshow("Input Color Card 1", imageCard1)
    cv2.imshow("Input Color Card 2", imageCard2)
    cv2.imshow("Input Color Card 3", imageCard3)
    cv2.imshow("Input Color Card 4", imageCard4)

# apply histogram matching from the color matching card in the
# reference image to the color matching card in the input image
print("[INFO] matching images...")

# imageCard2 = exposure.match_histograms(img1, ref,
# inputCard = exposure.match_histograms(inputCard, referenceCard, multichannel=True)
result2 = match_histograms_mod(imageCard1, imageCard2, imageCard3, imageCard4, rawCard, img1)
 
# show our input color matching card after histogram matching
cv2.imshow("Input Color Card After Matching", result2)


if args['view']:
    cv2.imshow("result2", result2)

if args['output']:
    file_ok = exists(args['output'].lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')))

    if file_ok:
        cv2.imwrite(args['output'], result2)
        print("[SUCCESSUL] Your Image was written to: "+args['output']+"")
    else:
        print("[WARNING] Sorry, But this is no valid Image Name "+args['output']+"\nPlease Change Parameter!")

if args['view']:
    cv2.waitKey(0)

if not args['view']:
    if not args['output']:
        print('[EMPTY] You Need at least one Paramter "--view" or "--output".')