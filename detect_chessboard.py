import cv2
import numpy as np

# This script does the following:
#
#   Reads an image file named 'chessboard.jpg'. Replace 'chessboard.jpg' with the name of your image file.
#
#   Converts the image to grayscale.
#
#   Tries to find the corners of a chessboard in the image. (8,8) specifies that it's an 8x8 chessboard. 
#   Change these values if you have a different sized chessboard.
#
#   If the chessboard is found (ret is True), it draws the corners on the image.
#
#   It displays the image and waits until you close the image window.
#
# Please note that this script is very basic and might not work perfectly with all images. 
# The chessboard needs to be clearly visible and the image needs to be of a certain quality. 
# The image should also be taken from a straight angle. This script doesn't handle piece detection, 
# which is a more complex task. It's a first step to get you started on your project.

# Load image
image = cv2.imread('chessboard.jpg')

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Find the chess board corners
ret, corners = cv2.findChessboardCorners(gray, (8,8), None)

# If found, add object points, image points (after refining them)
if ret == True:
    # Draw and display the corners
    cv2.drawChessboardCorners(image, (8,8), corners, ret)

    cv2.imshow('Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
