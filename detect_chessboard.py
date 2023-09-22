import cv2
import numpy as np

def auto_canny(image, sigma=0.33):
    v = np.median(image)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    return edged

img = cv2.imread("chessboard4.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imshow("Grayscale Image", gray)
cv2.waitKey(0)

cv2.imwrite("grayscale.jpg", gray)

blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edges = auto_canny(blurred)

contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key = cv2.contourArea, reverse = True)[:10]

board_contour = None
for contour in contours:
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.01 * perimeter, True)
    
    if len(approx) > 4:
        rect = cv2.minAreaRect(approx)
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        board_contour = np.array([box], dtype = "float32")

    else:
        board_contour = approx

    if board_contour is not None and len(board_contour) > 0:
        break

if board_contour is None:
    print("Chessboard not found in image.")
else:
    if len(board_contour) > 0:
        print(board_contour)

        board_contour = board_contour.reshape(-1, 1, 2).astype(np.int32)  # Reshape and convert to integer

        # Draw contour
        cv2.drawContours(img, [board_contour], -1, (0, 255, 0), 2)

        # Warp perspective to get top-down view of chessboard
        pts1 = np.float32(board_contour)
        # pts2 = np.float32([[0, 0], [0, 480], [480, 480], [480, 0]])  # assuming a square chessboard of 480x480 pixels
        pts2 = np.float32([[0, 0], [480, 0], [480, 480], [0, 480]])  # ordered: top-left, top-right, bottom-right, bottom-left

        s = np.sum(pts1, axis=2)
        diff = np.diff(pts1, axis=2)
        
        top_left = pts1[np.argmin(s)]
        top_right = pts1[np.argmax(diff)]
        bottom_right = pts1[np.argmax(s)]
        bottom_left = pts1[np.argmin(diff)]
        
        pts1_ordered = np.array([top_left, top_right, bottom_right, bottom_left], dtype = "float32")

        print("pts1 :");
        print(pts1);
        print("pts2 :");
        print(pts2);
        # matrix = cv2.getPerspectiveTransform(pts1, pts2)
        matrix = cv2.getPerspectiveTransform(pts1_ordered, pts2)
        result = cv2.warpPerspective(img, matrix, (480, 480))

        cv2.circle(img, tuple(map(int, top_left[0])), 10, (255, 0, 0), -1) # Blue
        cv2.circle(img, tuple(map(int, top_right[0])), 10, (0, 255, 0), -1) # Green
        cv2.circle(img, tuple(map(int, bottom_right[0])), 10, (0, 0, 255), -1) # Red
        cv2.circle(img, tuple(map(int, bottom_left[0])), 10, (255, 255, 0), -1) # Cyan


        cv2.imshow("Chessboard", result)
        cv2.waitKey(0)
