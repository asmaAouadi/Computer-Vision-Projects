#to learn how to show and detect Corners:

import cv2
import numpy
import glob

CHECKERBOARD = (9,6)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER , 30 , 0.001)

images = glob.glob('./images/*.png')

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)

    ret , corners = cv2.findChessboardCorners(gray , CHECKERBOARD , None)

    if ret:
        corners2 = cv2.cornerSubPix(gray , corners , (11,11) , (-1,-1) , criteria)

        cv2.drawChessboardCorners(img , CHECKERBOARD  , corners2 , ret)
        print(f"yes -> {fname} - corners shape : {corners2.shape}")
    else :
        print(f"No  -> {fname} - Not detected ")

    cv2.imshow('img' , img)
    cv2.waitKey(0)

cv2.destroyAllWindows()