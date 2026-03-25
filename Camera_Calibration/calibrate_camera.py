# this is a full calibration

import cv2
import numpy as np
import glob
import os

CHECKERBOARD = (9,6)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER , 30 , 0.001)

objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1] , 3) , np.float32)
objp[: , :2] = np.mgrid[0:CHECKERBOARD[0] , 0:CHECKERBOARD[1]].T.reshape(-1,2)

objpoints = []
imgpoints = []


images = glob.glob('./images/*.png')
gray =None

for fname in sorted(images):
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)

    ret , corners = cv2.findChessboardCorners(gray , CHECKERBOARD , None)

    if ret :
        corners2 = cv2.cornerSubPix(gray , corners , (11,11) , (-1 , -1) , criteria)
        objpoints.append(objp)
        imgpoints.append(corners2)
        print(f"yes -> {fname}")
    else:
        print(f"NO -> {fname}")

ret , mtx , dist , rvecs , tvecs = cv2.calibrateCamera(
    objpoints , imgpoints , gray.shape[::-1] , None , None
)


print(f"\nret   (reprojection error) : {ret:.4f}   ← must be < 1.0")
print(f"\nmtx   (camera matrix K):\n{mtx}")
print(f"\ndist  (distortion [k1,k2,p1,p2,k3]):\n{dist}")

# Per-image reprojection error — tells you which images are bad
print("\nPer-image error:")
for i in range(len(objpoints)):
    proj, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    err = cv2.norm(imgpoints[i], proj, cv2.NORM_L2) / len(proj)
    print(f"  image {i:02d}: {err:.4f} px")

# Save — same format as the teacher's .npy files
os.makedirs('./camera_params', exist_ok=True)
np.save('./camera_params/ret',   ret)
np.save('./camera_params/mtx',   mtx)
np.save('./camera_params/dist',  dist)
np.save('./camera_params/rvecs', np.array(rvecs, dtype=object))
np.save('./camera_params/tvecs', np.array(tvecs, dtype=object))
print("\n[SAVED] camera_params/ updated")