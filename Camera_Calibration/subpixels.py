# subpixels.py — CLEAN VERSION
# WHAT IT DOES: finds strong corners using Harris detector,
# then refines each one to sub-pixel accuracy.
#
# THEORY:
#   Harris score R = det(M) - k*trace(M)²
#   where M = structure tensor built from image gradients Ix, Iy.
#   High R = corner, negative R = edge, |R| small = flat.
#   cornerSubPix() then refines by finding the exact zero of
#   the gradient dot product in a local window.
#   Red  = raw Harris centroid (integer pixel)
#   Green = refined sub-pixel position

import cv2
import numpy as np

img  = cv2.imread('./images/frame0001.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# cornerHarris(src, blockSize, ksize, k)
#   blockSize = 2 : neighbourhood size for structure tensor M
#   ksize     = 3 : Sobel aperture (gradient filter size)
#   k         = 0.04 : Harris free parameter (typically 0.04-0.06)
gray_f = np.float32(gray)
dst = cv2.cornerHarris(gray_f, blockSize=2, ksize=3, k=0.04)

# Dilate just to make corners visible (not part of detection)
dst = cv2.dilate(dst, None)

# Threshold: keep only strong corners (top 1% of response)
ret, dst = cv2.threshold(dst, 0.01 * dst.max(), 255, 0)
dst = np.uint8(dst)

# connectedComponentsWithStats finds individual corner blobs
# centroids = float center of each blob = Harris corner location
ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)

# Refine centroids to sub-pixel using cornerSubPix
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
corners = cv2.cornerSubPix(gray_f, np.float32(centroids), (5, 5), (-1, -1), criteria)
# (5,5) = half-window size for refinement search

print(f"Harris corners found: {len(centroids)}")
print(f"Sample raw centroid : {centroids[1]}")
print(f"Sample refined      : {corners[1]}  ← sub-pixel precision")

# Draw: red=Harris raw, green=refined
centroids_int = np.intp(centroids)
corners_int   = np.intp(corners)
img[centroids_int[:, 1], centroids_int[:, 0]] = [0, 0, 255]   # red
img[corners_int[:, 1],   corners_int[:, 0]]   = [0, 255, 0]   # green

cv2.imwrite('./subpixels.png', img)
cv2.imshow('Harris SubPixels — red=raw  green=refined', img)
cv2.waitKey(0)
cv2.destroyAllWindows()