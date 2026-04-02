import sys
import cv2
import numpy
import matplotlib.pyplot as plt
import glob
import pathlib as path




# 1- we get the image

# 2- we calibrate the camera 

out = path("stereo_output")
out.mkdir(exist_ok=True)





cv2.calibrateCamera()