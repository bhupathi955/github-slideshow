#!/usr/bin/env python

import cv2
import os
from time import sleep

dest = "D:/MINI/dest"
camera = cv2.VideoCapture(1)

#for i in range(3):
#   return_value, image = camera.read()
#    sleep(1)
#    cv2.imwrite(os.path.join(dest,'img'+str(i)+'.jpg'), image)

return_value, image = camera.read()
cv2.imwrite(os.path.join(dest,'img.jpg'), image)

camera.release()
del(camera)