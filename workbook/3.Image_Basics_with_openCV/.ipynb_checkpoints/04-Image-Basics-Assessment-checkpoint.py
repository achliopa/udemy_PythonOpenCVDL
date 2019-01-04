import cv2
import numpy as np

###############
## FUNCTION ###
###############

def draw_circle(event,x,y,flags,param):
    if event == cv2.EVENT_RBUTTONDOWN:
        cv2.circle(img,(x,y),100,(0,0,255),10)

cv2.namedWindow(winname='PuppyBag')

cv2.setMouseCallback('PuppyBag',draw_circle)

################################
## SHOWING IMAGE WITH OPENCV ###
################################

img  = cv2.imread('../DATA/dog_backpack.jpg')
while True:
    cv2.imshow('PuppyBag',img)
    if cv2.waitKey(20) & 0xFF == 27:
        break
cv2.destroyAllWindows()