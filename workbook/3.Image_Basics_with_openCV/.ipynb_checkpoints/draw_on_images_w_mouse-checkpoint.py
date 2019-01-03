import cv2
import numpy as np

###############
## FUNCTION ###
###############

def draw_circle(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(img,(x,y),100,(0,255,0),-1)
    elif event == cv2.EVENT_RBUTTONDOWN:
        cv2.circle(img,(x,y),100,(0,0,255),-1)

cv2.namedWindow(winname='Blank')

cv2.setMouseCallback('Blank',draw_circle)

################################
## SHOWING IMAGE WITH OPENCV ###
################################

img = np.zeros((512,512,3))
while True:
    cv2.imshow('Blank',img)
    if cv2.waitKey(20) & 0xFF == 27:
        break
cv2.destroyAllWindows()