import cv2
import numpy as np

################
## VARIABLES ###
################

# true while mouse button down false while up
drawing = False
# starting point of rect temp vals
ix,iy = -1,-1

###############
## FUNCTION ###
###############

def draw_rectangle(event,x,y,flags,param):
    global ix,iy,drawing
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing=True
        ix,iy = x,y
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            cv2.rectangle(img,(ix,iy),(x,y),(0,255,0),-1)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing=False
        cv2.rectangle(img,(ix,iy),(x,y),(0,255,0),-1)

################################
## SHOWING IMAGE WITH OPENCV ###
################################

img = np.zeros((512,512,3))
cv2.namedWindow(winname='Blank3')
cv2.setMouseCallback('Blank3',draw_rectangle)

while True:
    cv2.imshow('Blank3',img)
    if cv2.waitKey(20) & 0xFF == 27:
        break
cv2.destroyAllWindows()