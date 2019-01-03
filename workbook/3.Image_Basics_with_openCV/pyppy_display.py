import cv2
img = cv2.imread('../DATA/00-puppy.jpg')
while True:
    cv2.imshow('Puppy',img)
    # IF we ve waited atleast 1ms and weve pressed the ESC key
    if cv2.waitKey(1) & 0xFF == 27:
        break
cv2.destroyAllWindows()