import cv2 as cv
cap = cv.VideoCapture('testvid.mp4')
status = 1
count = 0
while status == 1:
    status, im = cap.read()
    cv.imwrite(f'{count}.jpg',im)
    count+=1