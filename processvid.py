import cv2 as cv
cap = cv.VideoCapture('testvid4.mp4')
status = 1
count = 9000
while status == 1:
    status, im = cap.read()
    print(type(im))