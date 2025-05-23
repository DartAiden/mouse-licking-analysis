import cv2 as cv
cap = cv.VideoCapture('rects_2.mp4')
status = 1
count = 27000
while status == 1:
    status, im = cap.read()
    cv.imwrite(f'{count}.jpg',im)
    count+=1