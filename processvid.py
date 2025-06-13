import cv2 as cv
cap = cv.VideoCapture('20hz_crop\L09_20Hz_252905_CROP.mp4')
status = 1
count = 33000
while status == 1:
    status, im = cap.read()
    cv.imwrite(f'{count}.jpg',im)
    count+=1