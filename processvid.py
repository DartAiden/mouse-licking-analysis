import cv2 as cv
cap = cv.VideoCapture('mouse_crop_video.mp4')
status = 1
count = 24000
while status == 1:
    status, im = cap.read()
    cv.imwrite(f'{count}.jpg',im)
    count+=1