import cv2 as cv
vids = [
r'inputvids\T51_250429_lick_Video.mp4',
r'inputvids\T58_250429_lick_Video.mp4',
r'inputvids\T64_250429_lick_Video_CROP.mp4'
r'inputvids\T65_250429_lick_Video.mp4'
]

manuallicks = open('manual_lick_times.csv','w')
manualstims = open('manual_sitm_times.csv','w')

for i in vids:
    cap = cv.VideoCapture(i)
    stimtimes = []
    licktimes = []
    ret = True
    while ret:
        ret, img = cap.read()
        if not ret or img is None:
            break
        ts = (cap.get(cv.CAP_PROP_POS_MSEC))
        print(ts)
        cv.imshow('lick',img)
        if ord('s') == cv.waitKey(1):
            break