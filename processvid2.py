import cv2 as cv
cap = cv.VideoCapture('testvid4.mp4')
status = 1
count = 15000
vids = [
'T30_250429_lick_Video.mp4',
'T35_250429_lick_Video.mp4',
'T36_250429_lick_Video.mp4',
'T55_250429_lick_Video.mp4',
'T58_250429_lick_Video.mp4',
'T64_250429_lick_Video.mp4'
]

times = {
'T30_250429_lick_Video.mp4':125,
'T35_250429_lick_Video.mp4':240,
'T36_250429_lick_Video.mp4':180,
'T55_250429_lick_Video.mp4':94,
'T58_250429_lick_Video.mp4':110,
'T64_250429_lick_Video.mp4':0

}

for a in vids:
    cap = cv.VideoCapture(a)
    ts = cap.get(cv.CAP_PROP_POS_MSEC) * 1000

    while(ts < times[a]):
        ts = cap.get(cv.CAP_PROP_POS_MSEC) * 1000
    while ts < times[a] + 5:
        ts = cap.get(cv.CAP_PROP_POS_MSEC)
        status, im = cap.read()
        cv.imwrite(f'{count}.jpg',im)
        count+=1