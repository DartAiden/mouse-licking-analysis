import random
import os
import cv2 as cv
import shutil
vids =[
r'annotated_vids\T65_250429_lick_Video_annotated.mp4',
r'annotated_vids\T64_250429_lick_Video_CROP_annotated.mp4',
r'annotated_vids\T58_250429_lick_Video_annotated.mp4',
r'annotated_vids\T55_250429_lick_Video_annotated.mp4'
]
os.mkdir('all_frames')
count = 0
for a in vids:
    cap = cv.VideoCapture(a)
    ret = True
    while ret:
        ret, img = cap.read()
        if not ret or img is None:
            break
        title = f'{count}.jpg'
        full_path = os.path.join('all_frames',title)
        cv.imwrite(full_path, img)
        count+=1

files = os.listdir('all_frames')
random.shuffle(files)
for select in files[:1000]:
    full_path = os.path.join('all_frames', select)
    shutil.move(full_path, 'select_frames')