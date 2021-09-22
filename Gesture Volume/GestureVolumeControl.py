
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import cv2
import time
import numpy as np
import HandTrackMod as htm
import math

################################
w_cam, h_cam = 640, 480
prev_time = 0
curr_time = 0
################################

cap = cv2.VideoCapture(0)
cap.set(3,w_cam)
cap.set(4,h_cam)

detector = htm.HandDetection(det_confidence = 0.7)

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
# volume.GetMute()
# volume.GetMasterVolumeLevel()
vol_range = volume.GetVolumeRange()
minv = vol_range[0]
maxv = vol_range[1]

while True:
    success, img = cap.read()
    img = detector.detectHands(img)
    lm_list = detector.findPosition(img)

    if lm_list:
        x1, y1, x2, y2 = lm_list[4][1], lm_list[4][2], lm_list[8][1], lm_list[8][2]
        cx, cy = (x1 + x2)//2, (y1 + y2)//2
        
        cv2.circle(img, (x1, y1), 12, (150, 0, 100), cv2.FILLED)
        cv2.circle(img, (x2, y2), 12, (150, 0, 100), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (150, 0, 100), 2)
        cv2.circle(img, (cx, cy), 12, (150, 0, 100), cv2.FILLED)
        
        length = math.hypot(x2-x1, y2-y1)
        
        vol = np.interp(length, [50, 280],[minv, maxv])
        print(vol)
        volume.SetMasterVolumeLevel(vol, None)

        if length <= 50:
            cv2.circle(img, (cx, cy), 8, (150, 0, 0), cv2.FILLED)

    curr_time = time.time()
    fps = 1/(curr_time-prev_time)
    prev_time = curr_time

    cv2.putText(img, f'FPS: {int(fps)}', (10, 25), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0), 1)
    cv2.imshow('Image', img)
    cv2.waitKey(1)