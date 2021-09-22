import cv2
import mediapipe as mp
import time

class HandDetection():
    def __init__(self, stat_mode=False, max_hands=2, det_confidence=0.5, track_confidence=0.5):
        self.stat_mode = stat_mode
        self.max_hands = max_hands
        self.det_confidence = det_confidence
        self.track_confidence = track_confidence
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(self.stat_mode, self.max_hands, self.det_confidence, self.track_confidence)
        self.mp_draw = mp.solutions.drawing_utils
    
    def detectHands(self, img, draw=True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.proc_val = self.hands.process(img_rgb)
        
        if self.proc_val.multi_hand_landmarks:
            for hand_lms in self.proc_val.multi_hand_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(img, hand_lms, self.mp_hands.HAND_CONNECTIONS)

        return img
    
    def findPosition(self, img, hand_no=0):
        lm_list = list()

        if self.proc_val.multi_hand_landmarks:
            det_hand = self.proc_val.multi_hand_landmarks[hand_no]
            for idx, lm in enumerate(det_hand.landmark):
                h, w, c = img.shape
                px, py = int(lm.x*w), int(lm.y*h)
                lm_list.append([idx, px, py])
        
        return lm_list



def main():
    cap = cv2.VideoCapture(0)
    prev_time = 0
    curr_time = 0

    detector = HandDetection()

    while True:
        success, img = cap.read()
        img = detector.detectHands(img)
        lm_list = detector.findPosition(img)
        if len(lm_list) != 0:
            print(lm_list[0])

        curr_time = time.time()
        fps = 1/(curr_time-prev_time)
        prev_time = curr_time

        cv2.putText(img, f'FPS: {int(fps)}', (10, 60), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 2)
        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()