import cv2
import mediapipe as mp
import time


class Hand_Detector():
    def __init__(self, mode = False, maxHands = 2, detectionCon = 0.5, trackCon = 0.5):
        """
        the parameterized constructor of this class
        """
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.mpHands = mp.solutions.hands
        self.mpDraw = mp.solutions.drawing_utils
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.detectionCon,
                                self.trackCon)

    def find_Hands(self, img, draw = True):
        """
        to detect the hand(s) in the image and draw all the hand connections of the
        21 landmarks, if asked to, and also tell if the hand(s) detected is/are left or
        right
        """
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        handsType = []
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:  # iterating through each hand
                if draw:  # will draw the hand connections if draw equals True
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)

            for hand in self.results.multi_handedness:
                handType = hand.classification[0].label
                handsType.append(handType)

        return img, handsType

    def find_Position(self, img, handNo = 0, draw = True, landmark_To_Draw = 0,
                                                        draw_All_Landmarks = False):
        """
        to return the list containing the index number and the co-ordinates of all the
        21 landmarks of a particular hand, and draw a circle around either all landmarks
        or only that which is asked
        """
        lm_List = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark) :
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_List.append([id, cx, cy])
                if draw:
                    if draw_All_Landmarks :
                        cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)
                    else:
                        if id == landmark_To_Draw:
                            cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)
        return lm_List

    def return_Landmarks(self, img, lm_List, landmark = 0):
        """
        to return the list containing the index number and the co-ordinates of the given
        landmark of a particular hand"""
        return lm_List[landmark]


def main():
    """
    the main function
    """
    prevTime, curTime = 0, 0
    cap = cv2.VideoCapture(0)
    detector = Hand_Detector()
    while True:
        success, img = cap.read()
        img, handsType = detector.find_Hands(img)
        lm_List = detector.find_Position(img)  # list of 21 landmarks in every iteration
        if len(lm_List) != 0 :
            print(detector.return_Landmarks(img, lm_List))

        curTime = time.time()
        fps = 1 / (curTime - prevTime)
        prevTime = curTime
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                                                           (255, 0, 255), 3)
        cv2.imshow("Hand Tracking Video", img)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__": main()