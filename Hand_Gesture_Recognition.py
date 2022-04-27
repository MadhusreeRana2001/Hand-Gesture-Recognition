import cv2
import Hand_Tracking_Module as htm
import time
import os


wCam, hCam = 640, 480
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)


folder_Path = "Gesture Images"
my_List = os.listdir(folder_Path)
overlay_List = []  # since we're going to overlay each image on the video being captured
for img_Path in my_List:
    image = cv2.imread(f"{folder_Path}/{img_Path}")
    overlay_List.append(image)


def detect_Hand_Gesture(fingers, total_Fingers):
    """
    to detect the type of gesture being shown in the video being captured
    """

    conditions = {
        "HIGH-FIVE": [total_Fingers == 5],
        "OK": [total_Fingers == 3, fingers[0] == 0, fingers[1] == 0, fingers[2] == 1,
                                                    fingers[3] == 1, fingers[4] == 1],
        "PEACE": [total_Fingers == 2, fingers[0] == 0, fingers[1] == 1, fingers[2] == 1,
                                                     fingers[3] == 0, fingers[4] == 0],
        "ROCK": [total_Fingers == 2, fingers[0] == 0, fingers[1] == 1,
                                    fingers[2] == 0, fingers[3] == 0, fingers[4] == 1],
        "THUMBS-UP": [total_Fingers == 1, fingers[0] == 1, fingers[1] == 0, fingers[1] == 0,
                                                          fingers[1] == 0, fingers[1] == 0]
                }

    flag, count , gesture = 0, 0, ""
    for count, (gesture, condition) in enumerate(conditions.items()):
        if all(condition):
            flag = 1
            break
        else: continue
    if flag: return count, gesture
    else: return -1, ""


detector = htm.Hand_Detector(detectionCon=0.75)
prevTime = 0
tip_Ids = [8, 12, 16, 20]
while True:
    success, img = cap.read()
    img, handType = detector.find_Hands(img)
    lm_List = detector.find_Position(img, draw=False)
    total_Fingers = 0

    if len(lm_List) != 0:
        fingers = []
        # Thumb:
        if handType[0] == "Left":
            if lm_List[4][1] >= lm_List[3][1]: fingers.append(1)
            else: fingers.append(0)
        else:
            if lm_List[4][1] <= lm_List[3][1]: fingers.append(1)
            else: fingers.append(0)

        # For the other fingers:
        for id in tip_Ids:
            if lm_List[id][2] < lm_List[id - 2][2]: fingers.append(1)
            else: fingers.append(0)

        total_Fingers = fingers.count(1)
        count, gesture = detect_Hand_Gesture(fingers, total_Fingers)
        if count != -1:
            h, w, c = overlay_List[count].shape
            img[0: h, 0: w] = overlay_List[count]

            cv2.rectangle(img, (20, 225), (250, 300), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, gesture, (30, 278), cv2.FONT_HERSHEY_COMPLEX,
                                                                1, (255, 0, 0), 4)

    curTime = time.time()
    fps = 1 / (curTime - prevTime)
    prevTime = curTime
    cv2.putText(img, f"FPS: {str(int(fps))}", (520, 40), cv2.FONT_HERSHEY_COMPLEX, 0.6,
                (255, 0, 0), 2)

    cv2.imshow("Hand Gesture Recognition", img)
    if cv2.waitKey(1) & 0xFF == ord('q'): break