import cv2
import numpy as np
import time
import CompVision_HandTracking_Module as htm
import autopy
from autopy import mouse

wCam, hCam = 1280, 720
xm1, ym1, xm2, ym2 = 640, 120, 1140, 480  # cadre mousePad
frameR = 100
smoothening = 5

pTime = 0
pLocX, pLocY = 0, 0
cLocX, cLocY = 0, 0

cap = cv2.VideoCapture(0)
# cap.set(3, wCam)
# cap.set(4, hCam)
detector = htm.handDetector(maxHands=1, detectionCon=0.8)
wScr, hScr = autopy.screen.size()

while True:
    # 1. Find hand Landmarks
    success, img = cap.read()
    img = cv2.flip(img, 1)
    img = detector.findHands(img, draw=False)
    lmList, bbox = detector.findPosition(img, draw=False)

    # 2. Get the tips of the index finger
    if len(lmList) != 0:
        x1, y1 = lmList[5][1:]  # Base index
        x2, y2 = lmList[8][1:]  # top index

        # 3. check which fingers are up
        fingers = detector.fingersUp()
        # cadre du mousePad
        # cv2.rectangle(img, (xm1, ym1), (xm2, ym2), (255, 0, 255), 2)

        # 4. little finger down : mouse active mode
        if fingers[4] == 0:
            # 5. convert coordinates
            x3 = np.interp(x1, (xm1, xm2), (0, wScr))
            y3 = np.interp(y1, (ym1, ym2), (0, hScr))

            # 6. Smoothen values
            cLocX = pLocX + (x3 - pLocX) / smoothening
            cLocY = pLocY + (y3 - pLocY) / smoothening

            # 7. move mouse
            autopy.mouse.move(cLocX, cLocY)
            # cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            pLocX, pLocY = cLocX, cLocY

            # 9. index down : click
            if fingers[1] == 0:
                # cv2.circle(img, (x1, y1), 15, (0, 255, 0), cv2.FILLED)
                autopy.mouse.click()

    # 11. Frame rate
    # cTime = time.time()
    # fps = 1 / (cTime - pTime)
    # pTime = cTime
    # cv2.putText(img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    # 12. display
    cv2.imshow("image", img)

    # 13. closing display
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
