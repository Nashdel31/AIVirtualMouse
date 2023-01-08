import cv2
import numpy as np
import time
import CompVision_HandTracking_Module as htm
import autopy

wCam, hCam = 640, 480
frameR = 100
smoothening = 5

pTime = 0
pLocX, pLocY = 0, 0
cLocX, cLocY = 0, 0

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
detector = htm.handDetector(maxHands=1)
wScr, hScr = autopy.screen.size()

while True:
    # 1. Find hand Landmarks
    success, img = cap.read()
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img)

    # 2. Get the tip of the index and middle finger
    if len(lmList) != 0:
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]

        # 3. check which fingers are up
        fingers = detector.fingersUp()
        # cadre du mousePad
        cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR), (255, 0, 255), 2)

        # 4. only index finger : moving mode
        if fingers[1] == 1 and fingers[2] == 0:
            # 5. convert coordinates
            x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
            y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScr))

            # 6. Smoothen values
            cLocX = pLocX + (x3 - pLocX) / smoothening
            cLocY = pLocY + (y3 - pLocY) / smoothening

            # 7. move mouse
            autopy.mouse.move(wScr - cLocX, cLocY)
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            pLocX, pLocY = cLocX, cLocY

        # 8. both index and middle finger are up : clicking mode
        if fingers[1] == 1 and fingers[2] == 1:
            length, img, lineInfo = detector.findDistance(8, 12, img)

            # 9. find distance between fingers
            if length < 40:
                cv2.circle(img, (lineInfo[4], lineInfo[5]), 15, (0, 255, 0), cv2.FILLED)
                # 10. click mouse if distance short
                autopy.mouse.click()
                # autopy.mouse.toggle(True, button=LEFT_BUTTON)
            # else:
                # autopy.mouse.toggle(Button="LEFT", down=False)

    # 11. Frame rate
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    # 12. display
    cv2.imshow("image", img)

    # 13. closing display
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
