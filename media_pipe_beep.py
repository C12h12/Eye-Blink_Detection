import cv2
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector
from cvzone.PlotModule import LivePlot
import winsound  # for beep sound

# --------- Webcam ---------
cap = cv2.VideoCapture(0)

detector = FaceMeshDetector(maxFaces=1)
plotY = LivePlot(640, 360, [20, 50], invert=True)

idList = [22, 23, 24, 26, 110, 157, 158, 159, 160, 161, 130, 243]
ratioList = []
blinkCounter = 0
counter = 0
color = (255, 0, 255)

# --- Drowsiness variables ---
drowsy_counter = 0     # counts continuous closed-eye frames
DROWSY_THRESH = 40     # ~ 40 frames ≈ 1.3 sec if 30 FPS
alert_on = False

while True:
    success, img = cap.read()
    if not success:
        print("⚠️ Failed to access webcam")
        break

    img, faces = detector.findFaceMesh(img, draw=False)

    if faces:
        face = faces[0]
        leftUp = face[159]
        leftDown = face[23]
        leftLeft = face[130]
        leftRight = face[243]

        lengthVer, _ = detector.findDistance(leftUp, leftDown)
        lengthHor, _ = detector.findDistance(leftLeft, leftRight)

        ratio = int((lengthVer / lengthHor) * 100)
        ratioList.append(ratio)
        if len(ratioList) > 3:
            ratioList.pop(0)
        ratioAvg = sum(ratioList) / len(ratioList)

        # Blink detection
        if ratioAvg < 35 and counter == 0:
            blinkCounter += 1
            color = (0, 200, 0)
            counter = 1
        if counter != 0:
            counter += 1
            if counter > 10:
                counter = 0
                color = (255, 0, 255)

        # --- Drowsiness detection ---
        if ratioAvg < 35:  # eyes closed
            drowsy_counter += 1
        else:
            drowsy_counter = 0
            alert_on = False

        if drowsy_counter > DROWSY_THRESH:
            alert_on = True
            # 🔔 Beep sound
            winsound.Beep(2500, 1000)  # freq=2500Hz, duration=1s

        # --- Display info ---
        cvzone.putTextRect(img, f'Blink Count: {blinkCounter}', (50, 100), colorR=color)

        if alert_on:
            cvzone.putTextRect(img, "⚠️ DROWSINESS ALERT ⚠️", (180, 200),
                               scale=2, thickness=3, colorR=(0, 0, 255))

        imgPlot = plotY.update(ratioAvg, color)
        img = cv2.resize(img, (640, 360))
        imgStack = cvzone.stackImages([img, imgPlot], 2, 1)
    else:
        img = cv2.resize(img, (640, 360))
        imgStack = cvzone.stackImages([img, img], 2, 1)

    cv2.imshow("Drowsiness Detection", imgStack)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
        break

cap.release()
cv2.destroyAllWindows()
