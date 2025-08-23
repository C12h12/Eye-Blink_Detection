import streamlit as st
import cv2
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector
from cvzone.PlotModule import LivePlot

st.set_page_config(page_title="Drowsiness Detection", layout="wide")
st.title("👁️ Drowsiness Detection App")

# UI control
run_app = st.checkbox("Start Webcam", value=False)
FRAME_WINDOW = st.image([])

# Detector
detector = FaceMeshDetector(maxFaces=1)
plotY = LivePlot(640, 360, [20, 50], invert=True)

ratioList = []
blinkCounter = 0
counter = 0
color = (255, 0, 255)

# Drowsiness variables
drowsy_counter = 0
DROWSY_THRESH = 40
alert_on = False

cap = cv2.VideoCapture(0)

while run_app:
    success, img = cap.read()
    if not success:
        st.warning("⚠️ Failed to access webcam")
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

        # Drowsiness detection
        if ratioAvg < 35:
            drowsy_counter += 1
        else:
            drowsy_counter = 0
            alert_on = False

        if drowsy_counter > DROWSY_THRESH:
            alert_on = True

        # Info display
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

    FRAME_WINDOW.image(imgStack, channels="BGR")

cap.release()
