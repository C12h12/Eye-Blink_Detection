import streamlit as st
import cv2
import numpy as np
from cvzone.FaceMeshModule import FaceMeshDetector
import time

st.set_page_config(page_title="Drowsiness Detection", layout="wide")
st.title("👁️ Drowsiness Detection App (Cloud Webcam)")

detector = FaceMeshDetector(maxFaces=1)
ratioList = []
drowsy_counter = 0
alert_on = False

frame_placeholder = st.empty()

# ✅ Camera widget declared ONCE
img_file = st.camera_input("Camera")

if img_file is not None:
    # Continuous refresh (simulated live feed)
    bytes_data = img_file.getvalue()
    img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    img = cv2.resize(img, (640, 480))

    img, faces = detector.findFaceMesh(img, draw=False)

    if faces:
        face = faces[0]
        leftUp, leftDown, leftLeft, leftRight = face[159], face[23], face[130], face[243]
        lengthVer, _ = detector.findDistance(leftUp, leftDown)
        lengthHor, _ = detector.findDistance(leftLeft, leftRight)
        ratio = int((lengthVer / lengthHor) * 100)
        ratioList.append(ratio)
        if len(ratioList) > 5:
            ratioList.pop(0)
        ratioAvg = sum(ratioList) / len(ratioList)

        if ratioAvg < 35:
            drowsy_counter += 1
        else:
            drowsy_counter = 0
            alert_on = False

        if drowsy_counter > 5:
            alert_on = True

        cv2.putText(img, f'Blink Ratio: {int(ratioAvg)}', (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

        if alert_on:
            cv2.putText(img, "⚠️ DROWSINESS ALERT ⚠️", (100, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    frame_placeholder.image(img, channels="BGR")
