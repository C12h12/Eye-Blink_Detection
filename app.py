import streamlit as st
import cv2
import numpy as np
from cvzone.FaceMeshModule import FaceMeshDetector
import time

st.set_page_config(page_title="Drowsiness Detection", layout="wide")
st.title("👁️ Drowsiness Detection App (Simple Deployment)")

# Detector
detector = FaceMeshDetector(maxFaces=1)

# For blink/drowsiness logic
ratioList = []
blinkCounter = 0
drowsy_counter = 0
alert_on = False

# Camera input (Streamlit native, simpler than webrtc)
st.write("📷 Take a snapshot or keep refreshing for detection:")
img_file_buffer = st.camera_input("Camera")

if img_file_buffer is not None:
    # Convert captured frame to numpy
    bytes_data = img_file_buffer.getvalue()
    img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

    img = cv2.resize(img, (640, 480))
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
        if len(ratioList) > 5:
            ratioList.pop(0)
        ratioAvg = sum(ratioList) / len(ratioList)

        # Drowsiness check
        if ratioAvg < 35:
            drowsy_counter += 1
        else:
            drowsy_counter = 0
            alert_on = False

        if drowsy_counter > 5:
            alert_on = True

        # Display results
        cv2.putText(img, f'Blink Ratio: {int(ratioAvg)}', (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

        if alert_on:
            cv2.putText(img, "⚠️ DROWSINESS ALERT ⚠️", (100, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    # Show image in Streamlit
    st.image(img, channels="BGR")
