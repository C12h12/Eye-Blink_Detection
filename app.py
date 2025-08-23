import streamlit as st
import cv2
import numpy as np
from cvzone.FaceMeshModule import FaceMeshDetector
from cvzone.PlotModule import LivePlot
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode, RTCConfiguration

st.set_page_config(page_title="Drowsiness Detection", layout="wide")
st.title("👁️ Drowsiness Detection App (WebRTC)")

# Detector
detector = FaceMeshDetector(maxFaces=1)

# STUN + Twilio TURN
RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [
        {"urls": "stun:stun.l.google.com:19302"},
        {
            "urls": "turn:global.turn.twilio.com:3478?transport=udp",
            "username": "guest",
            "credential": "somepassword"
        }
    ]
})


class DrowsinessProcessor(VideoProcessorBase):
    def __init__(self):
        self.ratioList = []
        self.blinkCounter = 0
        self.counter = 0
        self.color = (255, 0, 255)
        self.drowsy_counter = 0
        self.alert_on = False
        self.plotY = LivePlot(640, 360, [20, 50], invert=True)

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr")
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
            self.ratioList.append(ratio)
            if len(self.ratioList) > 3:
                self.ratioList.pop(0)
            ratioAvg = sum(self.ratioList) / len(self.ratioList)

            # Blink detection
            if ratioAvg < 35 and self.counter == 0:
                self.blinkCounter += 1
                self.color = (0, 200, 0)
                self.counter = 1
            if self.counter != 0:
                self.counter += 1
                if self.counter > 10:
                    self.counter = 0
                    self.color = (255, 0, 255)

            # Drowsiness detection
            if ratioAvg < 35:
                self.drowsy_counter += 1
            else:
                self.drowsy_counter = 0
                self.alert_on = False

            if self.drowsy_counter > 40:
                self.alert_on = True

            # Info display
            cv2.putText(img, f'Blink Count: {self.blinkCounter}', (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, self.color, 2)
            if self.alert_on:
                cv2.putText(img, "⚠️ DROWSINESS ALERT ⚠️", (180, 200),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

            imgPlot = self.plotY.update(ratioAvg, self.color)
            img = cv2.resize(img, (640, 360))
            imgStack = np.hstack([img, imgPlot])
        else:
            img = cv2.resize(img, (640, 360))
            imgStack = np.hstack([img, img])

        return imgStack


# WebRTC streamer
webrtc_streamer(
    key="drowsiness-detection",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    video_processor_factory=DrowsinessProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True
)
