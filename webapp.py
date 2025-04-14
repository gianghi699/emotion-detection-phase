import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained 4-class emotion model
model = load_model("emotion_model_4classes.h5")
emotion_labels = ['Angry', 'Happy', 'Sad', 'Neutral']

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Define the video processor class
class EmotionProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y + h, x:x + w]
            try:
                roi_gray = cv2.resize(roi_gray, (48, 48))
            except:
                continue
            roi = roi_gray.astype("float") / 255.0
            roi = np.expand_dims(roi, axis=0)
            roi = np.expand_dims(roi, axis=-1)

            preds = model.predict(roi, verbose=0)[0]
            emotion = emotion_labels[np.argmax(preds)]

            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Streamlit UI
st.title("Real-Time Emotion Detection (4 Emotions)")
st.markdown("Detects: Angry, Happy, Sad, Neutral")

webrtc_streamer(
    key="emotion-detect-4class",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=EmotionProcessor,
    media_stream_constraints={"video": True, "audio": False},
)
