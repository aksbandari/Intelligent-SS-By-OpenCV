import cv2
import numpy as np
import pyaudio
import struct
import os
import uuid
import subprocess

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "face_trainer.yml")

def add_new_face(face_img, recognizer):
    name = input("Enter name for new face (or press Enter to skip): ").strip()
    if name == "":
        print("Skipped adding face")
        return

    dataset_dir = os.path.join(BASE_DIR, "dataset", name)
    os.makedirs(dataset_dir, exist_ok=True)

    img_name = f"{uuid.uuid4().hex}.jpg"
    img_path = os.path.join(dataset_dir, img_name)
    cv2.imwrite(img_path, face_img)

    print(f"Saved new face for {name}")
    print("Retraining model...")

    subprocess.run(["python", "train_faces.py"], check=True)

    recognizer.read(MODEL_PATH)
    print("Model updated and reloaded successfully")


# ----------------- LOAD MODELS -----------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(MODEL_PATH)

# ----------------- CAMERA -----------------
cap = cv2.VideoCapture(0)
ret, prev_frame = cap.read()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

# ----------------- NOISE DETECTION -----------------
audio = pyaudio.PyAudio()
stream = audio.open(format=pyaudio.paInt16, channels=1,
                    rate=44100, input=True, frames_per_buffer=1024)

def detect_noise(threshold=600):
    data = stream.read(1024, exception_on_overflow=False)
    rms = np.sqrt(np.mean(np.square(struct.unpack("h"*1024, data))))
    return rms > threshold

# ----------------- MAIN LOOP -----------------
while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    diff = cv2.absdiff(prev_gray, gray)
    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    motion = np.sum(thresh)

    if motion > 500000:
        cv2.putText(frame, "⚠ Motion Detected", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    unknown_face = None

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        label, confidence = recognizer.predict(face)

        if confidence < 70:
            name = "Authorized"
        else:
            name = "Unknown"
            unknown_face = face
            cv2.putText(frame, "Press A to Add Face", (x, y+h+30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

        cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)
        cv2.putText(frame, name, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

    if detect_noise():
        cv2.putText(frame, "⚠ Noise Alert", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

    cv2.imshow("Intelligent Security System", frame)
    prev_gray = gray

    key = cv2.waitKey(1) & 0xFF
    if key == ord('a') and unknown_face is not None:
        add_new_face(unknown_face, recognizer)
    elif key == ord('q'):
        break

# ----------------- CLEANUP -----------------
cap.release()
cv2.destroyAllWindows()
stream.stop_stream()
stream.close()
audio.terminate()
