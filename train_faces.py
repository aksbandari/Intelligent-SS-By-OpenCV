import cv2
import os
import numpy as np

dataset_path = "dataset"
BASE_DIR = os.path.dirname(__file__)
dataset_path = os.path.join(BASE_DIR, "dataset")
recognizer = cv2.face.LBPHFaceRecognizer_create()

faces = []
labels = []
label_map = {}
current_id = 0

for person in os.listdir(dataset_path):
    person_path = os.path.join(dataset_path, person)
    if person not in label_map:
        label_map[person] = current_id
        current_id += 1

    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        faces.append(img)
        labels.append(label_map[person])

recognizer.train(faces, np.array(labels))
recognizer.save("face_trainer.yml")

print("✅ Face training completed")
