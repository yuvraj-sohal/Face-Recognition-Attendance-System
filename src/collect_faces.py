import cv2
import os

face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

user_id = input("Enter user id (numeric): ")
user_name = input("Enter user name: ")

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
dataset_path = os.path.join(BASE_DIR, "dataset", f"user_{user_id}")
os.makedirs(dataset_path, exist_ok=True)

cap = cv2.VideoCapture(0)
count = 0

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        count += 1
        cv2.imwrite(f"{dataset_path}/{count}.jpg", gray[y:y+h, x:x+w])
        cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)

    cv2.imshow("Collecting Faces", frame)

    if cv2.waitKey(1) == 27 or count >= 100:
        break

cap.release()
cv2.destroyAllWindows()

print("Face samples collected successfully!")
