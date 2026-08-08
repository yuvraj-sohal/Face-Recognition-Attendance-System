import cv2, pandas as pd, os, csv
from datetime import datetime

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
model_path = os.path.join(BASE_DIR, "trainer", "trainer.yml")
today = datetime.now().strftime("%Y-%m-%d")
attendance_path = os.path.join(BASE_DIR, "attendance", f"attendance_{today}.csv")
user_map_path = os.path.join(BASE_DIR, "user_map.csv")

# Recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
print("Loading model from:", model_path)
recognizer.read(model_path)

# Names
name_dict = {}
with open(user_map_path, "r") as f:
    for row in csv.reader(f):
        if len(row) >= 2:
            name_dict[int(row[0])] = row[1]

# Face detector
face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Camera
cap = cv2.VideoCapture(0)
attendance = {}

print("Press ESC to exit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    status_msg = ""

    for (x, y, w, h) in faces:

        id_, confidence = recognizer.predict(gray[y:y+h, x:x+w])
        label = "Unknown"

        if confidence < 75:
            name = name_dict.get(id_, "Unknown")

            if name != "Unknown":
                label = f"{name} ({round(confidence,1)})"

                today = datetime.now().strftime("%Y-%m-%d")
                now = datetime.now().strftime("%H:%M:%S")
                key = f"{name}_{today}"

                if key not in attendance:
                    attendance[key] = [name, today, now]
                    status_msg = f"Attendance Marked: {name}"
                else:
                    status_msg = f"Recognized: {name}"

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(frame, label, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

    if status_msg:
        cv2.putText(frame, status_msg, (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

    cv2.putText(frame, "Press ESC to Exit", (10,460),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    cv2.imshow("Face Recognition Attendance", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()

# Save attendance
os.makedirs(os.path.join(BASE_DIR, "attendance"), exist_ok=True)

new_df = pd.DataFrame(list(attendance.values()), columns=["Name", "Date", "Time"])

if os.path.exists(attendance_path):

    try:
        old_df = pd.read_csv(attendance_path)

    except:
        old_df = pd.DataFrame(columns=["Name", "Date", "Time"])

    final_df = pd.concat([old_df, new_df]).drop_duplicates(
        subset=["Name", "Date"],
        keep="first"
    )

    final_df.to_csv(attendance_path, index=False)

else:
    new_df.to_csv(attendance_path, index=False)

print("Attendance saved successfully!")