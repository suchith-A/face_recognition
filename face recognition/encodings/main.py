from flask import Flask, make_response
import cv2
import numpy as np
import face_recognition
import os
import pandas as pd
from datetime import datetime, timedelta

# ================================
# Flask App Initialization
# ================================
app = Flask(__name__, template_folder=r"C:\Users\suchi\OneDrive\Desktop\face recognition\face recognition\template")

# ================================
# Base Directory
# ================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ================================
# Paths
# ================================
known_faces_path = os.path.join(BASE_DIR, 'data_setofatt')
recognized_faces_path = os.path.join(BASE_DIR, 'recognized_faces')
unknown_faces_path = os.path.join(BASE_DIR, 'unknown_faces')
attendance_file = os.path.join(BASE_DIR, 'Attendance.csv')

# Ensure folders exist
 
os.makedirs(recognized_faces_path, exist_ok=True)
os.makedirs(unknown_faces_path, exist_ok=True)
os.makedirs(known_faces_path, exist_ok=True)

# Load Known Faces

images = []
classNames = []
for cl in os.listdir(known_faces_path):
    curImg = cv2.imread(os.path.join(known_faces_path, cl))
    if curImg is not None:
        images.append(curImg)
        classNames.append(os.path.splitext(cl)[0])
print(f"✅ Loaded classes: {classNames}")

def findEncodings(images):
    encodeList = []
    for img in images:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encs = face_recognition.face_encodings(img_rgb)
        if encs:
            encodeList.append(encs[0])
        else:
            print("⚠️ No face found in one image, skipped.")
    return encodeList

encodeListKnown = findEncodings(images)
print('✅ Encoding Complete')

# ================================
# Variables
# ================================
cap = cv2.VideoCapture(0)
recognized_faces = set()
last_unknown_save_time = datetime.min
session_attendance = []

# ================================
# Attendance Function
# ================================
def markAttendance(name):
    now = datetime.now()
    dtString = now.strftime('%d:%m:%Y %H:%M:%S')
    if not os.path.isfile(attendance_file):
        with open(attendance_file, 'w') as f:
            f.write('Name,Time\n')
    df = pd.read_csv(attendance_file)
    if name not in df['Name'].values:
        new_row = pd.DataFrame({'Name':[name], 'Time':[dtString]})
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv(attendance_file, index=False)

# ================================
# Routes
# ================================
@app.route('/')
def start_attendance():
    global recognized_faces, last_unknown_save_time, session_attendance
    print("🎥 Starting webcam for real-time face recognition...")

    while True:
        success, img = cap.read()
        if not success:
            print("⚠️ Failed to grab frame from camera.")
            break

        # Resize frame for faster detection (scale down by 1/4)
        small_img = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
        rgb_small_img = cv2.cvtColor(small_img, cv2.COLOR_BGR2RGB)

        # Detect faces and encodings
        face_locations = face_recognition.face_locations(rgb_small_img)
        encodings = face_recognition.face_encodings(rgb_small_img, face_locations)

        for encode, loc in zip(encodings, face_locations):
            matches = face_recognition.compare_faces(encodeListKnown, encode, tolerance=0.5)
            face_dis = face_recognition.face_distance(encodeListKnown, encode)
            best_match_index = np.argmin(face_dis) if len(face_dis) > 0 else None

            y1, x2, y2, x1 = [v*4 for v in loc]  # Scale back to original frame

            if best_match_index is not None and matches[best_match_index]:
                name = classNames[best_match_index].upper()
                if name not in recognized_faces:
                    recognized_faces.add(name)
                    markAttendance(name)
                    recognized_filename = os.path.join(
                        recognized_faces_path,
                        f"{name}_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
                    )
                    cv2.imwrite(recognized_filename, img[y1:y2, x1:x2])
                    session_attendance.append((name, datetime.now().strftime('%d:%m:%Y %H:%M:%S')))

                # Draw rectangle & name
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(img, (x1, y2-35), (x2, y2), (0, 255, 0), cv2.FILLED)
                cv2.putText(img, name, (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 2)
            else:
                # Unknown face, save only every 2 seconds
                if (datetime.now() - last_unknown_save_time) > timedelta(seconds=2):
                    unknown_filename = os.path.join(
                        unknown_faces_path,
                        f"unknown_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
                    )
                    cv2.imwrite(unknown_filename, img[y1:y2, x1:x2])
                    last_unknown_save_time = datetime.now()

                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.rectangle(img, (x1, y2-35), (x2, y2), (0, 0, 255), cv2.FILLED)
                cv2.putText(img, "Unknown", (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 2)

        cv2.imshow('Face Recognition', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return "✅ Attendance process completed."

@app.route('/open-csv')
def open_csv():
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    csv_file_name = os.path.join(BASE_DIR, f'Attendance_{timestamp}.csv')
    with open(csv_file_name, 'w') as f:
        f.write("Name,Time\n")
        for name, time in session_attendance:
            f.write(f"{name},{time}\n")
    with open(csv_file_name, 'r') as f:
        content = f.read()
    response = make_response(content)
    response.headers["Content-Type"] = "text/csv"
    response.headers["Content-Disposition"] = f"inline; filename=Attendance_{timestamp}.csv"
    return response

# ================================
# Run App
# ================================
if __name__ == "__main__":
    print("✅ Application started successfully!")
    app.run(debug=False)
