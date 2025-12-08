**Face Recognition Attendance 📸🚀**

A lightweight Flask + OpenCV + face_recognition app that uses your webcam for real-time face recognition and records attendance.
This project automates marking attendance, saving recognized / unknown face images, and exporting per-session CSVs — all from a simple command you run locally.# face_recognition

Quick Demo (Example)
======== FACE RECOGNITION ATTENDANCE ========

✅ Loaded classes: ['alice', 'bob']
✅ Encoding Complete
✅ Application started successfully!

**🎥 Starting webcam for real-time face recognition...**
(An OpenCV window pops up titled "Face Recognition")
When a known face appears:
  - Green box + NAME shown on screen
  - Cropped image saved to `recognized_faces/`
  - Name & time appended to `Attendance.csv`

Press 'q' in the window to stop and return to the Flask route.
Visit: http://127.0.0.1:5000/open-csv  ->  downloads session CSV

**🚀 Features** 

**🧾 Generate Attendance** — Detect faces live and mark attendance automatically.

**📜 View Previous Invoices** — Persistent Attendance.csv stores recognized names & times

**🔎 Search / Filter** — search through CSVs by name or timestamp.

**💾 Save Images** — Saves cropped images for recognized faces (recognized_faces/) and unknown faces (unknown_faces/).

**🗑️ Delete Old Records** — delete or prune old images / CSV rows.

**⚙️ Menu-driven / Simple Controls** — Use the OpenCV UI and Flask endpoints to control sessions.

**📥 Installation & Setup**
**1️⃣ Clone the Repository**
git clone https://github.com/<your-username>/face-recognition-attendance.git
cd face-recognition-attendance

**2️⃣ Create and Activate Virtual Environment**

**Windows**

python -m venv venv
venv\Scripts\activate


**Linux/macOS**

python3 -m venv venv
source venv/bin/activate

**3️⃣ Install Dependencies**
pip install -r requirements.txt


If dlib fails, install via Conda or prebuilt wheels.

**📚 requirements.txt**
Flask>=2.0
opencv-python>=4.5
numpy>=1.21
pandas>=1.3
face_recognition>=1.3.0

**🖥️ Running the Application**
python app.py

**📍 Steps:**

Server starts

Go to:

http://127.0.0.1:5000/


Webcam opens

Recognized faces → marked & saved

Press Q to exit the webcam window

Download session attendance:

http://127.0.0.1:5000/open-csv

**🧠 Concepts Used** — Face embeddings (face_recognition), OpenCV webcam capture, Pandas file I/O, Flask routes.
