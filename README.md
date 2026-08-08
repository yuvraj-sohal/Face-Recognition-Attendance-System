# Face Recognition Attendance System (CLI)

A Python-based attendance system that uses computer vision and machine learning techniques to recognize registered users and record their attendance automatically.The Face Recognition Attendance System is a computer vision project developed using Python and OpenCV.
The system follows a complete face-recognition pipeline:

                Collect Face Images
                        ↓
                Create Dataset
                        ↓
                Train Recognition Model
                        ↓
                Recognize Face
                        ↓
                Mark Attendance
                        ↓
                Store Attendance Record

The project is designed to automate attendance recording while reducing the need for manual entry.

## Key Features

- Capture facial images for registered users
- Create individual face datasets
- Train a face recognition model
- Detect and recognize faces through a camera
- Automatically identify registered users
- Record attendance with date and time
- Store attendance records in CSV format
- Maintain a mapping between user IDs and names
- Separate scripts for data collection, model training, and attendance recognition

## Models Used

The system uses two main computer vision models as part of the face recognition pipeline.
### 1. Haar Cascade Classifier
The Haar Cascade Classifier is used for **face detection**.
It identifies faces in the camera frames and determines the location of detected faces before they are passed to the recognition stage.
The classifier is used during the face data collection and recognition processes.
### 2. Local Binary Patterns Histograms (LBPH)
The **LBPH Face Recognizer** is used for **face recognition**.
It is trained using the collected facial images and assigns a user ID to a detected face based on the learned facial patterns.
The recognition result is then used to identify the corresponding user and record attendance.


## Project Workflow

### 1. Face Data Collection

The `collect_faces.py` script is used to capture facial images for users.

```bash
python src/collect_faces.py
```

Captured images are organized according to the user's ID.
The actual face dataset is intentionally excluded from this public repository because it contains biometric data.

### 2. Model Training

After collecting the required face images, the recognition model can be trained using:

```bash
python src/train_model.py
```

The trained model is generated locally.
The trained model file is excluded from the public repository and should be generated locally using the training script.

### 3. Face Recognition and Attendance

Run the recognition and attendance system using:

```bash
python src/recognize_attendance.py
```

The system uses the connected camera to detect and recognize registered users and records their attendance.

## Technology Stack

| Technology | Purpose |
|------------|---------|
| **Python** | Programming language |
| **OpenCV** | Face detection and recognition |
| **NumPy** | Numerical operations |
| **pandas** | Attendance data handling |
| **Pillow** | Image processing |
| **CSV** | Attendance data storage |

## Project Structure

### `src/collect_faces.py`

Responsible for collecting facial images for registered users.

### `src/train_model.py`

Responsible for training the face recognition model using the collected dataset.

### `src/recognize_attendance.py`

Responsible for recognizing users through the camera and recording attendance.

## Attendance Records

Attendance records are generated locally as CSV files.

Example:

```text
attendance/
└── attendance_YYYY-MM-DD.csv
```

Each generated attendance file is intentionally excluded from the public repository.

## Privacy

This project works with facial images, which can contain sensitive biometric information.
For this reason, the public repository does not include:
- Personal face datasets
- Attendance records
- Personal user-name mappings
- Locally generated trained model files
Users should create and use their own dataset when running the project.

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yuvraj-sohal/Face-Recognition-Attendance-System.git
cd Face-Recognition-Attendance-System
```

### 2. Create a Virtual Environment

Windows:

```bash
python -m venv ML
```

Activate the virtual environment:

```bash
ML\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## Running the Project

### Step 1 — Collect Face Data

Run:

```bash
python src/collect_faces.py
```

Use the application to collect the required facial images for each registered user.

### Step 2 — Train the Model

After collecting the face dataset, run:

```bash
python src/train_model.py
```

This generates the trained recognition model locally.

### Step 3 — Start Attendance Recognition

Run:

```bash
python src/recognize_attendance.py
```

The system accesses the connected camera, detects and recognizes registered users, and records their attendance.
Make sure your computer has a working camera before starting the recognition system.

## Data Storage

The project uses CSV files to store attendance records.
Attendance files are generated locally using the date-based naming format:

```text
attendance_YYYY-MM-DD.csv
```

**Academic / Machine Learning & Computer Vision Project**

This project demonstrates practical implementation of computer vision, face recognition, machine learning, image processing, and automated attendance recording using Python.

## Developer

### Yuvraj Singh

B.Tech Computer Science & Engineering

GitHub: [@yuvraj-sohal](https://github.com/yuvraj-sohal)

## License

This project is licensed under the **MIT License**.

See the [LICENSE](LICENSE) file for details.
