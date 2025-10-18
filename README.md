# AI Face Recognition System 👁️‍🗨️

A real-time, browser-based face recognition application built using **TensorFlow.js** and the **BlazeFace** model. This system detects faces via a webcam, extracts key landmarks, and uses a distance-based metric to recognize known individuals or label them as "Unknown."

---

## ✨ Features

* **Real-Time Detection:** Uses the BlazeFace model for high-speed face detection.
* **Persistent Enrollment:** (Placeholder for future feature) The system can enroll new faces from the live video feed and store their unique "embeddings" (normalized landmarks).
* **Custom Distance Metric:** Employs a **normalized Euclidean distance** to compare faces, ensuring recognition is less affected by face size or position relative to the camera.
* **Visual Feedback:** Draws a bounding box and recognition label directly onto the video canvas.
* **Status Display:** Provides clear feedback on model loading, webcam status, and recognition results.

---

## 🚀 Getting Started

Since this project requires access to your webcam and local image files, it **must be run on a local web server**. Opening `index.html` directly will result in browser security errors.

### Prerequisites

You need **Visual Studio Code** and the **Live Server** extension (by Ritwick Dey) to run this project easily.

### 1. Project Setup

1.  **Clone or Download:** Get the project files and create a new directory (e.g., `face-recognition-app`).
2.  **File Structure:** Place all the provided files (`index.html`, `script.js`, `style.css`) into this directory.
3.  **Add Training Images:** Place your known face images (`Gaurav.jpg`, `Junaid12.jpg`, `4.jpg`, etc.) into the **same directory**.

Your final structure should look like this:
Face-Recognition-Web-Application/ ├── index.html ├── script.js ├── style.css ├── Junaid12.jpg └── ... (other training images)

### 2. Running Locally with VS Code

1.  Open the `face-recognition-app` folder in VS Code.
2.  Right-click on `index.html`.
3.  Select **"Open with Live Server"**.

The project will open in your browser (e.g., `http://127.0.0.1:5500/index.html`). Grant camera access when prompted.

---

## ⚙️ Configuration

The core recognition logic depends on the `THRESHOLD` value, which dictates how "strict" the system is.

### Adjusting Strictness (`script.js`)

In `script.js`, the `THRESHOLD` constant controls the acceptance level.

| Value | Effect | Rationale |
| :--- | :--- | :--- |
| **Lower Value** (e.g., `1.5` to `2.0`) | **Strict** 🔒 | Faces must be very close to an enrolled face to be recognized. Recommended to prevent false positives (detecting strangers as known users). |
| **Higher Value** (e.g., `3.0` to `5.0`) | **Permissive** 🔓 | Allows greater variation in lighting, expression, or head angle. May increase false positives. |

The current code uses a highly normalized distance metric, so the **default threshold is set to `2.0`**.

---

## 🧑‍💻 How to Enroll New Faces

The application includes a working enrollment feature to add faces dynamically:

1.  Ensure the person to be enrolled is facing the camera, and **only their face is visible** in the frame.
2.  Click the **"Capture Face for Enrollment"** button.
3.  A dialog will prompt you to **enter the name** for the new face.
4.  The system will capture the face's landmarks, normalize them, and immediately add the new face to the recognition database.

⚠️ **Note:** Faces enrolled via the button are stored in **browser memory (RAM)**. They will be **lost** when the page is refreshed or the server is restarted.

---

## 📚 Technical Details

The recognition pipeline relies on normalized Euclidean distance for comparison:

1.  **BlazeFace Landmarks:** The model provides 6 key points (eyes, nose, mouth corners, ear centers).
2.  **Normalization:** The landmarks are transformed to make them invariant to translation (centered on the nose) and scale (divided by the distance between the eyes).
3.  **Distance Calculation:** The **L2 Norm (Euclidean distance)** is calculated between the normalized 12-dimensional vector of the live face and every known normalized embedding.
4.  **Classification:** The closest match is found. If its distance is less than `THRESHOLD`, it's recognized; otherwise, it's labeled "Unknown."

---

## 🔗 Dependencies

This project uses CDN links to load external libraries:

* **TensorFlow.js:** The core machine learning library.
* **@tensorflow-models/blazeface:** The pre-trained model used for fast face detection.
