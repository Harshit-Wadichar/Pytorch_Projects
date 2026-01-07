# 🎓 Student Exam Score Predictor

An AI-powered Full Stack Web Application that predicts a student's potential exam score based on their study habits and attendance.

![Project Status](https://img.shields.io/badge/Status-Completed-success)

## 🌟 Overview

This project demonstrates a complete end-to-end Machine Learning pipeline integrated into a modern web application. It uses **PyTorch** for the neural network, **FastAPI** for the backend API, and **React** with **Tailwind CSS** for the frontend user interface.

### Key Features
*   **Predictive Model**: Regression Neural Network trained on student performance data.
*   **Real-time Prediction**: Instant score estimates via the web interface.
*   **Interactive UI**: Clean, responsive design for easy data entry.
*   **Rest API**: Fully documented API endpoints for model inference.

---

## 🏗️ Architecture

The application is split into two main components:

### 1. Backend (`/backend`)
*   **Technology**: Python, FastAPI, PyTorch, Pandas, Scikit-Learn.
*   **Functionality**:
    *   `model.py`: Defines the Neural Network structure (4 inputs -> 1 output).
    *   `train.py`: Handles data loading, preprocessing, training loops, and validation logic.
    *   `preprocessing.py`: Loads the dataset, normalizes features, and creates PyTorch DataLoaders.
    *   `main.py`: Serves the trained model via a REST API endpoint (`/predict`).

### 2. Frontend (`/frontend`)
*   **Technology**: React (Vite), Tailwind CSS.
*   **Functionality**:
    *   Provides a form to input Study Hours, Sleep Hours, Attendance %, and Previous Score.
    *   Communicates with the Backend API to fetch predictions.
    *   Displays the predicted exam score dynamically.

---

## 🚀 Getting Started

Follow these instructions to run the project locally.

### Prerequisites
*   Python 3.8+
*   Node.js & npm

### Step 1: Backend Setup
1.  Navigate to the backend directory:
    ```bash
    cd backend
    ```
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  Train the model (Optional, pre-trained `model.pth` is included):
    ```bash
    python train.py
    ```
4.  Start the server:
    ```bash
    uvicorn main:app --reload
    ```
    The API will run at `http://localhost:8000`.

### Step 2: Frontend Setup
1.  Open a new terminal and navigate to the frontend directory:
    ```bash
    cd frontend
    ```
2.  Install dependencies:
    ```bash
    npm install
    ```
3.  Start the application:
    ```bash
    npm run dev
    ```
    The App will open at `http://localhost:5173`.

---

## 🧠 How It Works (For Developers)

### The AI Model
The model is a Feed-Forward Neural Network consisting of:
*   **Input Layer**: Accepts 4 features (Study Hours, Sleep Hours, Attendance, Previous Score).
*   **Hidden Layers**: Linear transformations with ReLU activation.
*   **Output Layer**: A single linear unit predicting the Score (0-100).

### The API Bridge
**FastAPI** acts as the bridge between the Python model and the React frontend.
1.  **Request**: React sends a JSON object with user inputs.
2.  **Validation**: Pydantic ensures data types are correct.
3.  **Inference**: PyTorch calculates the prediction.
4.  **Response**: The server returns the predicted score.

---

## 📝 License
This project is for educational purposes.
