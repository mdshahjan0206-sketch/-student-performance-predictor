# 🎓 Student Performance Predictor

<p align="center">
  <img src="https://img.shields.io/badge/Project-Student%20Performance%20Predictor-16A765?style=for-the-badge" alt="Project Name"/>
  <img src="https://img.shields.io/badge/AI-Machine%20Learning-blue?style=for-the-badge" alt="Machine Learning"/>
  <img src="https://img.shields.io/badge/Status-Deployed-success?style=for-the-badge" alt="Deployment Status"/>
</p>

<p align="center">
  An AI-powered web application for analysing student data, predicting academic performance, and generating meaningful insights using machine learning.
</p>

<p align="center">
  <a href="https://student-performance-predictor-ay67wvav7-iron-man17.vercel.app/">
    <img src="https://img.shields.io/badge/🚀_Live_Demo-Visit_Website-16A765?style=for-the-badge" alt="Live Demo"/>
  </a>
  <a href="https://student-performance-predictor-f6xy.onrender.com/docs">
    <img src="https://img.shields.io/badge/⚡_API-Documentation-3776AB?style=for-the-badge" alt="API Documentation"/>
  </a>
</p>

---

## 📌 About the Project

**Student Performance Predictor** is a machine-learning-based web application designed to analyse student datasets and predict academic performance.

The application allows users to upload CSV datasets, explore data insights, prepare data, train machine-learning models, generate predictions, and export available results through a web interface.

## ✨ Key Features

- 📂 **Dataset Upload** — Upload student datasets in CSV format.
- 📊 **Data Analysis** — Explore dataset statistics and distributions.
- 🧹 **Data Preparation** — Identify and handle missing values where supported.
- 🤖 **Model Training** — Train machine-learning models on prepared datasets.
- 🎯 **Performance Prediction** — Generate predictions using trained models.
- ⚠️ **At-Risk Identification** — Highlight records that may need further review.
- 📈 **Visual Analytics** — Explore data and model insights.
- 📥 **Export Results** — Download supported results for further analysis.

## 🌐 Live Deployment

| Service | Link |
|---|---|
| 🖥️ Frontend | [Open Website](https://student-performance-predictor-ay67wvav7-iron-man17.vercel.app/) |
| ⚙️ Backend API | [Open Backend](https://student-performance-predictor-f6xy.onrender.com/) |
| 📘 API Documentation | [Explore API Docs](https://student-performance-predictor-f6xy.onrender.com/docs) |

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| ⚛️ React | Frontend user interface |
| ⚡ Vite | Frontend development and build |
| 🔷 TypeScript | Type-safe frontend development |
| 🎨 CSS | Styling and responsive UI |
| 🐍 Python | Backend and data processing |
| 🚀 FastAPI | REST API development |
| 🦄 Uvicorn | ASGI server |
| 🐼 Pandas | Dataset manipulation |
| 🔢 NumPy | Numerical computation |
| 🧠 Scikit-learn | Machine learning |
| 📊 Plotly | Interactive visualisations |
| 🔍 SHAP | Model explainability, where enabled |
| ▲ Vercel | Frontend hosting |
| ☁️ Render | Backend hosting |

## 🔄 Application Workflow

```text
📂 Upload Dataset
       ↓
📊 Analyse Data
       ↓
🧹 Prepare Dataset
       ↓
🤖 Train Model
       ↓
🎯 Generate Predictions
       ↓
📈 Review Results
       ↓
📥 Export Results
```

## 🚀 Getting Started

### 📋 Prerequisites

Install the following tools before running the project locally:

- 🐍 Python
- 🟢 Node.js and npm
- 🐙 Git

### 1️⃣ Clone the Repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd student-performance-predictor
```

Replace the placeholder with your actual GitHub repository URL.

### 2️⃣ Set Up the Backend

```bash
cd backend
python -m venv .venv
```

**Activate the virtual environment on Windows:**

```bat
.venv\Scripts\activate
```

**Install dependencies:**

```bash
pip install -r requirements.txt
```

**Start the backend server:**

```bash
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

🔗 Backend: `http://127.0.0.1:8000`

📘 API documentation: `http://127.0.0.1:8000/docs`

### 3️⃣ Set Up the Frontend

Open a second terminal from the project root:

```bash
cd frontend
npm install
npm run dev
```

🌍 Open the local URL displayed by Vite, usually `http://localhost:5173`.

### 4️⃣ Build for Production

From the frontend directory:

```bash
npm run build
```

📦 The production build is normally generated in `frontend/dist`.

## ☁️ Deployment Guide

### ▲ Frontend — Vercel

- 📁 Root Directory: `frontend`
- 🔨 Build Command: `npm run build`
- 📦 Output Directory: `dist`

### 🚀 Backend — Render

- 📁 Root Directory: `backend`
- 🔨 Build Command: `pip install -r requirements.txt`
- ▶️ Start Command:

```bash
uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

⚠️ Ensure the production frontend routes API requests to the deployed backend and that the configured API paths match the backend routes.

## 🧪 Troubleshooting

| Error | Possible Solution |
|---|---|
| ❌ CSV upload fails | Check CSV format, headers, and required columns. |
| 🔎 API returns 404 | Verify the request URL and API endpoint. |
| 🔥 API returns 500 | Check backend logs on Render. |
| 🔌 Frontend cannot connect | Verify the API URL and Vercel rewrite configuration. |
| 🐍 Backend fails locally | Activate the correct Python environment and install dependencies. |

## 🔐 Privacy & Responsible AI

- 🛡️ Use anonymised or synthetic datasets for testing.
- 🔑 Never commit credentials, `.env` files, or private student records to GitHub.
- 🔒 Protect student information and follow applicable institutional policies.
- ⚖️ Evaluate model performance and potential bias before using predictions.
- 🎓 Treat predictions as decision-support information, not as the sole basis for educational decisions.

## 📁 Project Structure

```text
student-performance-predictor/
│
├── 📂 backend/          # FastAPI and machine-learning backend
├── 📂 frontend/         # React + Vite frontend
├── 📄 .gitignore        # Git ignored files
└── 📘 README.md         # Project documentation
```

*This is a simplified overview; the actual repository may contain additional files and folders.*

## 🔮 Future Enhancements

- 📊 Enhanced model evaluation and reporting
- 🔍 Improved model explainability
- 🔐 Secure user-session isolation
- 💾 Reliable dataset and model persistence
- 📈 Additional visual analytics

## 📄 License

No open-source license is specified. Add a `LICENSE` file if you intend to distribute the project under a particular license.

---

<p align="center">
  <b>🎓 Student Performance Predictor</b>
  <br/>
  Turning student data into actionable academic insights.
  <br/><br/>
  ⭐ If you find this project useful, consider giving the repository a star!
</p>
