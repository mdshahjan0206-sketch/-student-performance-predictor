# Student Performance Predictor

An AI-powered web application for analysing student datasets, training machine learning models, predicting academic performance, and identifying students who may need additional academic support.

<p align="center">
  <a href="https://student-performance-predictor-ay67wvav7-iron-man17.vercel.app/">Live Demo</a> |
  <a href="https://student-performance-predictor-f6xy.onrender.com/docs">API Documentation</a>
</p>

## Overview

Student Performance Predictor transforms student data into meaningful insights using data analysis and machine learning. It provides a workflow for uploading datasets, analysing data, training models, generating predictions, and exporting results.

## Features

- **Dataset Upload:** Upload student datasets in CSV format.
- **Data Analysis:** Explore dataset statistics, features, and distributions.
- **Data Preparation:** Identify and handle missing values where supported.
- **Machine Learning:** Train models using prepared datasets.
- **Performance Prediction:** Generate predictions using trained models.
- **At-Risk Identification:** Highlight students who may need further academic review.
- **Visual Analytics:** Explore data and model insights through visualisations.
- **Results Export:** Download supported results for further analysis.

## Live Application

| Component | URL |
|---|---|
| Frontend | https://student-performance-predictor-ay67wvav7-iron-man17.vercel.app/ |
| Backend API | https://student-performance-predictor-f6xy.onrender.com/ |
| API Documentation | https://student-performance-predictor-f6xy.onrender.com/docs |

## Technology Stack

| Layer | Technologies |
|---|---|
| Frontend | React, Vite, TypeScript, CSS |
| Backend | Python, FastAPI, Uvicorn |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Data Visualisation | Plotly |
| Model Explainability | SHAP, where enabled |
| Deployment | Vercel and Render |

## Application Workflow

1. Upload a supported student dataset.
2. Analyse the dataset and inspect its features.
3. Prepare the data and handle missing values when necessary.
4. Train a machine learning model.
5. Generate student performance predictions.
6. Review and export the available results.

## Getting Started

### Prerequisites

- Python
- Node.js and npm
- Git

### 1. Clone the Repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd student-performance-predictor
```

Replace the placeholder with your actual GitHub repository URL.

### 2. Set Up the Backend

```bash
cd backend
python -m venv .venv
```

Activate the virtual environment on Windows:

```bat
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the backend server:

```bash
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Backend URL: `http://127.0.0.1:8000`

API documentation: `http://127.0.0.1:8000/docs`

### 3. Set Up the Frontend

Open a second terminal from the project root:

```bash
cd frontend
npm install
npm run dev
```

Open the local URL displayed in the terminal, usually `http://localhost:5173`.

### 4. Build for Production

From the frontend directory:

```bash
npm run build
```

The production build is normally generated in `frontend/dist`.

## Deployment

### Frontend — Vercel

- Root Directory: `frontend`
- Build Command: `npm run build`
- Output Directory: `dist`

### Backend — Render

- Root Directory: `backend`
- Build Command: `pip install -r requirements.txt`
- Start Command:

```bash
uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

Ensure that production API routing points to the deployed backend and that API paths match the routes defined by the application.

## Troubleshooting

| Issue | Suggested Solution |
|---|---|
| CSV upload fails | Verify the CSV format, headers, and required columns. |
| API returns 404 | Check the request URL and confirm the endpoint exists. |
| API returns 500 | Review the backend logs on Render. |
| Frontend cannot connect to backend | Verify the production API URL and Vercel rewrite configuration. |
| Backend fails locally | Activate the correct Python environment and install the required dependencies. |

## Privacy and Responsible Use

- Use anonymised or synthetic datasets during testing.
- Never commit private student records, credentials, `.env` files, or generated exports to GitHub.
- Protect student information and follow applicable institutional privacy policies.
- Evaluate model performance and potential bias before using predictions in practice.
- Treat predictions as decision-support information, not as the sole basis for educational decisions.

## Project Structure

```text
student-performance-predictor/
├── backend/
├── frontend/
├── .gitignore
└── README.md
```

This is a simplified overview; the actual repository may contain additional files and directories.

## Future Improvements

- Enhanced model evaluation and explainability.
- Improved dataset validation and reporting.
- Secure data persistence and user-session isolation.
- Additional visual analytics and prediction insights.

## License

No open-source license is currently specified. Add a `LICENSE` file if you intend to distribute the project under a particular open-source license.

---

**Student Performance Predictor**  
*Turning student data into actionable academic insights.*
